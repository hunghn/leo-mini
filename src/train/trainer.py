"""
3-Stage Trainer for LEO-MINI (Appendix A.2, Table 6).

Stage 1 — Warmup Visual Projector
  Frozen:    LLM, MMoE-Vision
  Trainable: VisualProjector
  Data:      EAGLE alignment dataset

Stage 2 — Full Supervised Fine-tuning
  Frozen:    —
  Trainable: all (LLM + MMoE-Vision + Projector)
  Data:      EAGLE SFT dataset

Stage 3 — Token Reduction Fine-tuning (core contribution)
  Frozen:    LLM backbone, MMoE-Vision
  Trainable: CoTR + MMoE-LLM (LoRA experts + routers)
  Data:      LLaVA-v1.5 665K
  Loss:      Cross-entropy + L_balance (λ=0.05)

Uses HuggingFace Trainer with DeepSpeed Zero2 config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from ..data.dataset import EAGLEDataset, LLaVADataset
from ..data.collator import LeoMiniCollator
from ..data.vitextvqa_dataset import ViTextVQADataset
from ..models.leo_mini import LeoMini
from ..models.vision_experts import Pix2StructExpert
from ..utils.logger import ViLeoMiniLogger, LeoMiniLoggingCallback, make_run_id


# ---------------------------------------------------------------------------
# Training argument dataclass
# ---------------------------------------------------------------------------

@dataclass
class LeoMiniTrainingArgs:
    # --- Paths ---
    llm_path:              str = "meta-llama/Meta-Llama-3-8B-Instruct"
    eagle_data_path:       str = ""   # alignment JSON  (Stage 1)
    eagle_sft_path:        str = ""   # full SFT JSON   (Stage 2)
    llava_data_path:       str = ""   # LLaVA-v1.5 JSON (Stage 3)
    image_dir:             str = ""   # root for image files
    output_dir:            str = "checkpoints"
    resume_from:           Optional[str] = None   # checkpoint to resume from
    # projector_path: path to projector_weights.pt saved at end of Stage 1 or 2.
    # Required for Stage 3 so that trained projector weights are not discarded.
    projector_path:        Optional[str] = None
    pix2struct_model_name: str = "google/pix2struct-large"

    # --- Vi-LEO-MINI paths (ViTextVQA variant) ---
    # When vi_train_path is set, the trainer uses ViTextVQADataset instead of
    # EAGLE/LLaVA for all three stages. Leave empty to use original datasets.
    vi_train_path:  str = ""   # ViTextVQA HuggingFace split name ("train")
    vi_val_path:    str = ""   # ViTextVQA split for mid-training eval ("validation")
    vi_image_dir:   str = ""   # unused (images come from HF), kept for compatibility
    vi_cache_dir:   str = ""   # local HF cache dir (empty = HF default)

    # Vision expert subset — e.g. ["clip", "pix2struct"] for Vi-LEO-MINI.
    # Empty list means use the original 4-expert set.
    vision_experts: List[str] = field(default_factory=list)

    # Log dir for structured JSON logs and plots
    log_dir:        str = "logs"

    # --- Architecture ---
    n_visual:     int = 64
    d_proj_cotr:  int = 256
    lora_rank:    int = 16
    num_special:  int = 3

    # --- Optimisation ---
    stage:                  int   = 3
    learning_rate:          float = 2e-5
    per_device_batch_size:  int   = 8
    gradient_accumulation:  int   = 8
    num_epochs:             int   = 1
    warmup_ratio:           float = 0.03
    max_grad_norm:          float = 1.0
    weight_decay:           float = 0.0
    lr_scheduler_type:      str   = "cosine"
    max_length:             int   = 2048
    dataloader_workers:     int   = 4
    save_steps:             int   = 500
    logging_steps:          int   = 10
    bf16:                   bool  = True
    fp16:                   bool  = False
    # "adamw_torch" (default) | "adamw_bnb_8bit" (bitsandbytes, needed for
    # Stage 2 with 3B+ models on a single 48 GB GPU to avoid OOM on fp32
    # Adam optimizer states).
    optim:                  str   = "adamw_torch"
    # Recompute activations during backward to trade compute for memory.
    # Required for Stage 2 with 3B+ models; optional elsewhere.
    gradient_checkpointing: bool  = False
    deepspeed:              Optional[str] = None
    balance_loss_lambda:    float = 0.05


# ---------------------------------------------------------------------------
# Custom Trainer subclass to handle multi-modal forward
# ---------------------------------------------------------------------------

class LeoMiniTrainer(Trainer):
    """
    Extends HuggingFace Trainer to:
    - call model.forward with correct keyword arguments
    - support multi-modal batch (pixel_values, pix2struct_inputs)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels            = inputs.pop("labels")
        pixel_values      = inputs.pop("pixel_values", None)
        pix2struct_inputs = inputs.pop("pix2struct_inputs", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pixel_values,
            pix2struct_inputs=pix2struct_inputs,
            labels=labels,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Override HF Trainer's save_model to handle tied weights.

        Most LLMs tie embed_tokens.weight == lm_head.weight (same storage).
        safetensors.save_file() raises RuntimeError on duplicate data_ptrs.
        We clone the duplicate tensor to break the tie before saving.

        The saved model.safetensors contains the full LeoMini state dict and
        is loadable by _load_from_checkpoint for resume via load_state_dict.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        state_dict = self.model.state_dict()

        # Detect and break any shared-memory ties (embed_tokens / lm_head, etc.)
        seen: dict[int, str] = {}
        for key in list(state_dict):
            ptr = state_dict[key].data_ptr()
            if ptr in seen:
                state_dict[key] = state_dict[key].clone()
            else:
                seen[ptr] = key

        import safetensors.torch as sf
        sf.save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: LeoMiniTrainingArgs) -> None:
    # --- Load model ---
    # Stage 2 needs Stage 1's trained projector; Stage 3 needs Stage 2's.
    # When --model_config is used, projector_path is auto-derived in __main__.
    # This warning fires only when running without --model_config and the user
    # forgot to set projector_path manually in the stage YAML config.
    if args.stage in (2, 3) and args.projector_path is None:
        print(
            f"WARNING: Stage {args.stage} training started without projector_path. "
            f"The projector will be randomly initialised, discarding Stage "
            f"{args.stage - 1} warmup. "
            f"Use --model_config (auto-derives the path) or set projector_path "
            f"explicitly in the stage YAML config."
        )
    # ------------------------------------------------------------------
    # Structured logger — one JSON file per run
    # ------------------------------------------------------------------
    run_id  = make_run_id(os.path.basename(args.llm_path), args.stage)
    log_dir = args.log_dir or "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.json")

    vi_logger = ViLeoMiniLogger(
        log_path=log_path,
        run_id=run_id,
        config={k: str(v) for k, v in vars(args).items()},
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = LeoMini.from_pretrained(
        llm_path=args.llm_path,
        projector_path=args.projector_path,
        enable_stage3_modules=(args.stage == 3),
        n_visual=args.n_visual,
        d_proj=args.d_proj_cotr,
        lora_rank=args.lora_rank,
        num_special=args.num_special,
        balance_loss_lambda=args.balance_loss_lambda,
        vision_experts=args.vision_experts if args.vision_experts else None,
        pix2struct_model_name=args.pix2struct_model_name,
    )
    model.set_stage(args.stage)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vi_logger.log_model_info({
        "total_params":     n_params,
        "trainable_params": n_trainable,
        "llm_path":         args.llm_path,
        "vision_experts":   args.vision_experts or "all",
        "n_visual":         args.n_visual,
    })

    # --- Dataset ---
    tokenizer        = model.tokenizer
    image_processor  = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    # Only load Pix2Struct processor if the model actually has a Pix2Struct expert.
    # This avoids loading the processor when the expert set is customised to
    # exclude it (e.g. a future Vicuna-7B variant without Pix2Struct).
    _has_pix2struct = any(isinstance(e, Pix2StructExpert) for e in model.vision.experts)
    pix2struct_processor = (
        AutoProcessor.from_pretrained(args.pix2struct_model_name)
        if _has_pix2struct else None
    )

    collator = LeoMiniCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_length=args.max_length,
    )

    # ------------------------------------------------------------------
    # Dataset selection: ViTextVQA path (vi_train_path set) or original
    # ------------------------------------------------------------------
    _use_vi = bool(args.vi_train_path)

    if _use_vi:
        from ..data.vitextvqa_dataset import KTVICDataset, OpenViVQADataset, ViTextVQADataset
        if args.stage == 1:
            DatasetClass = KTVICDataset
            hf_dataset_name = "ai-enthusiasm-community/KTVIC"
        elif args.stage == 2:
            DatasetClass = OpenViVQADataset
            hf_dataset_name = "uitnlp/OpenViVQA-dataset"
        else: # stage 3
            DatasetClass = ViTextVQADataset
            hf_dataset_name = "minhquan6203/ViTextVQA"

        train_split = args.vi_train_path  # e.g. "train"
        train_dataset = DatasetClass(
            split=train_split,
            tokenizer=tokenizer,
            image_processor=image_processor,
            pix2struct_processor=pix2struct_processor,
            max_length=args.max_length,
            hf_dataset_name=hf_dataset_name,
            cache_dir=args.vi_cache_dir or None,
        )
        vi_logger.log_custom(
            type="dataset_info",
            dataset=DatasetClass.__name__,
            split=train_split,
            n_samples=len(train_dataset),
        )
    else:
        if args.stage == 1:
            data_path = args.eagle_data_path
            DatasetClass = EAGLEDataset
        elif args.stage == 2:
            data_path = args.eagle_sft_path
            DatasetClass = EAGLEDataset
        else:
            data_path = args.llava_data_path
            DatasetClass = LLaVADataset

        train_dataset = DatasetClass(
            data_path=data_path,
            image_dir=args.image_dir,
            tokenizer=tokenizer,
            image_processor=image_processor,
            pix2struct_processor=pix2struct_processor,
            max_length=args.max_length,
        )
        vi_logger.log_custom(
            type="dataset_info",
            dataset=DatasetClass.__name__,
            data_path=data_path,
            n_samples=len(train_dataset),
        )

    # Enable gradient checkpointing on the LLM before wrapping with Trainer
    # so that HuggingFace's Trainer does not attempt to call the method itself
    # (which would fail for non-HuggingFace nn.Module wrappers like LeoMini).
    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()

    # --- HuggingFace TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"stage{args.stage}"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        # Pass False here — we call gradient_checkpointing_enable() above
        # directly on model.llm so HuggingFace does not try to call it on
        # the outer LeoMini nn.Module wrapper (which has no such method).
        gradient_checkpointing=False,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        deepspeed=args.deepspeed,
    )

    logging_cb = LeoMiniLoggingCallback(logger=vi_logger, stage=args.stage)

    trainer = LeoMiniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[logging_cb],
    )

    # --- Resume if checkpoint exists ---
    last_ckpt = None
    if args.resume_from:
        last_ckpt = args.resume_from
    elif os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    # Validate that the checkpoint directory actually contains loadable weights.
    # A directory can exist (from a previously crashed run where save failed)
    # without any model weights — HF Trainer would raise ValueError in that case.
    if last_ckpt is not None:
        _valid_weight_files = (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if not any(os.path.isfile(os.path.join(last_ckpt, f)) for f in _valid_weight_files):
            print(
                f"[Resume] Checkpoint at {last_ckpt} has no model weights "
                f"(likely from a crashed save). Starting from scratch."
            )
            last_ckpt = None

    trainer.train(resume_from_checkpoint=last_ckpt)

    # --- Save checkpoint ---
    if args.stage == 3:
        # Save projector + CoTR + MMoE-LLM adapters
        model.save_stage3_weights(
            os.path.join(training_args.output_dir, "stage3_adapter_weights.pt")
        )
    else:
        # Stages 1 and 2: save two artifacts needed by the next stage —
        #
        #  llm_checkpoint/  — LLM weights in HuggingFace format, loadable by
        #      AutoModelForCausalLM.from_pretrained().  trainer.save_model()
        #      saves the full LeoMini nn.Module state dict instead, which HF
        #      cannot load directly and would silently give the wrong weights.
        #
        #  projector_weights.pt — projector state dict.  Stage 2 warm-starts
        #      from Stage 1's trained projector; Stage 3 continues from Stage 2's.
        #      Without this, each stage re-initialises the projector randomly,
        #      discarding the previous stage's alignment training.
        llm_hf_dir = os.path.join(training_args.output_dir, "llm_checkpoint")
        print(f"[Stage {args.stage}] Saving HF-format LLM to {llm_hf_dir} ...")
        model.llm.save_pretrained(llm_hf_dir)
        model.tokenizer.save_pretrained(llm_hf_dir)

        model.save_projector_weights(
            os.path.join(training_args.output_dir, "projector_weights.pt")
        )

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to stage YAML config")
    parser.add_argument(
        "--model_config",
        default=None,
        help=(
            "Optional model-specific YAML override (e.g. configs/models/llama3_2_3b.yaml). "
            "Must contain 'base_model_path' (HuggingFace model ID for Stage 1). "
            "May also contain 'output_dir' (isolates checkpoints per model) and "
            "other per-model settings (batch size, optim, gradient_checkpointing). "
            "Do NOT set 'llm_path' in model configs for normal use — the trainer "
            "auto-derives it from output_dir/stage{n-1}/llm_checkpoint/. "
            "Set 'llm_path' only to skip stages (e.g. use a pre-trained EAGLE ckpt)."
        ),
    )
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    if cli.model_config is not None:
        with open(cli.model_config) as f:
            model_cfg = yaml.safe_load(f) or {}

        # ── Stage-aware llm_path resolution ──────────────────────────────────
        #
        # Model configs use 'base_model_path' (HuggingFace model ID) instead of
        # 'llm_path' so they cannot accidentally clobber the stage checkpoint paths.
        #
        # Resolution priority (first match wins):
        #
        #  (1) Explicit 'llm_path' in model_cfg
        #      → User is manually controlling the LLM path (skip-stage scenario,
        #        e.g. "llm_path: NVEagle/Eagle-X4-8B-Plus" to go Stage 1→3).
        #        Honour it without any modification.
        #
        #  (2) 'base_model_path' in model_cfg (normal case)
        #      Stage 1 → base_model_path (load from HuggingFace)
        #      Stage 2 → output_dir/stage1/llm_checkpoint/   ← saved by train()
        #      Stage 3 → output_dir/stage2/llm_checkpoint/   ← saved by train()
        #      If the expected llm_checkpoint/ directory is missing → sys.exit(1).
        #      No silent fallback: the user must either run the previous stage
        #      or explicitly set 'llm_path' in the model config to skip it.
        #
        #  (3) Neither key in model_cfg
        #      → Keep whatever llm_path the stage YAML already has.

        base_model_path   = model_cfg.pop("base_model_path", None)
        explicit_llm_path = model_cfg.get("llm_path")       # before cfg.update()

        # Apply remaining model-specific overrides (output_dir, batch, optim, …)
        cfg.update(model_cfg)

        stage   = cfg.get("stage", 1)
        out_dir = cfg.get("output_dir", "checkpoints")

        if explicit_llm_path:
            # Priority 1: explicit override in model config — trust the user
            cfg["llm_path"] = explicit_llm_path

        elif base_model_path:
            # Priority 2: normal multi-stage flow
            if stage == 1:
                cfg["llm_path"] = base_model_path
            else:
                # train() saves model.llm via save_pretrained() to this exact dir.
                # AutoModelForCausalLM.from_pretrained() can load it directly.
                prev_llm_dir = os.path.join(out_dir, f"stage{stage - 1}", "llm_checkpoint")
                if not os.path.isdir(prev_llm_dir):
                    print(
                        f"\nERROR: Stage {stage - 1} LLM checkpoint not found.\n"
                        f"  Expected: {prev_llm_dir}\n"
                        f"\n"
                        f"  Option A — run Stage {stage - 1} first:\n"
                        f"    bash scripts/run_stage{stage - 1}.sh {cli.model_config}\n"
                        f"\n"
                        f"  Option B — skip stages by adding to your model config:\n"
                        f"    llm_path: \"NVEagle/Eagle-X4-8B-Plus\"  # or any HF model\n",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                cfg["llm_path"] = prev_llm_dir

        # ── Projector handoff ─────────────────────────────────────────────────
        #
        # Stage 2 must warm-start from Stage 1's trained projector; without it
        # the visual projector is re-initialised randomly and Stage 1 warmup is lost.
        # Stage 3 continues fine-tuning Stage 2's projector.
        #
        # train() saves projector_weights.pt alongside llm_checkpoint/ so the path
        # is always output_dir/stage{n-1}/projector_weights.pt.
        #
        # We auto-set projector_path for both Stage 2 and Stage 3 when --model_config
        # is used (and the user hasn't set it manually in the stage YAML).
        if stage in (2, 3) and not cfg.get("projector_path"):
            prev_proj = os.path.join(out_dir, f"stage{stage - 1}", "projector_weights.pt")
            cfg["projector_path"] = prev_proj
            if not os.path.exists(prev_proj):
                print(
                    f"\nWARNING: projector_weights.pt not found at '{prev_proj}'.\n"
                    f"  Stage {stage} will start with a randomly initialized projector,\n"
                    f"  discarding Stage {stage - 1} warmup. Run Stage {stage - 1} first.\n",
                    file=sys.stderr,
                )

    args = LeoMiniTrainingArgs(**cfg)
    train(args)
