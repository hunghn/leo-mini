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
    AutoTokenizer,
    CLIPImageProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from ..data.dataset import EAGLEDataset, LLaVADataset
from ..data.collator import LeoMiniCollator
from ..models.leo_mini import LeoMini


# ---------------------------------------------------------------------------
# Training argument dataclass
# ---------------------------------------------------------------------------

@dataclass
class LeoMiniTrainingArgs:
    # --- Paths ---
    llm_path:          str = "meta-llama/Meta-Llama-3-8B-Instruct"
    eagle_data_path:   str = ""   # alignment JSON  (Stage 1)
    eagle_sft_path:    str = ""   # full SFT JSON   (Stage 2)
    llava_data_path:   str = ""   # LLaVA-v1.5 JSON (Stage 3)
    image_dir:         str = ""   # root for image files
    output_dir:        str = "checkpoints"
    resume_from:       Optional[str] = None   # checkpoint to resume from

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
    deepspeed:              Optional[str] = "configs/deepspeed_zero2.json"
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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: LeoMiniTrainingArgs) -> None:
    # --- Load model ---
    use_stage3 = args.stage == 3
    model = LeoMini.from_pretrained(
        llm_path=args.llm_path,
        cotr_path=None,
        stage3_weights=None,
        n_visual=args.n_visual,
        d_proj=args.d_proj_cotr,
        lora_rank=args.lora_rank,
        num_special=args.num_special,
    )
    model.set_stage(args.stage)

    # --- Dataset ---
    tokenizer       = model.tokenizer
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    collator        = LeoMiniCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_length=args.max_length,
    )

    if args.stage == 1:
        data_path = args.eagle_data_path
        DatasetClass = EAGLEDataset
    elif args.stage == 2:
        data_path = args.eagle_sft_path
        DatasetClass = EAGLEDataset
    else:  # Stage 3
        data_path = args.llava_data_path
        DatasetClass = LLaVADataset

    train_dataset = DatasetClass(
        data_path=data_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )

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
        fp16=not args.bf16,
        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        deepspeed=args.deepspeed,
    )

    trainer = LeoMiniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # --- Resume if checkpoint exists ---
    last_ckpt = None
    if args.resume_from:
        last_ckpt = args.resume_from
    elif os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=last_ckpt)

    # --- Save Stage 3 (only CoTR + MMoE-LLM adapters) ---
    if args.stage == 3:
        model.save_stage3_weights(
            os.path.join(training_args.output_dir, "stage3_adapter_weights.pt")
        )
    else:
        trainer.save_model()

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to stage YAML config")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    args = LeoMiniTrainingArgs(**cfg)
    train(args)
