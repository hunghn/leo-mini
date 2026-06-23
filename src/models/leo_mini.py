"""
LEO-MINI — Full pipeline (Section 3.1, Figure 2).

End-to-end flow:
  Image(s) + Text
      │
      ├─ MMoE-Vision ──► [I_1,...,I_m]          # per-expert visual tokens
      │
      ├─ CoTR (Stage 3) ──► Ī ∈ R^{B,N^V,d^V}  # consolidated (64 tokens)
      │   or direct concat (Stage 1+2)
      │
      ├─ VisualProjector ──► Ī ∈ R^{B,N^V,d_LLM}
      │
      └─ LLM (+ MMoE-LLM in Stage 3)
             ├─ embed_tokens(text_ids) → T ∈ R^{B,N_T,d_LLM}
             ├─ input = concat([Ī, T], dim=1)
             └─ autoregressive generation → Y

Training stages are controlled externally via set_stage().
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .vision_experts import MMoEVision, build_mmoe_vision
from .cotr import CoTR
from .projector import VisualProjector
from .mmoe_llm import ContextBuffer, apply_mmoe_to_model, collect_balance_loss


# Token id used as image placeholder in the text sequence
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200  # sentinel value in input_ids


class LeoMini(nn.Module):
    """
    LEO-MINI multimodal LLM.

    Args:
        llm:            pretrained LlamaForCausalLM (backbone)
        vision:         MMoEVision instance
        projector:      VisualProjector
        cotr:           CoTR instance (None in Stages 1+2)
        tokenizer:      corresponding tokenizer
        use_cotr:       whether CoTR is active (True in Stage 3)
        use_mmoe_llm:   whether MMoE-LLM is active (True in Stage 3)
        n_visual:       N^V output tokens from CoTR
    """

    def __init__(
        self,
        llm:                  AutoModelForCausalLM,
        vision:               MMoEVision,
        projector:            VisualProjector,
        tokenizer:            AutoTokenizer,
        cotr:                 Optional[CoTR] = None,
        use_mmoe_llm:         bool  = False,
        n_visual:             int   = 64,
        lora_rank:            int   = 16,
        num_special:          int   = 3,
        balance_loss_lambda:  float = 0.05,
    ) -> None:
        super().__init__()
        self.llm                 = llm
        self.vision              = vision
        self.projector           = projector
        self.tokenizer           = tokenizer
        self.cotr                = cotr
        self.n_visual            = n_visual
        self.balance_loss_lambda = balance_loss_lambda

        # Context buffer shared by all MMoELinear layers
        self.ctx_buffer = ContextBuffer()

        if use_mmoe_llm:
            apply_mmoe_to_model(
                self.llm,
                context_buffer=self.ctx_buffer,
                lora_rank=lora_rank,
                num_special=num_special,
            )

        # Register image token
        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            llm.resize_token_embeddings(len(tokenizer))

    # ------------------------------------------------------------------
    # Stage configuration
    # ------------------------------------------------------------------

    def set_stage(self, stage: int) -> None:
        """
        Configure trainable parameters per training stage.

          Stage 1: projector only
          Stage 2: all modules
          Stage 3: projector + cotr + mmoe_llm (down_proj LoRAs + routers)
        """
        self.stage = stage
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad_(False)

        if stage == 1:
            for p in self.projector.parameters():
                p.requires_grad_(True)

        elif stage == 2:
            for p in self.parameters():
                p.requires_grad_(True)

        elif stage == 3:
            # Visual Projector remains trainable in Stage 3 (paper Table 6)
            for p in self.projector.parameters():
                p.requires_grad_(True)
            # CoTR: randomly initialised and fine-tuned (paper Appendix A.2)
            if self.cotr is not None:
                for p in self.cotr.parameters():
                    p.requires_grad_(True)
            # MMoE-LLM: only LoRA adapters + router (original down_proj stays frozen)
            from .mmoe_llm import MMoELinear
            for m in self.llm.modules():
                if isinstance(m, MMoELinear):
                    for p in m.lora_gen.parameters():
                        p.requires_grad_(True)
                    for expert in m.lora_experts:
                        for p in expert.parameters():
                            p.requires_grad_(True)
                    for p in m.router.parameters():
                        p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Stage {stage}] Trainable parameters: {n_trainable:,}")

    # ------------------------------------------------------------------
    # Visual token extraction
    # ------------------------------------------------------------------

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
        pix2struct_inputs: Optional[Dict] = None,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the full visual encoding pipeline:
          pixel_values → MMoE-Vision → [CoTR] → Projector
          Returns projected visual tokens (B, N^V, d_LLM).
        """
        # MMoE-Vision: list of (B, N_i, d_i) per expert
        visual_list = self.vision(pixel_values, pix2struct_inputs)

        if self.cotr is not None and text_tokens is not None:
            # Stage 3: CoTR consolidation → (B, N^V, Σd_i)
            visual_concat = self.cotr(visual_list, text_tokens)
        else:
            # Stage 1+2: channel-wise concat across experts that share N tokens
            # All ViT experts produce 576 tokens; concatenate features
            visual_concat = torch.cat(visual_list, dim=-1)  # (B, 576, Σd_i)

        # Project to LLM dimension
        return self.projector(visual_concat)  # (B, N^V or 576, d_LLM)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:         torch.Tensor,                # (B, L) — includes IMAGE_TOKEN_INDEX
        attention_mask:    Optional[torch.Tensor] = None,
        pixel_values:      Optional[torch.Tensor] = None,
        pix2struct_inputs: Optional[Dict] = None,
        labels:            Optional[torch.Tensor] = None,
        return_loss:       bool = True,
    ) -> CausalLMOutputWithPast:
        """
        Multi-modal forward pass.
        Visual tokens replace IMAGE_TOKEN_INDEX positions in the input sequence.
        """
        # 1. Text embeddings
        embed = self.llm.get_input_embeddings()
        # Replace image tokens with zero placeholders for now
        safe_ids = input_ids.clone()
        safe_ids[safe_ids == IMAGE_TOKEN_INDEX] = 0
        text_embeds = embed(safe_ids)                    # (B, L, d_LLM)

        if pixel_values is not None:
            # 2. Encode images
            # For CoTR text context we use current text embeddings
            text_for_cotr = text_embeds if self.cotr is not None else None
            vis_tokens = self._encode_images(
                pixel_values, pix2struct_inputs, text_for_cotr
            )  # (B, N^V, d_LLM)

            # 3. Set router context (before LLM forward)
            # text context = the portion of the sequence that is text
            text_global = text_embeds  # (B, L, d_LLM) — router uses mean internally
            self.ctx_buffer.set(vis_tokens, text_global)

            # 4. Replace IMAGE_TOKEN_INDEX positions with visual tokens
            inputs_embeds = self._merge_visual_text(
                text_embeds, vis_tokens, input_ids
            )  # (B, L', d_LLM)
        else:
            inputs_embeds = text_embeds
            self.ctx_buffer.clear()

        # 5. Build attention mask and labels for merged sequence
        if pixel_values is not None:
            n_vis = vis_tokens.shape[1]
            if attention_mask is not None:
                attention_mask = self._merge_attention_mask(
                    attention_mask, n_vis, input_ids
                )
            if labels is not None:
                labels = self._merge_labels(labels, n_vis, input_ids)

        # 6. LLM forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # 7. Add balanced loss in Stage 3
        if self.training and any(
            m.__class__.__name__ == "MMoELinear" for m in self.llm.modules()
        ):
            balance_loss = collect_balance_loss(self.llm, lambda_balance=self.balance_loss_lambda)
            if outputs.loss is not None:
                outputs = outputs.__class__(
                    loss=outputs.loss + balance_loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        self.ctx_buffer.clear()
        return outputs

    # ------------------------------------------------------------------
    # Helper: merge visual tokens into text embedding sequence
    # ------------------------------------------------------------------

    def _merge_visual_text(
        self,
        text_embeds: torch.Tensor,   # (B, L, d_LLM)
        vis_tokens:  torch.Tensor,   # (B, N^V, d_LLM)
        input_ids:   torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """
        For each sample in the batch: replace the IMAGE_TOKEN_INDEX slice
        in the text embedding with visual tokens.
        Returns merged (B, L - 1 + N^V, d_LLM).
        """
        B, L, D = text_embeds.shape
        N = vis_tokens.shape[1]
        merged = []

        for b in range(B):
            img_pos = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged.append(text_embeds[b])
                continue

            pos = img_pos[0].item()  # position of the single <image> token
            before = text_embeds[b, :pos, :]             # (pos, D)
            after  = text_embeds[b, pos + 1:, :]         # (L-pos-1, D)
            seq = torch.cat([before, vis_tokens[b], after], dim=0)  # (pos+N+L-pos-1, D)
            merged.append(seq)

        # Pad to same length if sequences differ (shouldn't happen with proper batching)
        max_len = max(s.shape[0] for s in merged)
        padded = torch.zeros(B, max_len, D, device=text_embeds.device, dtype=text_embeds.dtype)
        for b, seq in enumerate(merged):
            padded[b, :seq.shape[0]] = seq
        return padded

    def _merge_labels(
        self,
        labels:    torch.Tensor,   # (B, L)
        n_visual:  int,
        input_ids: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """
        Extend labels to match the merged sequence length (B, L - 1 + N^V).
        Visual token positions receive IGNORE_INDEX (-100) so they are not
        included in the cross-entropy loss.
        """
        IGNORE = -100
        B, L = labels.shape
        new_labels_list = []
        for b in range(B):
            img_pos = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                new_labels_list.append(labels[b])
                continue
            pos = img_pos[0].item()
            before  = labels[b, :pos]
            vis_lbl = labels.new_full((n_visual,), IGNORE)
            after   = labels[b, pos + 1:]
            new_labels_list.append(torch.cat([before, vis_lbl, after]))

        max_len = max(lbl.shape[0] for lbl in new_labels_list)
        out = labels.new_full((B, max_len), IGNORE)
        for b, lbl in enumerate(new_labels_list):
            out[b, :lbl.shape[0]] = lbl
        return out

    def _merge_attention_mask(
        self,
        attention_mask: torch.Tensor,  # (B, L)
        n_visual: int,
        input_ids: torch.Tensor,        # (B, L)
    ) -> torch.Tensor:
        """Expand attention_mask to account for inserted visual tokens."""
        B, L = attention_mask.shape
        new_mask_list = []
        for b in range(B):
            img_pos = (input_ids[b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                new_mask_list.append(attention_mask[b])
                continue
            pos = img_pos[0].item()
            before = attention_mask[b, :pos]
            vis_mask = attention_mask.new_ones(n_visual)
            after = attention_mask[b, pos + 1:]
            new_mask_list.append(torch.cat([before, vis_mask, after]))

        max_len = max(m.shape[0] for m in new_mask_list)
        out = attention_mask.new_zeros(B, max_len)
        for b, m in enumerate(new_mask_list):
            out[b, :m.shape[0]] = m
        return out

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids:         torch.Tensor,
        pixel_values:      Optional[torch.Tensor] = None,
        pix2struct_inputs: Optional[Dict] = None,
        attention_mask:    Optional[torch.Tensor] = None,
        max_new_tokens:    int = 512,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation.  Returns token ids (B, new_len).
        """
        # Build inputs_embeds with visual tokens merged in
        embed = self.llm.get_input_embeddings()
        safe_ids = input_ids.clone()
        safe_ids[safe_ids == IMAGE_TOKEN_INDEX] = 0
        text_embeds = embed(safe_ids)

        if pixel_values is not None:
            text_for_cotr = text_embeds if self.cotr is not None else None
            vis_tokens = self._encode_images(pixel_values, pix2struct_inputs, text_for_cotr)
            self.ctx_buffer.set(vis_tokens, text_embeds)
            inputs_embeds = self._merge_visual_text(text_embeds, vis_tokens, input_ids)
            if attention_mask is not None:
                attention_mask = self._merge_attention_mask(
                    attention_mask, vis_tokens.shape[1], input_ids
                )
        else:
            inputs_embeds = text_embeds

        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        self.ctx_buffer.clear()
        return out

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        llm_path:              str,
        tokenizer_path:        Optional[str]       = None,
        projector_path:        Optional[str]       = None,
        cotr_path:             Optional[str]       = None,
        stage3_weights:        Optional[str]       = None,
        enable_stage3_modules: bool                = False,
        n_visual:              int                 = 64,
        d_proj:                int                 = 256,
        lora_rank:             int                 = 16,
        num_special:           int                 = 3,
        balance_loss_lambda:   float               = 0.05,
        load_in_4bit:          bool                = False,
        vision_experts:        Optional[List[str]] = None,
        pix2struct_model_name: str                 = "google/pix2struct-large",
        **kwargs,
    ) -> "LeoMini":
        """
        Build LeoMini from a pretrained LLM checkpoint (EAGLE / LLaVA-1.5 style).

        projector_path: path to projector_weights.pt saved after Stage 1 or 2.
            Must be provided for Stage 3 training so that the trained projector is
            not silently replaced by random initialisation.
        enable_stage3_modules: set True for Stage 3 training from scratch to
            randomly initialise CoTR and apply MMoE-LLM.
        stage3_weights: path to the adapter checkpoint saved by save_stage3_weights();
            implicitly enables Stage 3 modules and loads projector + CoTR + adapters.
        """
        from transformers import BitsAndBytesConfig

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"Loading LLM from {llm_path} ...")
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            **kwargs,
        )

        tok_path = tokenizer_path or llm_path
        tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        # Ensure pad token exists — required for batched training.
        # Phi uses <|endoftext|> as eos; Llama uses <|eot_id|>.
        # Qwen2.5 uses <|im_end|> as eos (id=151645).
        # All lack a dedicated pad token, so we reuse eos.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Vision experts — accepts an optional list of expert names so that
        # variants (e.g. Vi-LEO-MINI with ["clip","pix2struct"]) skip loading
        # unnecessary models and reduce VRAM usage.
        vision = build_mmoe_vision(
            experts=vision_experts,
            pix2struct_model_name=pix2struct_model_name,
        )

        # Projector: infer d_visual from expert dims
        d_visual = sum(vision.expert_dims)
        d_llm    = llm.config.hidden_size
        projector = VisualProjector(d_visual, d_llm)

        # Restore projector weights from a prior training stage if provided.
        # Without this, Stage 3 would start from a randomly-initialised projector,
        # discarding all progress from Stages 1 and 2.
        if projector_path is not None:
            proj_ckpt = torch.load(projector_path, map_location="cpu")
            projector.load_state_dict(proj_ckpt, strict=True)
            print(f"Loaded projector weights from {projector_path}")

        # CoTR + MMoE-LLM: create when explicitly requested (Stage 3 training from
        # scratch) OR when loading from a saved checkpoint.
        need_stage3 = enable_stage3_modules or (stage3_weights is not None) or (cotr_path is not None)
        cotr = None
        use_mmoe_llm = False
        if need_stage3:
            cotr = CoTR(
                expert_dims=vision.expert_dims,
                d_text=d_llm,
                d_proj=d_proj,
                n_visual=n_visual,
            )
            use_mmoe_llm = True

        model = cls(
            llm=llm,
            vision=vision,
            projector=projector,
            tokenizer=tokenizer,
            cotr=cotr,
            use_mmoe_llm=use_mmoe_llm,
            n_visual=n_visual,
            lora_rank=lora_rank,
            num_special=num_special,
            balance_loss_lambda=balance_loss_lambda,
        )

        # Load stage-3 adapter weights if provided
        if stage3_weights is not None:
            ckpt = torch.load(stage3_weights, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            # Log by module group to surface real mismatches
            missing_groups   = sorted({k.split(".")[0] for k in missing})
            unexpected_groups = sorted({k.split(".")[0] for k in unexpected})
            print(
                f"Loaded stage3 weights — "
                f"missing keys: {len(missing)} (groups: {missing_groups}), "
                f"unexpected keys: {len(unexpected)} (groups: {unexpected_groups})"
            )

        # Load standalone CoTR weights if provided separately
        if cotr_path is not None and model.cotr is not None:
            cotr_ckpt = torch.load(cotr_path, map_location="cpu")
            model.cotr.load_state_dict(cotr_ckpt, strict=True)
            print(f"Loaded CoTR weights from {cotr_path}")

        return model

    def save_stage3_weights(self, save_path: str) -> None:
        """
        Save Stage-3 trainable weights: projector + CoTR + MMoE-LLM adapters.

        Projector is included because it is trainable in Stage 3 (paper Table 6).
        Frozen original down_proj weights are excluded — they live in the LLM
        checkpoint and are restored by from_pretrained at load time.
        """
        from .mmoe_llm import MMoELinear
        state = {}
        # Projector (trainable in Stage 3, paper Table 6)
        for k, v in self.projector.state_dict().items():
            state[f"projector.{k}"] = v
        # CoTR
        if self.cotr is not None:
            for k, v in self.cotr.state_dict().items():
                state[f"cotr.{k}"] = v
        # MMoE-LLM adapters — skip frozen original down_proj weights
        for name, m in self.llm.named_modules():
            if isinstance(m, MMoELinear):
                for k, v in m.state_dict().items():
                    if not k.startswith("original"):
                        state[f"llm.{name}.{k}"] = v
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(state, save_path)
        print(f"Saved Stage-3 weights to {save_path}")

    def save_projector_weights(self, save_path: str) -> None:
        """
        Save projector state dict for use in subsequent stages.

        Call after Stage 1 or 2 so that Stage 3 can initialise the projector
        from the trained weights via from_pretrained(projector_path=...).
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(self.projector.state_dict(), save_path)
        print(f"Saved projector weights to {save_path}")
