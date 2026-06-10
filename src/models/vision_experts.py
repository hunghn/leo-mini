"""
MMoE-Vision — Section 3.3, "Effective visual comprehension".

Wraps 4 domain-specific vision experts following the EAGLE design (Shi et al., 2024):

  Expert         Model ID                                      Tokens  Dim
  ─────────────────────────────────────────────────────────────────────────
  CLIP           openai/clip-vit-large-patch14-336             576     1024
  EVA-02         EVA02-L-14-336 (open_clip, QuanSun/EVA-CLIP)  576     1024
  ConvNeXt       laion/CLIP-convnext_large_d_320.laion2B-…    576*    768
  Pix2Struct     google/pix2struct-large                       576*    2048

  * ConvNeXt and Pix2Struct spatial features are interpolated/truncated to
    576 tokens to match the ViT-based experts for Stages 1+2.
    In Stage 3 CoTR handles variable token counts natively (N_i may differ).

All experts are frozen during Stage 3 training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPVisionModel,
    Pix2StructVisionModel,
    CLIPImageProcessor,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Standard token count used by CLIP/EVA-02 at 336px (24×24 grid)
STANDARD_N_TOKENS = 576


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VisionExpert(nn.Module):
    """Common interface for all vision experts."""

    feature_dim: int          # d_i  — output feature dimension
    n_tokens:    int          # N_i  — number of output tokens at standard resolution

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return visual tokens of shape (B, N_i, d_i)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Expert 1 — CLIP ViT-L/14-336
# ---------------------------------------------------------------------------

class CLIPExpert(VisionExpert):
    """
    OpenAI CLIP ViT-Large-patch14-336.
    Returns the 576 patch tokens (CLS excluded).
    """

    feature_dim = 1024
    n_tokens    = STANDARD_N_TOKENS

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14-336") -> None:
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        """PIL images → pixel_values tensor."""
        return self.processor(images=images, return_tensors="pt").pixel_values

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        # last_hidden_state: (B, 1+576, 1024) — drop CLS at position 0
        return out.last_hidden_state[:, 1:, :]  # (B, 576, 1024)


# ---------------------------------------------------------------------------
# Expert 2 — EVA-02 CLIP-L-14-336
# ---------------------------------------------------------------------------

class EVA02Expert(VisionExpert):
    """
    EVA-02 CLIP-L-14-336 (Fang et al., 2024c) loaded via open_clip.
    Weights: QuanSun/EVA-CLIP (downloaded automatically on first use).
    """

    feature_dim = 1024
    n_tokens    = STANDARD_N_TOKENS

    def __init__(
        self,
        model_name: str = "EVA02-L-14-336",
        pretrained: str = "merged2b_s6b_b61k",
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("open_clip required: pip install open-clip-torch") from e

        clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = clip_model.visual

    @torch.no_grad()
    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self._preprocess(img) for img in images]
        return torch.stack(tensors)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # forward_features returns (B, 1+N, D) with CLS at index 0
        features = self.visual.forward_features(pixel_values)
        return features[:, 1:, :]  # (B, 576, 1024)


# ---------------------------------------------------------------------------
# Expert 3 — ConvNeXt-Large via open_clip
# ---------------------------------------------------------------------------

class ConvNeXtExpert(VisionExpert):
    """
    ConvNeXt-Large-D trained by LAION via open_clip.
    Extracts spatial feature maps from the last CNN stage and reshapes them
    to a sequence of patch tokens. Interpolated to STANDARD_N_TOKENS=576
    so the expert is compatible with Stages 1+2 without CoTR.
    """

    feature_dim = 768    # final feature dim of convnext_large_d
    n_tokens    = STANDARD_N_TOKENS

    def __init__(
        self,
        model_name: str = "convnext_large_d",
        pretrained: str = "laion2b_s26b_b102k_augreg",
        target_tokens: int = STANDARD_N_TOKENS,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("open_clip required: pip install open-clip-torch") from e

        self.target_h = int(target_tokens ** 0.5)  # 24 for 576 tokens
        self.target_w = self.target_h

        clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.trunk = clip_model.visual.trunk  # ConvNeXt backbone
        self._out_channels = self._get_out_channels()
        self.feature_dim = self._out_channels
        n_tokens = target_tokens
        ConvNeXtExpert.n_tokens = n_tokens

    def _get_out_channels(self) -> int:
        """Infer output channels by a dummy forward pass."""
        dummy = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            feat = self._extract_spatial(dummy)
        return feat.shape[-1]

    def _extract_spatial(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run trunk, returning last-stage spatial features (B, H*W, C)."""
        x = self.trunk.stem(pixel_values)
        for stage in self.trunk.stages:
            x = stage(x)               # (B, C, H, W) at last stage
        # Interpolate to target grid
        x = F.interpolate(
            x.float(),
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        ).to(x.dtype)
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 576, C)

    @torch.no_grad()
    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        tensors = [self._preprocess(img) for img in images]
        return torch.stack(tensors)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._extract_spatial(pixel_values)  # (B, 576, C)


# ---------------------------------------------------------------------------
# Expert 4 — Pix2Struct-Large (OCR/document expert)
# ---------------------------------------------------------------------------

class Pix2StructExpert(VisionExpert):
    """
    Pix2Struct-Large (Lee et al., 2023) — specialised for OCR and document
    understanding. Produces 2048-dim patch embeddings.

    The model tokenises images into variable-length patches; we take the first
    STANDARD_N_TOKENS patches (or pad with zeros) to keep consistent shape.
    """

    feature_dim = 2048
    n_tokens    = STANDARD_N_TOKENS

    def __init__(
        self,
        model_name: str = "google/pix2struct-large",
        target_tokens: int = STANDARD_N_TOKENS,
    ) -> None:
        super().__init__()
        self.target_tokens = target_tokens
        self.encoder   = Pix2StructVisionModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size  # 2048

    @torch.no_grad()
    def preprocess(
        self,
        images: List[Image.Image],
        texts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with flattened_patches and attention_mask.
        Pix2Struct requires these special inputs.
        """
        if texts is None:
            texts = [""] * len(images)
        return self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            max_patches=self.target_tokens,
        )

    def forward(
        self,
        flattened_patches: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        flattened_patches: (B, target_tokens, patch_dim)
        Returns (B, target_tokens, 2048) patch embeddings.
        """
        out = self.encoder(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
        )
        tokens = out.last_hidden_state  # (B, N, 2048)
        B, N, D = tokens.shape
        if N < self.target_tokens:
            pad = tokens.new_zeros(B, self.target_tokens - N, D)
            tokens = torch.cat([tokens, pad], dim=1)
        elif N > self.target_tokens:
            tokens = tokens[:, : self.target_tokens, :]
        return tokens  # (B, target_tokens, 2048)


# ---------------------------------------------------------------------------
# MMoE-Vision: combines all experts
# ---------------------------------------------------------------------------

class MMoEVision(nn.Module):
    """
    Wraps all vision experts and returns a list of per-expert token tensors.
    Each expert is independent and frozen in Stage 3.

    Expert ordering (Llama3-8B):  [CLIP, EVA-02, ConvNeXt, Pix2Struct]
    Expert ordering (Vicuna-7B):  [CLIP, EVA-02, ConvNeXt, Pix2Struct, SAM]

    Returns: List[Tensor]  where tensor i has shape (B, N_i, d_i)
    """

    def __init__(self, experts: List[VisionExpert]) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)

    @property
    def expert_dims(self) -> List[int]:
        return [e.feature_dim for e in self.experts]

    @property
    def expert_n_tokens(self) -> List[int]:
        return [e.n_tokens for e in self.experts]

    def freeze(self) -> None:
        """Freeze all expert weights (called in Stage 3)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        pixel_values: torch.Tensor,                    # (B, 3, H, W) — shared for ViT experts
        pix2struct_inputs: Optional[Dict] = None,      # special dict for Pix2Struct
    ) -> List[torch.Tensor]:
        """
        Returns list of (B, N_i, d_i) tensors, one per expert.
        """
        results: List[torch.Tensor] = []
        for expert in self.experts:
            if isinstance(expert, Pix2StructExpert):
                if pix2struct_inputs is not None:
                    tokens = expert(
                        flattened_patches=pix2struct_inputs["flattened_patches"],
                        attention_mask=pix2struct_inputs.get("attention_mask"),
                    )
                else:
                    # Fallback: create dummy patches if Pix2Struct inputs unavailable
                    B = pixel_values.shape[0]
                    tokens = pixel_values.new_zeros(B, expert.target_tokens, expert.feature_dim)
            else:
                tokens = expert(pixel_values)
            results.append(tokens)
        return results


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_mmoe_vision(
    use_clip:       bool = True,
    use_eva02:      bool = True,
    use_convnext:   bool = True,
    use_pix2struct: bool = True,
    use_sam:        bool = False,  # Vicuna-7B only
) -> MMoEVision:
    """Build and return MMoEVision with the requested experts."""
    experts: List[VisionExpert] = []

    if use_clip:
        print("Loading CLIP ViT-L/14-336 ...")
        experts.append(CLIPExpert())

    if use_eva02:
        print("Loading EVA-02 CLIP-L-14-336 ...")
        experts.append(EVA02Expert())

    if use_convnext:
        print("Loading ConvNeXt-Large-D via open_clip ...")
        experts.append(ConvNeXtExpert())

    if use_pix2struct:
        print("Loading Pix2Struct-Large ...")
        experts.append(Pix2StructExpert())

    if use_sam:
        raise NotImplementedError("SAM expert is used only in Vicuna-7B variant.")

    return MMoEVision(experts)
