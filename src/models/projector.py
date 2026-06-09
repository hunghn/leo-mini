"""
Visual Projector — Section 3.1.

Maps consolidated visual tokens Ī ∈ R^{B, N^V, d^V} to LLM input space
R^{B, N^V, d_LLM} using a 2-layer MLP with GELU activation.

Architecture (from paper):
  Linear(d^V → d_hidden) → GELU → Linear(d_hidden → d_LLM)

The projector is:
  • Trained in Stage 1 (only trainable component)
  • Kept trainable in Stage 2
  • Frozen in Stage 3 (loaded from EAGLE / Stage 2 checkpoint)
"""

from typing import Optional

import torch
import torch.nn as nn


class VisualProjector(nn.Module):
    """
    2-layer MLP projector.

    Architecture per PLAN.md Section 2.3:
      Linear(d^V → d_hidden) → GELU → Linear(d_hidden → d_LLM)

    Args:
        d_visual:  input feature dim d^V = Σ expert_dims  (e.g. 4864)
        d_llm:     LLM hidden dim d_LLM                  (e.g. 4096)
        d_hidden:  inner hidden dim (default = 4 * d_llm, per PLAN.md)
    """

    def __init__(
        self,
        d_visual: int,
        d_llm:    int,
        d_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_hidden is None:
            d_hidden = 4 * d_llm

        self.proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_llm),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        visual_tokens: (B, N^V, d^V)
        Returns:       (B, N^V, d_LLM)
        """
        return self.proj(visual_tokens)

    def freeze(self) -> None:
        """Freeze all projector weights (Stage 3)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)


