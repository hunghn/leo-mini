"""
Conditional Token Reduction (CoTR) — Section 3.2, Eq. 2–7.

For each vision expert i, CoTR aggregates N_i visual tokens into N^V=64 consolidated
tokens by attending jointly over:
  - a learnable query               (Eq. 2  — query-visual attention)
  - self-similarity within expert   (Eq. 3  — self-attention)
  - cross-expert global context     (Eq. 4  — cross-expert attention)
  - text instruction context        (Eq. 5  — text-visual attention)

Final output Ī is the channel-wise concatenation of per-expert consolidated tokens.
"""

import math
from typing import List

import torch
import torch.nn as nn


class CoTR(nn.Module):
    """
    Conditional Token Reduction.

    Args:
        expert_dims:  list of visual feature dims [d_1, ..., d_m] per expert
        d_text:       text token feature dim (typically d_LLM, e.g. 4096)
        d_proj:       common projection dim for computing attention scores
        n_visual:     desired number of output tokens N^V (default 64)
    """

    def __init__(
        self,
        expert_dims: List[int],
        d_text: int,
        d_proj: int = 256,
        n_visual: int = 64,
    ) -> None:
        super().__init__()
        self.expert_dims = expert_dims
        self.m = len(expert_dims)
        self.d_proj = d_proj
        self.n_visual = n_visual

        # Learnable query Q_i ∈ R^{N^V × d_i} for each expert.
        # Each query learns which visual tokens are most informative for expert i.
        self.queries = nn.ParameterList(
            [nn.Parameter(torch.empty(n_visual, d_i)) for d_i in expert_dims]
        )

        # Per-expert projections into common d_proj space
        self.proj_Q = nn.ModuleList(
            [nn.Linear(d_i, d_proj, bias=False) for d_i in expert_dims]
        )
        self.proj_I = nn.ModuleList(
            [nn.Linear(d_i, d_proj, bias=False) for d_i in expert_dims]
        )

        # Single text projection shared across experts
        self.proj_T = nn.Linear(d_text, d_proj, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for q in self.queries:
            nn.init.trunc_normal_(q, std=0.02)
        for m in [*self.proj_Q, *self.proj_I, self.proj_T]:
            nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        visual_tokens: List[torch.Tensor],  # I_i: (B, N_i, d_i)
        text_tokens: torch.Tensor,           # T:   (B, N_T, d_text)
    ) -> torch.Tensor:
        """
        Returns Ī = concat([Ī_1,...,Ī_m], dim=-1) ∈ R^{B, N^V, Σd_i}.

        The output token count is always self.n_visual = 64; the feature
        dimension is the sum of all expert feature dims (original space).
        """
        B = text_tokens.shape[0]

        # --- Project all visual tokens and text to common d_proj space ---
        # I_bars[i]: (B, N_i, d_proj)
        I_bars: List[torch.Tensor] = [
            self.proj_I[i](visual_tokens[i]) for i in range(self.m)
        ]
        # T_hat: (B, N_T, d_proj)
        T_hat = self.proj_T(text_tokens)

        # Global text representation: (B, 1, d_proj)
        # Corresponds to "1 · T̂" in Eq. 5 — mean over text token positions
        T_global = T_hat.mean(dim=1, keepdim=True)

        consolidated: List[torch.Tensor] = []

        for i in range(self.m):
            I_i = visual_tokens[i]   # (B, N_i, d_i) — original, used in Eq. 7
            I_bar_i = I_bars[i]      # (B, N_i, d_proj) — projected
            N_i = I_i.shape[1]

            # --- Projected query: (N^V, d_proj) → (B, N^V, d_proj) ---
            Q_bar_i = self.proj_Q[i](self.queries[i])             # (N^V, d_proj)
            Q_bar_i = Q_bar_i.unsqueeze(0).expand(B, -1, -1)     # (B, N^V, d_proj)

            # Eq. 2 — query-visual attention
            # s_QUERY = Q̄_i · Ī_i^T ∈ R^{B, N^V, N_i}
            s_query = torch.bmm(Q_bar_i, I_bar_i.transpose(1, 2))

            # Eq. 3 — self-attention: **1** · Ī_i · Ī_i^T ∈ R^{B, 1, N_i}
            # "1" acts as a summation vector over the token axis, giving the
            # global representation of expert i that attends to each token.
            I_global = I_bar_i.mean(dim=1, keepdim=True)          # (B, 1, d_proj)
            s_self   = torch.bmm(I_global, I_bar_i.transpose(1, 2))  # (B, 1, N_i)

            # Eq. 4 — cross-expert attention: Σ_{j≠i} mean(Ī_j) · Ī_i^T ∈ R^{B, 1, N_i}
            # Aggregates global representations from all other experts.
            s_cross = I_bar_i.new_zeros(B, 1, N_i)
            for j in range(self.m):
                if j != i:
                    I_j_global = I_bars[j].mean(dim=1, keepdim=True)  # (B, 1, d_proj)
                    s_cross = s_cross + torch.bmm(
                        I_j_global, I_bar_i.transpose(1, 2)
                    )

            # Eq. 5 — text-visual attention: **1** · T̂ · Ī_i^T ∈ R^{B, 1, N_i}
            s_text = torch.bmm(T_global, I_bar_i.transpose(1, 2))    # (B, 1, N_i)

            # Eq. 6 — aggregate and normalise (scale by √d_proj, the dot-product space dim)
            # Broadcast: s_query (B, N^V, N_i) + others (B, 1, N_i) → (B, N^V, N_i)
            scale    = math.sqrt(self.d_proj)
            attn_raw = (s_query + s_self + s_cross + s_text) / scale
            alpha_i  = torch.softmax(attn_raw, dim=-1)               # (B, N^V, N_i)

            # Eq. 7 — weighted aggregation in ORIGINAL expert feature space
            # Ī_i = α_i · I_i ∈ R^{B, N^V, d_i}
            I_out = torch.bmm(alpha_i, I_i)                          # (B, N^V, d_i)
            consolidated.append(I_out)

        # Concat channel-wise: Ī ∈ R^{B, N^V, Σd_i}
        return torch.cat(consolidated, dim=-1)

    # ------------------------------------------------------------------
    @property
    def d_out(self) -> int:
        """Total output feature dimension after channel-wise concat."""
        return sum(self.expert_dims)
