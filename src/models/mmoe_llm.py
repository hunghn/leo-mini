"""
MMoE-LLM — Section 3.3, Eq. 8–9.

Replaces the down_proj linear layer in every MLP block of the LLM with:
  y = f_ORI(x) + f_GEN(x) + Σ_{i∈E'} f_i^E(x) / k

where:
  f_ORI  — original frozen weight
  f_GEN  — general LoRA expert (always active, rank=16)
  f_i^E  — special LoRA expert i (top-k=1 selected by router)

Router: 2-layer MLP + GELU conditioned on (Ī, T, x)
  R = softmax(f_ROUTING(Ī_global, T_global, x)) ∈ R^E

Balanced loss (Eq. balance, λ=0.05):
  L_balance = λ · Σ_i (fraction_i − 1/E)²
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared context buffer — set once per forward, read by all MMoE layers
# ---------------------------------------------------------------------------

class ContextBuffer:
    """
    Holds projected visual tokens and text embeddings for the current batch.
    Populated in LeoMini.forward() before the LLM forward call.
    All MMoELinear layers hold a reference to the same ContextBuffer instance.
    """

    def __init__(self) -> None:
        self.visual: Optional[torch.Tensor] = None  # (B, N^V, d_LLM)
        self.text:   Optional[torch.Tensor] = None  # (B, N_T, d_LLM)

    def set(self, visual: torch.Tensor, text: torch.Tensor) -> None:
        self.visual = visual
        self.text   = text

    def clear(self) -> None:
        self.visual = None
        self.text   = None


# ---------------------------------------------------------------------------
# LoRA expert
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """
    Low-rank adaptation (LoRA) for a single linear layer.
    f_LoRA(x) = B @ (A @ x) * (alpha / rank)
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.rank   = rank
        self.scale  = alpha / rank
        self.lora_A = nn.Linear(d_in, rank,  bias=False)
        self.lora_B = nn.Linear(rank,  d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class MMoERouter(nn.Module):
    """
    2-layer MLP + GELU router.

    Input: per-token hidden state x concatenated with projected global visual
           and text context vectors.
    Output: routing probability R ∈ R^{E} (before top-k selection).

    Args:
        d_hidden:     dim of x  (= d_in of down_proj = intermediate_size, e.g. 14336)
        d_hidden_ctx: dim of visual/text context tensors (= d_LLM = hidden_size, e.g. 4096)
        d_context:    projected dim for visual/text global repr (default 128)
        num_experts:  number of special experts E (default 3)
    """

    def __init__(
        self,
        d_hidden:     int,
        d_hidden_ctx: int,
        d_context:    int = 128,
        num_experts:  int = 3,
    ) -> None:
        super().__init__()
        # Project visual/text (in d_LLM space) to small d_context vectors
        self.proj_visual = nn.Linear(d_hidden_ctx, d_context, bias=False)
        self.proj_text   = nn.Linear(d_hidden_ctx, d_context, bias=False)

        d_in = d_hidden + 2 * d_context
        # Hidden size = d_hidden // 4 (keeps router lightweight)
        d_mid = max(d_hidden // 4, num_experts * 16)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, num_experts),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,               # (B, seq_len, d_hidden)  — MLP input hidden state
        visual: torch.Tensor,           # (B, N^V, d_hidden)
        text: torch.Tensor,             # (B, N_T, d_hidden)
    ) -> torch.Tensor:
        # Global visual / text representations: (B, 1, d_context)
        v_ctx = self.proj_visual(visual.mean(dim=1, keepdim=True))  # (B, 1, d_context)
        t_ctx = self.proj_text(text.mean(dim=1, keepdim=True))      # (B, 1, d_context)

        # Expand global context to match sequence length
        seq_len = x.shape[1]
        v_ctx = v_ctx.expand(-1, seq_len, -1)   # (B, seq_len, d_context)
        t_ctx = t_ctx.expand(-1, seq_len, -1)   # (B, seq_len, d_context)

        # Concatenate per-token hidden state with global context
        router_in = torch.cat([x, v_ctx, t_ctx], dim=-1)  # (B, seq_len, d_hidden+2*d_context)
        return self.mlp(router_in)                          # (B, seq_len, num_experts)


# ---------------------------------------------------------------------------
# MMoELinear — drop-in replacement for nn.Linear in LLM MLP
# ---------------------------------------------------------------------------

class MMoELinear(nn.Module):
    """
    Wraps a frozen nn.Linear (the original down_proj) with:
      - 1 general LoRA expert (always active)
      - E special LoRA experts (top-1 selected per token)
      - A router conditioned on (Ī, T, x)

    Eq. 9: y = f_ORI(x) + f_GEN(x) + f_top1(x)

    balance_stats is accumulated during forward and reset each step.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        context_buffer:  ContextBuffer,
        lora_rank:       int = 16,
        lora_alpha:      float = 16.0,
        num_special:     int = 3,
        d_context:       int = 128,
    ) -> None:
        super().__init__()
        d_in  = original_linear.in_features   # intermediate_size (e.g. 14336)
        d_out = original_linear.out_features   # hidden_size = d_LLM (e.g. 4096)
        self._d_out = d_out

        # f_ORI: frozen original weight
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad_(False)

        # f_GEN: general LoRA (always active)
        self.lora_gen = LoRALayer(d_in, d_out, rank=lora_rank, alpha=lora_alpha)

        # f_i^E: E special LoRA experts
        self.num_special = num_special
        self.lora_experts = nn.ModuleList(
            [LoRALayer(d_in, d_out, rank=lora_rank, alpha=lora_alpha) for _ in range(num_special)]
        )

        # Router: x is in d_in (intermediate) space; visual/text are in d_out (d_LLM) space
        self.router = MMoERouter(
            d_hidden=d_in,
            d_hidden_ctx=d_out,
            d_context=d_context,
            num_experts=num_special,
        )

        # Shared context buffer (set externally before LLM forward)
        self.ctx = context_buffer

        # Accumulate routing fractions for balanced loss
        self.register_buffer("_expert_counts", torch.zeros(num_special))
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_in)  — intermediate MLP activations (input to down_proj)
        """
        # f_ORI (frozen)
        y = self.original(x)

        # f_GEN (always active)
        y = y + self.lora_gen(x)

        # Router — uses visual/text context when available
        if self.ctx.visual is not None and self.ctx.text is not None:
            logits = self.router(x, self.ctx.visual, self.ctx.text)  # (B, seq_len, E)
        else:
            # Text-only fallback: use x itself for both visual and text context
            logits = self.router(x, x, x)

        probs  = torch.softmax(logits, dim=-1)                       # (B, seq_len, E)
        chosen = probs.argmax(dim=-1)                                 # (B, seq_len)

        # Accumulate routing stats for balanced loss
        if self.training:
            with torch.no_grad():
                for e in range(self.num_special):
                    self._expert_counts[e] += (chosen == e).float().sum()
                self._total_tokens += chosen.numel()

        # Top-1 expert output: build a weighted sum (differentiable through probs)
        # For each expert e: probs[..., e] * lora_experts[e](x)
        # Hard top-1 with straight-through: use one-hot * probs for routing
        one_hot = F.one_hot(chosen, self.num_special).to(x.dtype)    # (B, seq_len, E)

        # Straight-through estimator: gradient flows via probs
        routing_weights = one_hot + (probs - probs.detach())          # (B, seq_len, E)

        expert_out = x.new_zeros(*x.shape[:-1], self._d_out)
        for e, expert in enumerate(self.lora_experts):
            w_e = routing_weights[..., e].unsqueeze(-1)               # (B, seq_len, 1)
            expert_out = expert_out + w_e * expert(x)

        return y + expert_out                                          # Eq. 9

    # ------------------------------------------------------------------
    def get_balance_loss(self, lambda_balance: float = 0.05) -> torch.Tensor:
        """
        L_balance = λ · Σ_i (fraction_i − 1/E)²
        Call after each forward pass, then reset stats.
        """
        if self._total_tokens == 0:
            return torch.tensor(0.0, device=self._expert_counts.device)

        fractions = self._expert_counts / self._total_tokens
        target    = 1.0 / self.num_special
        loss      = lambda_balance * ((fractions - target) ** 2).sum()
        return loss

    def reset_balance_stats(self) -> None:
        self._expert_counts.zero_()
        self._total_tokens = 0


# ---------------------------------------------------------------------------
# Utility: apply MMoE-LLM to every MLP block in the LLM
# ---------------------------------------------------------------------------

def apply_mmoe_to_model(
    model:          nn.Module,
    context_buffer: ContextBuffer,
    lora_rank:      int = 16,
    lora_alpha:     float = 16.0,
    num_special:    int = 3,
    d_context:      int = 128,
) -> None:
    """
    Replaces model.layers[i].mlp.down_proj (nn.Linear) with MMoELinear
    for every transformer block in the LLM.

    Works for LLaMA / Vicuna / Mistral family models where each block is
    accessible as model.layers or model.model.layers.
    """
    # Support both unwrapped (LlamaModel) and wrapped (LlamaForCausalLM)
    llm_layers = None
    for attr in ("layers", "model"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            layers = getattr(candidate, "layers", candidate)
            if hasattr(layers, "__len__") and hasattr(layers[0], "mlp"):
                llm_layers = layers
                break

    if llm_layers is None:
        raise ValueError(
            "Cannot locate transformer layers in the provided model. "
            "Expected model.layers or model.model.layers with .mlp.down_proj."
        )

    replaced = 0
    for layer in llm_layers:
        mlp = layer.mlp
        if not hasattr(mlp, "down_proj"):
            continue
        original_down = mlp.down_proj
        mlp.down_proj = MMoELinear(
            original_linear=original_down,
            context_buffer=context_buffer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            num_special=num_special,
            d_context=d_context,
        )
        replaced += 1

    print(f"[MMoE-LLM] Replaced down_proj in {replaced} MLP blocks.")


# ---------------------------------------------------------------------------
# Helper: collect balance loss across all MMoELinear layers
# ---------------------------------------------------------------------------

def collect_balance_loss(
    model: nn.Module,
    lambda_balance: float = 0.05,
    reset: bool = True,
) -> torch.Tensor:
    """Sums L_balance from every MMoELinear in the model."""
    total = None
    for module in model.modules():
        if isinstance(module, MMoELinear):
            l = module.get_balance_loss(lambda_balance)
            total = l if total is None else total + l
            if reset:
                module.reset_balance_stats()
    if total is None:
        return torch.tensor(0.0)
    return total
