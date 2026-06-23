"""
Chart generation for Vi-LEO-MINI training and evaluation history.

Reads the structured JSON log produced by ViLeoMiniLogger and exports:
  training_loss_stageN.png     — loss curve per stage
  balance_loss_stage3.png      — balance loss vs CE loss (stage 3)
  eval_metrics_history.png     — ANLS / EM / F1 over training steps
  learning_rate_schedule.png   — LR schedule per stage
  anls_distribution.png        — histogram of per-sample ANLS scores
  expert_routing_stage3.png    — placeholder (needs routing stats hook)

Standalone usage:
    python -m src.utils.visualizer --log logs/run.json --output results/plots/

Programmatic:
    from src.utils.visualizer import plot_training_history
    plot_training_history("logs/run.json", "results/plots/")
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..eval.vi_evaluator import EvalResult


# ---------------------------------------------------------------------------
# Internal: lazy matplotlib import so non-GUI servers don't need a display
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_log(log_path: str) -> Dict[str, Any]:
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)


def _events(log: Dict, event_type: str) -> List[Dict]:
    return [e for e in log.get("events", []) if e.get("type") == event_type]


def _save(plt, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved {path}")


# ---------------------------------------------------------------------------
# Plot: training loss per stage
# ---------------------------------------------------------------------------

def plot_training_loss(log: Dict, output_dir: str) -> None:
    plt = _plt()
    steps_by_stage: Dict[int, List] = {}
    losses_by_stage: Dict[int, List] = {}

    for e in _events(log, "train_step"):
        s = e.get("stage", 0)
        steps_by_stage.setdefault(s, []).append(e["global_step"])
        losses_by_stage.setdefault(s, []).append(e["loss"])

    for stage in sorted(steps_by_stage):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps_by_stage[stage], losses_by_stage[stage],
                linewidth=0.8, color="steelblue", label=f"Stage {stage} loss")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — Stage {stage}")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(plt, os.path.join(output_dir, "plots", f"training_loss_stage{stage}.png"))


# ---------------------------------------------------------------------------
# Plot: balance loss vs CE loss (Stage 3)
# ---------------------------------------------------------------------------

def plot_balance_loss(log: Dict, output_dir: str) -> None:
    plt = _plt()
    stage3_events = [e for e in _events(log, "train_step") if e.get("stage") == 3]
    if not stage3_events:
        return

    steps  = [e["global_step"]                       for e in stage3_events]
    losses = [e["loss"]                               for e in stage3_events]
    bals   = [e.get("balance_loss", None)             for e in stage3_events]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, losses, color="steelblue", linewidth=0.8, label="Total loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Loss", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    if any(b is not None for b in bals):
        ax2 = ax1.twinx()
        bal_vals = [b if b is not None else float("nan") for b in bals]
        ax2.plot(steps, bal_vals, color="tomato", linewidth=0.8,
                 linestyle="--", label="Balance loss")
        ax2.set_ylabel("Balance Loss", color="tomato")
        ax2.tick_params(axis="y", labelcolor="tomato")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend()

    ax1.set_title("Stage 3 — Total Loss vs Balance Loss")
    ax1.grid(alpha=0.3)
    _save(plt, os.path.join(output_dir, "plots", "balance_loss_stage3.png"))


# ---------------------------------------------------------------------------
# Plot: eval metrics history (ANLS / EM / F1)
# ---------------------------------------------------------------------------

def plot_eval_history(log: Dict, output_dir: str) -> None:
    plt = _plt()
    eval_events = _events(log, "eval")
    if not eval_events:
        return

    steps = [e["global_step"] for e in eval_events]
    anls  = [e["anls"] * 100  for e in eval_events]
    em    = [e["em"]   * 100  for e in eval_events]
    f1    = [e["f1"]   * 100  for e in eval_events]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, anls, marker="o", color="steelblue",  label="ANLS (%)")
    ax.plot(steps, em,   marker="s", color="tomato",     label="EM (%)")
    ax.plot(steps, f1,   marker="^", color="seagreen",   label="F1 (%)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Score (%)")
    ax.set_title("Evaluation Metrics History — ViTextVQA")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(plt, os.path.join(output_dir, "plots", "eval_metrics_history.png"))


# ---------------------------------------------------------------------------
# Plot: learning rate schedule
# ---------------------------------------------------------------------------

def plot_lr_schedule(log: Dict, output_dir: str) -> None:
    plt = _plt()
    steps_by_stage: Dict[int, List] = {}
    lrs_by_stage:   Dict[int, List] = {}

    for e in _events(log, "train_step"):
        if "learning_rate" not in e:
            continue
        s = e.get("stage", 0)
        steps_by_stage.setdefault(s, []).append(e["global_step"])
        lrs_by_stage.setdefault(s, []).append(e["learning_rate"])

    if not steps_by_stage:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue", "tomato", "seagreen"]
    for i, stage in enumerate(sorted(steps_by_stage)):
        ax.plot(steps_by_stage[stage], lrs_by_stage[stage],
                color=colors[i % len(colors)], linewidth=0.8, label=f"Stage {stage}")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(plt, os.path.join(output_dir, "plots", "learning_rate_schedule.png"))


# ---------------------------------------------------------------------------
# Plot: ANLS score distribution over val/test samples
# ---------------------------------------------------------------------------

def plot_anls_distribution(result: "EvalResult", output_dir: str) -> None:
    plt = _plt()
    scores = [s["anls"] for s in result.per_sample]
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    ax.axvline(result.anls, color="tomato", linestyle="--",
               label=f"Mean ANLS = {result.anls_pct:.2f}%")
    ax.set_xlabel("ANLS Score")
    ax.set_ylabel("# Samples")
    ax.set_title(f"ANLS Distribution — Stage {result.stage} / split={result.split}")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(plt, os.path.join(output_dir, "plots",
                            f"anls_distribution_stage{result.stage}_{result.split}.png"))


# ---------------------------------------------------------------------------
# Convenience: plot everything from a single log file
# ---------------------------------------------------------------------------

def plot_training_history(log_path: str, output_dir: str) -> None:
    """Generate all available charts from a training log JSON."""
    log = _load_log(log_path)
    plot_training_loss(log, output_dir)
    plot_balance_loss(log, output_dir)
    plot_eval_history(log, output_dir)
    plot_lr_schedule(log, output_dir)
    print(f"[Visualizer] All charts written to {output_dir}/plots/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    required=True, help="Path to JSON log file")
    parser.add_argument("--output", default="results/plots", help="Output directory")
    args = parser.parse_args()
    plot_training_history(args.log, args.output)
