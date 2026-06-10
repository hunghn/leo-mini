"""
Benchmark Evaluator for LEO-MINI (Section 4, Appendix A.1).

Uses the lmms-eval framework (same as LLaVA) to evaluate on the 12 benchmarks
reported in the paper.  Can also run a quick evaluation on a small subset for
rapid verification (used in the Colab demo).

Benchmarks (Table 1, 2, 3 of the paper):
  MME, MMBench, SEED-Bench, GQA, ScienceQA, MMMU, POPE,
  AI2D, TextVQA, ChartQA, OCRBench, VizWiz

Usage (GPU server, full eval):
  python -m src.eval.evaluator \
      --model_path checkpoints/leomini_stage3 \
      --tasks mme,mmbench,seedbench,gqa,scienceqa,mmmu,pope,ai2d,textvqa,chartqa,ocrbench \
      --output_dir results/

Usage (Colab, quick eval):
  python -m src.eval.evaluator \
      --model_path /content/drive/MyDrive/leomini_stage3 \
      --tasks pope,textvqa,scienceqa \
      --limit 500 \
      --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Wrapper script that imports the LEO-MINI lmms-eval adapter before delegating
# to lmms-eval's CLI.  Using a wrapper (rather than `python -m lmms_eval`) is
# necessary because the adapter must be imported in the same process as
# lmms-eval for the @register_model decorator to take effect.
_EVAL_WRAPPER = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "run_eval_leomini.py")


# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------

BENCHMARK_CONFIG: Dict[str, Dict] = {
    "mme":        {"lmms_task": "mme",           "metric": "mme_perception_score", "target": 1583.0},
    "mmbench":    {"lmms_task": "mmbench_en_dev", "metric": "mmbench_overall",     "target": 77.0},
    "seedbench":  {"lmms_task": "seedbench",      "metric": "seedbench_overall",   "target": 75.8},
    "gqa":        {"lmms_task": "gqa",            "metric": "gqa_exact_match",     "target": 64.5},
    "scienceqa":  {"lmms_task": "scienceqa_img",  "metric": "scienceqa_acc",       "target": 84.5},
    "mmmu":       {"lmms_task": "mmmu_val",       "metric": "mmmu_acc",            "target": 38.8},
    "pope":       {"lmms_task": "pope",           "metric": "pope_acc",            "target": 90.3},
    "ai2d":       {"lmms_task": "ai2d",           "metric": "ai2d_acc",            "target": 75.7},
    "textvqa":    {"lmms_task": "textvqa_val",    "metric": "textvqa_acc",         "target": 75.1},
    "chartqa":    {"lmms_task": "chartqa",         "metric": "chartqa_relaxed_acc", "target": 80.5},
    "ocrbench":   {"lmms_task": "ocrbench",        "metric": "ocrbench_acc",        "target": 62.4},
    "vizwiz":     {"lmms_task": "vizwiz_vqa_val",  "metric": "vizwiz_vqa_acc",      "target": 69.3},
    "docvqa":     {"lmms_task": "docvqa_val",      "metric": "docvqa_acc",          "target": None},
}

ALL_TASKS = list(BENCHMARK_CONFIG.keys())


# ---------------------------------------------------------------------------
# lmms-eval wrapper
# ---------------------------------------------------------------------------

class LeoMiniEvaluator:
    """
    Wrapper around lmms-eval CLI.

    Args:
        model_path:   path to LeoMini checkpoint (or HuggingFace model id)
        model_name:   human-readable label used in comparison tables
        output_dir:   directory to save results JSON
        limit:        max samples per task (None = full eval)
        load_in_4bit: use 4-bit quantisation (for Colab T4)
        batch_size:   lmms-eval batch size
    """

    def __init__(
        self,
        model_path:       str,
        model_name:       str  = "",
        output_dir:       str  = "results",
        limit:            Optional[int] = None,
        load_in_4bit:     bool = False,
        batch_size:       int  = 1,
        num_fewshot:      int  = 0,
        extra_model_args: str  = "",
    ) -> None:
        self.model_path       = model_path
        self.model_name       = model_name or Path(model_path).name
        self.output_dir       = Path(output_dir)
        self.limit            = limit
        self.load_in_4bit     = load_in_4bit
        self.batch_size       = batch_size
        self.num_fewshot      = num_fewshot
        self.extra_model_args = extra_model_args
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run lmms-eval on the specified tasks (default: all benchmarks).
        Returns dict {task_name: score}.
        """
        if tasks is None:
            tasks = ALL_TASKS

        results: Dict[str, float] = {}
        for task in tasks:
            cfg = BENCHMARK_CONFIG.get(task)
            if cfg is None:
                print(f"Unknown task '{task}', skipping.")
                continue

            score = self._run_single_task(task, cfg["lmms_task"])
            results[task] = score
            target = cfg["target"]
            if target is not None:
                diff = f"+{score - target:.1f}" if score >= target else f"{score - target:.1f}"
                print(f"  {task:12s}: {score:.1f}  (target {target:.1f}, {diff})")
            else:
                print(f"  {task:12s}: {score:.1f}  (no target)")

        # Save results — include model_name for comparison tracking
        payload = {"model": self.model_name, "results": results}
        out_file = self.output_dir / "results.json"
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_file}")
        return results

    def _run_single_task(self, task_name: str, lmms_task: str) -> float:
        """Run lmms-eval for one task, parse and return the primary metric score."""
        log_dir = self.output_dir / task_name
        log_dir.mkdir(exist_ok=True)

        model_args = f"pretrained={self.model_path}"
        if self.load_in_4bit:
            model_args += ",load_in_4bit=True"
        if self.extra_model_args:
            model_args += f",{self.extra_model_args}"

        # Run via the wrapper so that the LEO-MINI lmms-eval adapter is
        # registered before lmms-eval parses --model leomini.
        cmd = [
            sys.executable, _EVAL_WRAPPER,
            "--model",       "leomini",
            "--model_args",  model_args,
            "--tasks",       lmms_task,
            "--num_fewshot", str(self.num_fewshot),
            "--batch_size",  str(self.batch_size),
            "--log_samples",
            "--output_path", str(log_dir),
        ]

        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]

        print(f"\n[Eval] Running {task_name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"lmms-eval failed for {task_name}:\n{result.stderr[:500]}")
            return float("nan")

        # Parse score from output JSON
        return self._parse_score(log_dir, lmms_task, task_name)

    def _parse_score(
        self, log_dir: Path, lmms_task: str, task_name: str
    ) -> float:
        """Load lmms-eval result JSON and extract the primary metric."""
        result_files = list(log_dir.glob("*.json"))
        if not result_files:
            return float("nan")
        with open(result_files[0]) as f:
            data = json.load(f)
        # lmms-eval nests results under data["results"][task_name]
        task_res = data.get("results", {}).get(lmms_task, {})
        cfg = BENCHMARK_CONFIG[task_name]
        metric = cfg["metric"]
        score = task_res.get(metric, task_res.get(f"{metric},none", float("nan")))
        return float(score) * 100 if isinstance(score, float) and score <= 1.0 else float(score)


# ---------------------------------------------------------------------------
# Ablation: vary number of visual tokens
# ---------------------------------------------------------------------------

def run_token_ablation(
    model_path:  str,
    output_dir:  str = "results/ablation",
    token_counts: List[int] = [1, 16, 64, 256],
    tasks: Optional[List[str]] = None,
) -> None:
    """
    Replicate Table 2 ablation: test different N^V token counts.
    Requires that the model checkpoint supports configuring n_visual at load time.
    """
    if tasks is None:
        tasks = ["mme", "pope", "textvqa", "scienceqa"]

    print("\n=== Visual Token Ablation ===")
    for n in token_counts:
        print(f"\n--- N^V = {n} ---")
        # n_visual is passed via model_args (lmms-eval CLI format), not URL query string
        evaluator = LeoMiniEvaluator(
            model_path=model_path,
            output_dir=os.path.join(output_dir, f"n{n}"),
            batch_size=1,
            extra_model_args=f"n_visual={n}",
        )
        evaluator.run(tasks)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(
    models:     List[Dict],
    tasks:      Optional[List[str]] = None,
    output_dir: str = "results/comparison",
    limit:      Optional[int] = None,
    load_in_4bit: bool = False,
) -> None:
    """
    Evaluate multiple LEO-MINI checkpoints and print a side-by-side table.

    Args:
        models: list of dicts, each with keys:
                  name          — display label (e.g. "Llama-3.2-1B")
                  model_path    — path to LLM checkpoint
                  stage3_weights — optional path to stage3_adapter_weights.pt
        tasks:  benchmark subset (default: pope, scienceqa, textvqa, mme, mmmu)
        output_dir: where to save per-model results and comparison.json
        limit:  max samples per task per model
        load_in_4bit: use 4-bit quant (for Colab / limited VRAM)

    Example:
        compare_models([
            {"name": "Llama-1B",  "model_path": "ckpts/1b_stage3"},
            {"name": "Llama-3B",  "model_path": "ckpts/3b_stage3"},
            {"name": "Phi-3.5",   "model_path": "ckpts/phi_stage3"},
        ])
    """
    if tasks is None:
        tasks = ["pope", "scienceqa", "textvqa", "mme", "mmmu"]

    all_results: Dict[str, Dict[str, float]] = {}

    for spec in models:
        name        = spec["name"]
        model_path  = spec["model_path"]
        stage3_wts  = spec.get("stage3_weights", "")
        model_dir   = os.path.join(output_dir, name.replace(" ", "_").replace("/", "-"))

        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"{'='*60}")

        extra = f"stage3_weights={stage3_wts}" if stage3_wts else ""
        evaluator = LeoMiniEvaluator(
            model_path=model_path,
            model_name=name,
            output_dir=model_dir,
            limit=limit,
            load_in_4bit=load_in_4bit,
            batch_size=1,
            extra_model_args=extra,
        )
        all_results[name] = evaluator.run(tasks)

    # ------------------------------------------------------------------ #
    # Print comparison table                                               #
    # ------------------------------------------------------------------ #
    col_w = 10
    model_names = [spec["name"] for spec in models]
    header_cols = ["Task", "Paper(8B)"] + model_names
    sep = "-" * (14 + (col_w + 2) * len(header_cols))

    print(f"\n{'='*60}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*60}")
    print(sep)
    print(f"  {'Task':<12s}  {'Paper(8B)':>{col_w}s}", end="")
    for name in model_names:
        print(f"  {name:>{col_w}s}", end="")
    print()
    print(sep)

    for task in tasks:
        target = BENCHMARK_CONFIG.get(task, {}).get("target")
        target_str = f"{target:.1f}" if target is not None else "   -"
        print(f"  {task:<12s}  {target_str:>{col_w}s}", end="")
        for name in model_names:
            score = all_results.get(name, {}).get(task, float("nan"))
            score_str = f"{score:.1f}" if not (score != score) else "  nan"
            delta = ""
            if target is not None and score == score:
                delta = f"({'+'if score>=target else ''}{score-target:.1f})"
            print(f"  {score_str:>{col_w}s}", end="")
        print()
    print(sep)

    # Save combined comparison JSON
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, "comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(
            {
                "tasks": tasks,
                "paper_targets": {t: BENCHMARK_CONFIG[t]["target"] for t in tasks},
                "models": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nComparison saved to {comparison_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LEO-MINI on vision-language benchmarks")
    parser.add_argument("--model_path",    default=None,
                        help="Single model checkpoint path (use --compare for multi-model)")
    parser.add_argument("--model_name",    default="",
                        help="Display name for the model (used in output)")
    parser.add_argument("--tasks",         default=",".join(ALL_TASKS))
    parser.add_argument("--output_dir",    default="results")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Samples per task (None=full eval)")
    parser.add_argument("--load_in_4bit",  action="store_true",
                        help="4-bit quantisation (Colab T4 / limited VRAM)")
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--ablation",      action="store_true",
                        help="Run visual token count ablation (Table 2)")
    # Multi-model comparison mode
    parser.add_argument("--compare",       action="store_true",
                        help="Compare multiple models side-by-side")
    parser.add_argument("--model_specs",   default=None,
                        help=(
                            "Comma-separated model specs for --compare mode. "
                            "Two formats are accepted:\n"
                            "  'Name:base_llm_path'  — Stage-2 checkpoint only\n"
                            "  'Name:base_llm_path:stage3_weights_path'  — full LEO-MINI\n"
                            "base_llm_path is passed to AutoModelForCausalLM.from_pretrained().\n"
                            "stage3_weights_path points to stage3_adapter_weights.pt.\n"
                            "Example:\n"
                            "  'Llama-1B:meta-llama/Llama-3.2-1B-Instruct:ckpts/1b/stage3_adapter_weights.pt,"
                            "Phi:microsoft/Phi-3.5-mini-instruct:ckpts/phi/stage3_adapter_weights.pt'"
                        ))
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.compare:
        if not args.model_specs:
            parser.error("--compare requires --model_specs")
        model_list = []
        for spec in args.model_specs.split(","):
            # Supported formats (both use ':' as separator):
            #   "Name:base_llm_path"
            #       → base LLM only (no Stage-3 adapters); useful for Stage-2 eval
            #   "Name:base_llm_path:stage3_weights_path"
            #       → base LLM + Stage-3 adapter file (normal LEO-MINI inference)
            #
            # base_llm_path  — HuggingFace model ID or full model checkpoint dir
            #                  passed to AutoModelForCausalLM.from_pretrained()
            # stage3_weights — path to stage3_adapter_weights.pt produced by
            #                  LeoMini.save_stage3_weights(); contains projector,
            #                  CoTR, and MMoE-LLM LoRA weights
            parts = spec.split(":", 2)
            if len(parts) < 2:
                parser.error(
                    f"Invalid --model_specs entry '{spec}'. "
                    "Expected 'Name:base_path' or 'Name:base_path:stage3_weights'."
                )
            entry = {
                "name":          parts[0].strip(),
                "model_path":    parts[1].strip(),
                "stage3_weights": parts[2].strip() if len(parts) == 3 else "",
            }
            model_list.append(entry)
        compare_models(
            models=model_list,
            tasks=tasks,
            output_dir=args.output_dir,
            limit=args.limit,
            load_in_4bit=args.load_in_4bit,
        )

    elif args.ablation:
        if not args.model_path:
            parser.error("--ablation requires --model_path")
        run_token_ablation(args.model_path, args.output_dir, tasks=tasks)

    else:
        if not args.model_path:
            parser.error("--model_path required (or use --compare)")
        evaluator = LeoMiniEvaluator(
            model_path=args.model_path,
            model_name=args.model_name,
            output_dir=args.output_dir,
            limit=args.limit,
            load_in_4bit=args.load_in_4bit,
            batch_size=args.batch_size,
        )
        results = evaluator.run(tasks)

        print("\n=== Summary ===")
        for task, score in results.items():
            target = BENCHMARK_CONFIG[task]["target"]
            if target is not None:
                status = "✓" if score >= target else "✗"
                print(f"  {status} {task:12s}: {score:.1f} / {target:.1f}")
            else:
                print(f"  - {task:12s}: {score:.1f}")
