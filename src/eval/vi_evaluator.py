"""
Vi-LEO-MINI Evaluator — ViTextVQA benchmark.

Runs the model on val or test split, computes ANLS / EM / F1,
saves per-sample results as JSON, and triggers chart generation.

Usage (standalone):
    python -m src.eval.vi_evaluator \\
        --model_path Qwen/Qwen2.5-3B-Instruct \\
        --stage3_weights checkpoints/qwen2_5_3b_vi/stage3/stage3_adapter_weights.pt \\
        --split test \\
        --output_dir results/vi_leomini/ \\
        --log_dir logs/ \\
        [--load_in_4bit] [--limit 200] [--batch_size 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, AutoProcessor

from .metrics import compute_dataset_metrics
from ..utils.logger import ViLeoMiniLogger, make_run_id
from ..data.vitextvqa_dataset import ViTextVQADataset
from ..models.leo_mini import LeoMini, IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from ..models.vision_experts import Pix2StructExpert


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    stage:       int
    split:       str
    anls:        float
    em:          float
    f1:          float
    anls_pct:    float
    em_pct:      float
    f1_pct:      float
    n_samples:   int
    per_sample:  List[Dict[str, Any]] = field(default_factory=list)
    timestamp:   str                  = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage":      self.stage,
            "split":      self.split,
            "anls":       self.anls,
            "em":         self.em,
            "f1":         self.f1,
            "anls_pct":   self.anls_pct,
            "em_pct":     self.em_pct,
            "f1_pct":     self.f1_pct,
            "n_samples":  self.n_samples,
            "per_sample": self.per_sample,
            "timestamp":  self.timestamp,
        }

    def summary_str(self) -> str:
        return (
            f"ANLS={self.anls_pct:.2f}%  "
            f"EM={self.em_pct:.2f}%  "
            f"F1={self.f1_pct:.2f}%  "
            f"(n={self.n_samples})"
        )


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class ViTextVQAEvaluator:
    """
    Runs inference on ViTextVQA and computes metrics.

    Args:
        model:          LeoMini instance (already loaded)
        stage:          which training stage this checkpoint comes from
        device:         torch device string, e.g. "cuda:0"
        max_new_tokens: generation budget per answer
        batch_size:     inference batch size (1 recommended for variable-len)
    """

    def __init__(
        self,
        model:          LeoMini,
        stage:          int,
        device:         str  = "cuda",
        max_new_tokens: int  = 64,
        batch_size:     int  = 1,
        logger:         Optional[ViLeoMiniLogger] = None,
    ) -> None:
        self.model          = model
        self.stage          = stage
        self.device         = device
        self.max_new_tokens = max_new_tokens
        self.batch_size     = batch_size
        self.logger         = logger

        self.model.eval()

    # ------------------------------------------------------------------
    def evaluate(
        self,
        split:           str,
        hf_dataset_name: str           = "minhquan6203/ViTextVQA",
        cache_dir:       Optional[str] = None,
        image_dir:       Optional[str] = None,
        limit:           Optional[int] = None,
        global_step:     int           = 0,
        pix2struct_model_name: str     = "google/pix2struct-large",
        verbose:         bool          = False,
        ocr_json:        Optional[str] = None,
    ) -> EvalResult:
        """Run evaluation on the given split. Returns EvalResult."""
        tokenizer       = self.model.tokenizer
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        _has_pix2struct = any(isinstance(e, Pix2StructExpert) for e in self.model.vision.experts)
        pix2struct_proc = (
            AutoProcessor.from_pretrained(pix2struct_model_name)
            if _has_pix2struct else None
        )

        dataset = ViTextVQADataset(
            split=split,
            tokenizer=tokenizer,
            image_processor=image_processor,
            pix2struct_processor=pix2struct_proc,
            max_length=256,   # inference: question-only, shorter is fine
            hf_dataset_name=hf_dataset_name,
            cache_dir=cache_dir,
            for_eval=True,
            image_dir=image_dir,
            ocr_json=ocr_json,
        )

        return self.evaluate_dataset(
            dataset, limit=limit, global_step=global_step, split=split, verbose=verbose
        )

    # ------------------------------------------------------------------
    def evaluate_dataset(
        self,
        dataset,
        limit:       Optional[int] = None,
        global_step: int           = 0,
        split:       str           = "validation",
        verbose:     bool          = False,
    ) -> EvalResult:
        """
        Run evaluation on an already-built dataset (for_eval=True).

        Split out from evaluate() so callers that need to run this repeatedly
        (e.g. mid-training validation, see LeoMiniMidTrainEvalCallback in
        trainer.py) can build the ViTextVQADataset once and reuse it across
        calls, instead of re-downloading/re-parsing the split JSON every time.
        Does not mutate `dataset` — `limit` only bounds the loop range.

        verbose: print image name, question, prediction, and ground truth for
            every sample as it runs (not just the aggregate every-100 progress
            line). Meant for small eval_limit runs (mid-training monitoring),
            not for full-dataset eval where it would flood the console.
        """
        total = len(dataset) if limit is None else min(limit, len(dataset))

        predictions: List[str]         = []
        ground_truths: List[List[str]] = []

        print(f"[Eval] Running inference on {total} samples (split={split}) ...")
        t0 = time.time()

        for idx in range(total):
            sample = dataset[idx]
            question   = sample.get("_question", "")
            answers    = sample.get("_answers", [])
            image_name = sample.get("_image_name", str(idx))

            pred = self._generate_one(sample)
            predictions.append(pred)
            ground_truths.append(answers)

            if verbose:
                anls_i = compute_dataset_metrics([pred], [answers])["anls"]
                print(
                    f"  [{idx+1}/{total}] Đang validation hình '{image_name}'\n"
                    f"    Q       : {question}\n"
                    f"    predict : {pred!r}\n"
                    f"    ground truth: {answers}\n"
                    f"    ANLS    : {anls_i:.2f}"
                )
            elif (idx + 1) % 100 == 0 or idx == total - 1:
                elapsed = time.time() - t0
                print(f"  [{idx+1}/{total}]  {elapsed:.0f}s elapsed")

        # ------------------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------------------
        metrics = compute_dataset_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
        )

        result = EvalResult(
            stage=self.stage,
            split=split,
            anls=metrics["anls"],
            em=metrics["em"],
            f1=metrics["f1"],
            anls_pct=metrics["anls_pct"],
            em_pct=metrics["em_pct"],
            f1_pct=metrics["f1_pct"],
            n_samples=metrics["n_samples"],
            per_sample=metrics["per_sample"],
        )

        if self.logger is not None:
            self.logger.log_eval(
                stage=self.stage,
                global_step=global_step,
                split=split,
                anls=result.anls,
                em=result.em,
                f1=result.f1,
                n_samples=result.n_samples,
                per_sample=metrics["per_sample"],
            )

        print(f"[Eval done] {result.summary_str()}")
        return result

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_one(self, sample: Dict[str, Any]) -> str:
        """Run single-sample inference and return the decoded prediction."""
        tokenizer = self.model.tokenizer

        input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)

        pixel_values = None
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            pixel_values = sample["pixel_values"].unsqueeze(0).to(self.device)

        pix2struct_inputs = None
        if "pix2struct_inputs" in sample and sample["pix2struct_inputs"] is not None:
            pix2struct_inputs = {
                k: v.unsqueeze(0).to(self.device)
                for k, v in sample["pix2struct_inputs"].items()
            }

        out_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pix2struct_inputs=pix2struct_inputs,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

        # LeoMini.generate() calls self.llm.generate(inputs_embeds=...) internally,
        # so out_ids contains ONLY the newly generated tokens (no prompt prefix).
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        return decoded


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_eval_result(result: EvalResult, output_dir: str, tag: str = "") -> str:
    """
    Save EvalResult to JSON.
    Returns the path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"eval_stage{result.stage}_{result.split}"
    if tag:
        fname += f"_{tag}"
    fname += f"_{ts}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[Eval] Results saved to {path}")
    return path


def load_eval_history(output_dir: str) -> List[Dict[str, Any]]:
    """Load all eval JSON files from output_dir, sorted by timestamp."""
    results = []
    if not os.path.isdir(output_dir):
        return results
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("eval_") and fname.endswith(".json"):
            with open(os.path.join(output_dir, fname), encoding="utf-8") as f:
                results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Vi-LEO-MINI evaluator")
    parser.add_argument("--model_path",    required=True, help="Base LLM HF path or local dir")
    parser.add_argument(
        "--projector_path",
        default=None,
        help=(
            "Path to projector_weights.pt saved after Stage 1/2. Auto-derived as "
            "the sibling 'projector_weights.pt' of --model_path if not set. "
            "Without it (and without --stage3_weights, which also carries the "
            "projector) the VisualProjector is randomly initialised."
        ),
    )
    parser.add_argument("--stage3_weights", default=None,  help="Path to stage3_adapter_weights.pt")
    parser.add_argument("--split",          default="test", choices=["train","validation","test"])
    parser.add_argument("--output_dir",     default="results/vi_leomini")
    parser.add_argument("--log_dir",        default="logs")
    parser.add_argument("--stage",          type=int, default=3)
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample question/prediction/ground-truth/ANLS while running. "
             "Use with a small --limit to diagnose why metrics are 0.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--load_in_4bit",   action="store_true")
    parser.add_argument("--vision_experts", nargs="+", default=["clip","pix2struct"])
    parser.add_argument("--n_visual",       type=int, default=128)
    parser.add_argument("--pix2struct_model_name", default="google/pix2struct-large")
    parser.add_argument("--hf_dataset",     default="minhquan6203/ViTextVQA")
    # Note: The default hf_dataset is for Stage 3. For other stages, you must
    # override this, e.g., --hf_dataset "uitnlp/OpenViVQA-dataset" for Stage 2.
    parser.add_argument("--cache_dir",      default=None)
    parser.add_argument(
        "--ocr_json",
        default=None,
        help=(
            "Path to OCR map JSON {image_name: text} from scripts/precompute_ocr.py. "
            "When set, OCR text is prepended to every question (must match how the "
            "checkpoint was trained)."
        ),
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help=(
            "Local directory containing images by filename. Required when the HF "
            "dataset does not embed image bytes (e.g. ViTextVQA images must be "
            "downloaded separately from textvqa.org and placed here)."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve projector_path: explicit > auto-derived sibling of model_path.
    # stage3_weights (if given) also carries the projector and is loaded
    # afterwards in from_pretrained(), so it takes precedence either way.
    projector_path = args.projector_path
    if projector_path is None:
        candidate = os.path.join(
            os.path.dirname(os.path.normpath(args.model_path)), "projector_weights.pt"
        )
        if os.path.isfile(candidate):
            projector_path = candidate
            print(f"[Eval] Auto-derived projector_path: {projector_path}")
        elif not args.stage3_weights:
            print(
                f"\nERROR: No --projector_path given and none found at '{candidate}'.\n"
                f"  Without it the VisualProjector is randomly initialised, and the\n"
                f"  model will effectively be blind to the image regardless of\n"
                f"  --model_path. Pass --projector_path explicitly, or --stage3_weights\n"
                f"  (which bundles the trained projector).\n",
                file=sys.stderr,
            )
            sys.exit(1)

    run_id  = make_run_id(os.path.basename(args.model_path), args.stage)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = ViLeoMiniLogger(
        log_path=os.path.join(args.log_dir, f"{run_id}_eval.json"),
        run_id=run_id,
        config=vars(args),
    )

    print(f"Loading model from {args.model_path} ...")
    model = LeoMini.from_pretrained(
        llm_path=args.model_path,
        projector_path=projector_path,
        stage3_weights=args.stage3_weights,
        enable_stage3_modules=bool(args.stage3_weights),
        n_visual=args.n_visual,
        load_in_4bit=args.load_in_4bit,
        vision_experts=args.vision_experts,
        pix2struct_model_name=args.pix2struct_model_name,
    )
    model.to(device)

    evaluator = ViTextVQAEvaluator(
        model=model,
        stage=args.stage,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        logger=logger,
    )

    result = evaluator.evaluate(
        split=args.split,
        hf_dataset_name=args.hf_dataset,
        cache_dir=args.cache_dir,
        image_dir=args.image_dir or None,
        limit=args.limit,
        verbose=args.verbose,
        ocr_json=args.ocr_json,
    )

    save_eval_result(result, args.output_dir)

    # Trigger chart generation
    try:
        from ..utils.visualizer import plot_training_history, plot_anls_distribution
        log_files = [
            os.path.join(args.log_dir, f)
            for f in os.listdir(args.log_dir)
            if f.endswith(".json")
        ]
        if log_files:
            plot_training_history(log_files[0], args.output_dir)
        plot_anls_distribution(result, args.output_dir)
    except Exception as e:
        print(f"[Eval] Chart generation skipped: {e}")


if __name__ == "__main__":
    _cli()
