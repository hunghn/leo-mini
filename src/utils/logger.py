"""
Structured JSON logger for Vi-LEO-MINI training and evaluation.

All training steps, evaluation results, and stage transitions are written
to a single JSON file that can be replayed for plotting or analysis.
Console output mirrors every event for realtime monitoring.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


# ---------------------------------------------------------------------------
# Core logger
# ---------------------------------------------------------------------------

class ViLeoMiniLogger:
    """
    Append-only structured JSON logger.

    The log file format:
    {
        "run_id": "...",
        "config": { ... },
        "events": [ { "type": ..., ... }, ... ]
    }
    """

    def __init__(
        self,
        log_path: str,
        run_id: str,
        config: Dict[str, Any],
    ) -> None:
        self.log_path = log_path
        self.run_id   = run_id
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

        self._data: Dict[str, Any] = {
            "run_id": run_id,
            "config": config,
            "events": [],
        }
        self._flush()
        print(f"[Logger] Writing to {log_path}")

    # ------------------------------------------------------------------
    def _now(self) -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _flush(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def _append(self, event: Dict[str, Any]) -> None:
        event.setdefault("timestamp", self._now())
        self._data["events"].append(event)
        self._flush()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_train_step(
        self,
        stage: int,
        global_step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        balance_loss: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        **extra: Any,
    ) -> None:
        event: Dict[str, Any] = {
            "type":          "train_step",
            "stage":         stage,
            "global_step":   global_step,
            "epoch":         round(epoch, 4),
            "loss":          round(float(loss), 6),
            "learning_rate": float(learning_rate),
        }
        if grad_norm is not None:
            event["grad_norm"] = round(float(grad_norm), 6)
        if balance_loss is not None:
            event["balance_loss"] = round(float(balance_loss), 6)
        if tokens_per_sec is not None:
            event["tokens_per_sec"] = round(float(tokens_per_sec), 1)
        event.update(extra)
        self._append(event)

        bal_str = f"  balance={balance_loss:.4f}" if balance_loss is not None else ""
        print(
            f"[S{stage} step={global_step:>6} ep={epoch:.3f}] "
            f"loss={loss:.4f}  lr={learning_rate:.2e}{bal_str}"
        )

    def log_eval(
        self,
        stage: int,
        global_step: int,
        split: str,
        anls: float,
        em: float,
        f1: float,
        n_samples: int = 0,
        per_sample: Optional[List[Dict]] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "type":        "eval",
            "stage":       stage,
            "global_step": global_step,
            "split":       split,
            "anls":        round(float(anls), 4),
            "em":          round(float(em), 4),
            "f1":          round(float(f1), 4),
            "n_samples":   n_samples,
        }
        if per_sample is not None:
            event["per_sample"] = per_sample
        self._append(event)
        print(
            f"[EVAL S{stage} step={global_step} {split}] "
            f"ANLS={anls:.4f}  EM={em*100:.2f}%  F1={f1*100:.2f}%  (n={n_samples})"
        )

    def log_stage_end(
        self,
        stage: int,
        duration_sec: float,
        checkpoint_path: str,
        best_loss: Optional[float] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "type":             "stage_end",
            "stage":            stage,
            "duration_sec":     round(duration_sec, 1),
            "checkpoint_path":  checkpoint_path,
        }
        if best_loss is not None:
            event["best_loss"] = round(float(best_loss), 6)
        self._append(event)
        mins = duration_sec / 60
        print(f"[Stage {stage} END] {mins:.1f} min  ckpt={checkpoint_path}")

    def log_model_info(self, info: Dict[str, Any]) -> None:
        self._append({"type": "model_info", **info})

    def log_custom(self, **kwargs: Any) -> None:
        self._append(dict(kwargs))

    # ------------------------------------------------------------------
    # Convenience: get all events of a given type
    # ------------------------------------------------------------------

    def get_events(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self._data["events"] if e.get("type") == event_type]


# ---------------------------------------------------------------------------
# HuggingFace TrainerCallback integration
# ---------------------------------------------------------------------------

class LeoMiniLoggingCallback(TrainerCallback):
    """
    Hooks into HuggingFace Trainer to mirror train-step logs to ViLeoMiniLogger.
    Attach via trainer.add_callback(LeoMiniLoggingCallback(logger, stage)).
    """

    def __init__(self, logger: ViLeoMiniLogger, stage: int) -> None:
        self.logger      = logger
        self.stage       = stage
        self._t0         = time.time()

    def on_log(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        logs:    Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        self.logger.log_train_step(
            stage=self.stage,
            global_step=state.global_step,
            epoch=state.epoch or 0.0,
            loss=logs["loss"],
            learning_rate=logs.get("learning_rate", 0.0),
            grad_norm=logs.get("grad_norm"),
        )

    def on_train_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.logger.log_stage_end(
            stage=self.stage,
            duration_sec=time.time() - self._t0,
            checkpoint_path=args.output_dir,
        )


# ---------------------------------------------------------------------------
# Run-id helper
# ---------------------------------------------------------------------------

def make_run_id(model_name: str, stage: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = model_name.replace("/", "_").replace("-", "_")
    return f"{safe}_stage{stage}_{ts}"
