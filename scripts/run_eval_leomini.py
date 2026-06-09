#!/usr/bin/env python3
"""
Entry-point wrapper for lmms-eval that registers the LEO-MINI model adapter
before delegating to the standard lmms-eval CLI.

Usage (called automatically by src/eval/evaluator.py):
    python scripts/run_eval_leomini.py \\
        --model leomini \\
        --model_args pretrained=/path/to/llm,stage3_weights=/path/to/ckpt \\
        --tasks mme,pope \\
        --batch_size 1 \\
        --output_path results/

Running this script directly works the same way as `lmms-eval ...` but
with the "leomini" model name available.
"""

import os
import runpy
import sys

# Ensure the repo root is on sys.path so `from src.eval.lmms_adapter import ...`
# resolves correctly regardless of the working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import the adapter — the @register_model("leomini") decorator fires here.
import src.eval.lmms_adapter  # noqa: F401, E402

# Hand off to lmms-eval's __main__ with the original sys.argv intact.
# runpy.run_module respects sys.argv, so all CLI flags pass through unchanged.
runpy.run_module("lmms_eval", run_name="__main__")
