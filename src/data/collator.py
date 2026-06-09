"""
Data collator for LEO-MINI.

Handles variable-length sequences and optional pix2struct inputs.
Pads input_ids / labels / attention_mask to the batch max length.
Stacks pixel_values when all samples have images.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


class LeoMiniCollator:
    """
    Collates a list of dataset items into a model-ready batch.

    Args:
        pad_token_id: tokenizer pad id (used to pad input_ids)
        max_length:   hard cap on sequence length after padding
    """

    def __init__(self, pad_token_id: int = 0, max_length: int = 2048) -> None:
        self.pad_token_id = pad_token_id
        self.max_length   = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}

        # --- input_ids: left-truncate, right-pad ---
        input_ids_list = [
            f["input_ids"][: self.max_length] for f in features
        ]
        batch["input_ids"] = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        # --- labels: pad with IGNORE_INDEX ---
        labels_list = [
            f["labels"][: self.max_length] for f in features
        ]
        batch["labels"] = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        # --- attention_mask ---
        attn_list = [
            f["attention_mask"][: self.max_length] for f in features
        ]
        batch["attention_mask"] = pad_sequence(
            attn_list,
            batch_first=True,
            padding_value=0,
        )

        # --- pixel_values: stack if all samples have images ---
        pv_list = [f.get("pixel_values") for f in features]
        if all(pv is not None for pv in pv_list):
            batch["pixel_values"] = torch.stack(pv_list)
        elif any(pv is not None for pv in pv_list):
            # Mixed batch: skip image samples or use per-sample loop
            # For simplicity, only include batches where all samples have images
            batch["pixel_values"] = None

        # --- pix2struct_inputs: stack flattened_patches and attention_mask ---
        p2s_list = [f.get("pix2struct_inputs") for f in features]
        if all(p is not None for p in p2s_list):
            batch["pix2struct_inputs"] = {
                "flattened_patches": torch.stack([p["flattened_patches"] for p in p2s_list]),
                "attention_mask":    torch.stack([p["attention_mask"]    for p in p2s_list]),
            }
        else:
            batch["pix2struct_inputs"] = None

        return batch
