"""
Dataset loaders for LEO-MINI training stages.

Stage 1+2 — EAGLE dataset (Shi et al., 2024):
  Format: JSON list of conversation dicts with image paths
  {"id": ..., "image": "path/to/img.jpg", "conversations": [...]}

Stage 3   — LLaVA-v1.5 665K instruction data:
  Same JSON format as EAGLE; image paths rooted at a configurable base dir.

Both datasets are served as HuggingFace-style __getitem__ returning:
  {
    "input_ids":          LongTensor  (L,)
    "labels":             LongTensor  (L,)  — -100 at prompt, valid at response
    "pixel_values":       FloatTensor (3, H, W)
    "pix2struct_inputs":  dict  (optional)
    "attention_mask":     LongTensor  (L,)
  }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor

from ..models.leo_mini import IMAGE_TOKEN, IMAGE_TOKEN_INDEX


IGNORE_INDEX = -100  # label mask for prompt tokens


# ---------------------------------------------------------------------------
# Conversation formatting (LLaVA / EAGLE style)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def format_conversation(
    conversations: List[Dict[str, str]],
    image_token:   str = IMAGE_TOKEN,
    add_image_at:  int = 0,  # insert <image> before this turn's content
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Convert LLaVA-style conversation list to a flat string and return
    the (start, end) byte offsets of each assistant turn for label masking.

    conversations: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
    """
    text = SYSTEM_PROMPT + "\n"
    assistant_ranges: List[Tuple[int, int]] = []

    for i, turn in enumerate(conversations):
        role = "USER" if turn["from"] == "human" else "ASSISTANT"
        content = turn["value"]

        # Insert image token into first human turn if not already present
        if i == 0 and role == "USER" and image_token not in content:
            content = f"{image_token}\n{content}"

        if role == "USER":
            text += f"USER: {content} "
        else:
            start = len(text)
            text += f"ASSISTANT: {content}{chr(2)}"  # chr(2) = EOS surrogate
            end = len(text)
            assistant_ranges.append((start, end))

    return text, assistant_ranges


from typing import Tuple  # noqa: E402


# ---------------------------------------------------------------------------
# Base multimodal dataset
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """
    Loads conversation + image data in LLaVA / EAGLE JSON format.

    Args:
        data_path:     path to JSON list file
        image_dir:     root directory for images
        tokenizer:     HuggingFace tokenizer
        image_processor: CLIPImageProcessor (for pixel_values)
        pix2struct_processor: Pix2Struct processor (optional)
        max_length:    maximum sequence length
    """

    def __init__(
        self,
        data_path:           str,
        image_dir:           str,
        tokenizer:           AutoTokenizer,
        image_processor:     CLIPImageProcessor,
        pix2struct_processor: Optional[Any] = None,
        max_length:          int = 2048,
    ) -> None:
        super().__init__()
        self.image_dir            = image_dir
        self.tokenizer            = tokenizer
        self.image_processor      = image_processor
        self.pix2struct_processor = pix2struct_processor
        self.max_length           = max_length

        with open(data_path) as f:
            self.data: List[Dict] = json.load(f)

        print(f"Loaded {len(self.data):,} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        conversations = sample.get("conversations", sample.get("conversation", []))
        image_file    = sample.get("image", None)

        # --- Tokenize conversation ---
        text, asst_ranges = format_conversation(conversations)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]   # (L,)

        # Replace <image> token id with IMAGE_TOKEN_INDEX sentinel
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX

        # --- Build labels: mask prompt tokens with IGNORE_INDEX ---
        labels = input_ids.clone()
        # Mask everything before the first assistant response
        for start_char, end_char in asst_ranges:
            # char offsets → token offsets (approximate via char_to_token)
            # We use a simpler strategy: mask prompt = everything up to ASSISTANT:
            pass  # handled below via string search in token space

        # Simple heuristic: find "ASSISTANT:" boundaries in token space
        labels = self._build_labels(input_ids, text)

        # --- Load and process image ---
        pixel_values    = None
        pix2struct_dict = None
        if image_file is not None:
            img_path = os.path.join(self.image_dir, image_file)
            try:
                image = Image.open(img_path).convert("RGB")
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values[0]  # (3, H, W)

                if self.pix2struct_processor is not None:
                    p2s = self.pix2struct_processor(
                        images=image,
                        text="",
                        return_tensors="pt",
                        max_patches=576,
                    )
                    pix2struct_dict = {
                        "flattened_patches": p2s.flattened_patches[0],
                        "attention_mask":    p2s.attention_mask[0],
                    }
            except Exception as e:
                print(f"Warning: failed to load image {img_path}: {e}")

        result: Dict[str, Any] = {
            "input_ids":   input_ids,
            "labels":      labels,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if pix2struct_dict is not None:
            result["pix2struct_inputs"] = pix2struct_dict

        return result

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        text: str,
    ) -> torch.Tensor:
        """
        Mask prompt tokens: assign IGNORE_INDEX to all positions that
        are part of the system prompt or USER turns.  Only ASSISTANT
        tokens are supervised.
        """
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Decode token by token to find ASSISTANT responses
        # Simpler: find "ASSISTANT:" prefix in the tokenised sequence
        asst_token_ids = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

        ids_list = input_ids.tolist()
        n = len(asst_token_ids)

        i = 0
        while i < len(ids_list):
            if ids_list[i : i + n] == asst_token_ids:
                # Start of assistant response — copy from end of "ASSISTANT:" marker
                j = i + n
                while j < len(ids_list):
                    # Find EOS (2) or end of sequence
                    labels[j] = ids_list[j]
                    if ids_list[j] == self.tokenizer.eos_token_id:
                        j += 1
                        break
                    j += 1
                i = j
            else:
                i += 1

        return labels


# ---------------------------------------------------------------------------
# Concrete dataset aliases
# ---------------------------------------------------------------------------

class EAGLEDataset(MultimodalDataset):
    """Stage 1+2 — EAGLE alignment / SFT data."""
    pass


class LLaVADataset(MultimodalDataset):
    """Stage 3 — LLaVA-v1.5 665K instruction data."""
    pass
