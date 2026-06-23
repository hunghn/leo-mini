"""
ViTextVQA Dataset — minhquan6203/ViTextVQA on HuggingFace.

Downloads the dataset automatically on first use via the `datasets` library.
Converts each QA pair into Qwen2.5 chat format and returns the same dict
schema as the existing MultimodalDataset so the collator and trainer work
unchanged.

HuggingFace dataset schema (minhquan6203/ViTextVQA):
  image       — PIL.Image.Image
  question    — str  (Vietnamese)
  answers     — list[str]  (multiple annotator answers)
  image_name  — str  (optional, for logging)

Output per __getitem__:
  {
    "input_ids":         LongTensor (L,)
    "labels":            LongTensor (L,)   -100 at prompt
    "attention_mask":    LongTensor (L,)
    "pixel_values":      FloatTensor (3, H, W)
    "pix2struct_inputs": dict | None
    # for evaluation only (not used during training):
    "_question":         str
    "_answers":          list[str]
    "_image_name":       str
  }
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor

from ..models.leo_mini import IMAGE_TOKEN, IMAGE_TOKEN_INDEX

IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Qwen2.5 chat template constants
# ---------------------------------------------------------------------------

VI_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thông minh, hãy trả lời câu hỏi dựa trên nội dung hình ảnh "
    "bằng tiếng Việt một cách chính xác và ngắn gọn."
)

# Qwen2.5 uses <|im_start|> / <|im_end|> as role delimiters.
# IMAGE_TOKEN is inserted inside the user turn before the question.
def _qwen2_format(question: str, answer: str, image_token: str = IMAGE_TOKEN) -> str:
    """Build a full Qwen2.5 single-turn conversation string."""
    return (
        f"<|im_start|>system\n{VI_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_token}\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def _qwen2_prompt_only(question: str, image_token: str = IMAGE_TOKEN) -> str:
    """Build prompt-only string (used for inference / label masking boundary)."""
    return (
        f"<|im_start|>system\n{VI_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_token}\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Label-building helper
# ---------------------------------------------------------------------------

def _build_labels_qwen2(
    input_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Mask all prompt tokens with IGNORE_INDEX.
    The prompt ends just before the assistant response text starts.
    We find the split by matching the tokenised prompt prefix.
    """
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    prompt_len = len(prompt_ids)

    # Verify the prefix actually matches (truncation may shorten it)
    if input_ids[:prompt_len].tolist() == prompt_ids.tolist():
        labels[prompt_len:] = input_ids[prompt_len:]
    else:
        # Fallback: find the last <|im_start|>assistant boundary via token search.
        # Token id 77091 = "assistant" is variable across vocab; search by prompt
        # length heuristic — mask the first prompt_len tokens approximately.
        safe_len = min(prompt_len, len(input_ids) - 1)
        labels[safe_len:] = input_ids[safe_len:]

    return labels


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class ViTextVQADataset(Dataset):
    """
    ViTextVQA dataset with automatic HuggingFace download.

    Args:
        split:               "train" | "validation" | "test"
        tokenizer:           Qwen2.5 tokenizer (or any AutoTokenizer)
        image_processor:     CLIPImageProcessor for pixel_values
        pix2struct_processor: Pix2Struct AutoProcessor (optional)
        max_length:          maximum tokenized sequence length
        hf_dataset_name:     HuggingFace dataset repo id
        cache_dir:           local cache for HF datasets
        for_eval:            if True, also return _question/_answers/_image_name
    """

    HF_DATASET = "minhquan6203/ViTextVQA"

    def __init__(
        self,
        split:                str,
        tokenizer:            AutoTokenizer,
        image_processor:      CLIPImageProcessor,
        pix2struct_processor: Optional[Any]   = None,
        max_length:           int             = 2048,
        hf_dataset_name:      str             = HF_DATASET,
        cache_dir:            Optional[str]   = None,
        for_eval:             bool            = False,
    ) -> None:
        super().__init__()
        self.tokenizer            = tokenizer
        self.image_processor      = image_processor
        self.pix2struct_processor = pix2struct_processor
        self.max_length           = max_length
        self.for_eval             = for_eval

        print(f"[ViTextVQA] Downloading split='{split}' from {hf_dataset_name} ...")
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        self._ds = load_dataset(hf_dataset_name, split=split, cache_dir=cache_dir)
        print(f"[ViTextVQA] Loaded {len(self._ds):,} samples (split={split})")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._ds[idx]
        question   = sample.get("question", "")
        answers    = self._parse_answers(sample)
        image_obj  = sample.get("image")
        image_name = sample.get("image_name", sample.get("img_name", str(idx)))

        # Use first answer for training (covers Stage 1/2/3)
        answer = answers[0] if answers else ""

        # ------------------------------------------------------------------
        # Tokenize Qwen2.5 conversation
        # ------------------------------------------------------------------
        full_text   = _qwen2_format(question, answer)
        prompt_text = _qwen2_prompt_only(question)

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids  = full_enc["input_ids"][0]   # (L,)
        prompt_ids = prompt_enc["input_ids"][0]

        # Replace <image> token id with sentinel
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if image_token_id is not None and image_token_id != self.tokenizer.unk_token_id:
            input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX

        labels = _build_labels_qwen2(input_ids, prompt_ids)

        attention_mask = (input_ids != (self.tokenizer.pad_token_id or 0)).long()

        # ------------------------------------------------------------------
        # Image processing
        # ------------------------------------------------------------------
        pixel_values    = None
        pix2struct_dict = None

        if image_obj is not None:
            image = self._to_pil(image_obj)
            try:
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values[0]   # (3, H, W)
            except Exception as e:
                print(f"[ViTextVQA] CLIP preprocess failed for idx={idx}: {e}")

            if self.pix2struct_processor is not None:
                try:
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
                    print(f"[ViTextVQA] Pix2Struct preprocess failed for idx={idx}: {e}")

        result: Dict[str, Any] = {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if pix2struct_dict is not None:
            result["pix2struct_inputs"] = pix2struct_dict

        if self.for_eval:
            result["_question"]   = question
            result["_answers"]    = answers
            result["_image_name"] = image_name
            result["_image"]      = self._to_pil(image_obj) if image_obj is not None else None

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_answers(sample: Dict[str, Any]) -> List[str]:
        """
        Robustly extract the list of GT answers from different field names.
        minhquan6203/ViTextVQA may use 'answers', 'answer', or nested dicts.
        """
        raw = sample.get("answers", sample.get("answer", []))
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            out = []
            for a in raw:
                if isinstance(a, str):
                    out.append(a)
                elif isinstance(a, dict):
                    # Some VQA datasets wrap answers as {"answer": "...", "answer_confidence": "..."}
                    out.append(a.get("answer", ""))
            return [a for a in out if a]
        return []

    @staticmethod
    def _to_pil(image_obj: Any) -> Image.Image:
        """Convert various image representations to PIL.Image.Image."""
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")
        if isinstance(image_obj, bytes):
            return Image.open(io.BytesIO(image_obj)).convert("RGB")
        if isinstance(image_obj, dict):
            # datasets library stores images as {"bytes": b"...", "path": "..."}
            if "bytes" in image_obj and image_obj["bytes"]:
                return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            if "path" in image_obj and image_obj["path"]:
                return Image.open(image_obj["path"]).convert("RGB")
        raise ValueError(f"Cannot convert image object of type {type(image_obj)} to PIL")
