"""
Vietnamese Multimodal Dataset Loader.

Downloads the dataset automatically on first use via the `datasets` library.
Converts each QA pair into Qwen2.5 chat format and returns the same dict
schema as the existing MultimodalDataset so the collator and trainer work
unchanged.
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional
import zipfile

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
# Load-error helper (module-level so _load_with_fallback can reference it)
# ---------------------------------------------------------------------------

def _raise_load_error(hf_name: str, split: str, exc: Exception) -> None:
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    cause_str = f"\n  Root cause: {type(cause).__name__}: {cause}" if cause else f"\n  Error: {exc}"
    raise RuntimeError(
        f"\n[ViTextVQA] All loading attempts failed for '{hf_name}' split='{split}'.{cause_str}\n\n"
        f"  Manual fixes to try in order:\n"
        f"  1. rm -rf ~/.cache/huggingface/datasets/minhquan6203*\n"
        f"  2. pip install --upgrade 'datasets>=2.14' Pillow pandas pyarrow\n"
        f"  3. df -h ~/.cache  (check disk space)\n"
        f"  4. hf login  (if dataset requires auth)\n"
        f"  5. Set vi_cache_dir to a different path in model config\n"
    ) from exc


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class VietnameseMultimodalDataset(Dataset):
    """
    Base class for Vietnamese multimodal datasets.

    Args:
        split:               "train" | "validation" | "test"
        tokenizer:           Qwen2.5 tokenizer (or any AutoTokenizer)
        image_processor:     CLIPImageProcessor for pixel_values
        pix2struct_processor: Pix2Struct AutoProcessor (optional)
        max_length:          maximum tokenized sequence length
        hf_dataset_name:     HuggingFace dataset repo id
        cache_dir:           local cache for HF datasets
        for_eval:            if True, also return _question/_answers/_image_name
        image_dir:           local directory containing images by name (used when
                             the HF dataset does not embed image bytes, e.g.
                             ViTextVQA whose images must be downloaded separately
                             from TextVQA). When set, falls back to loading
                             os.path.join(image_dir, image_name) when the
                             dataset's "image" field is None.
    """

    def __init__(
        self,
        split:                str,
        tokenizer:            AutoTokenizer,
        image_processor:      CLIPImageProcessor,
        pix2struct_processor: Optional[Any]   = None,
        max_length:           int             = 2048,
        hf_dataset_name:      str             = "minhquan6203/ViTextVQA",
        cache_dir:            Optional[str]   = None,
        for_eval:             bool            = False,
        image_dir:            Optional[str]   = None,
    ) -> None:
        super().__init__()
        self.tokenizer            = tokenizer
        self.image_processor      = image_processor
        self.pix2struct_processor = pix2struct_processor
        self.max_length           = max_length
        self.for_eval             = for_eval
        self.image_dir            = image_dir

        print(f"[VietnameseDataset] Downloading split='{split}' from {hf_dataset_name} ...")
        try:
            from datasets import load_dataset, Image as HFImage, DownloadMode
        except ImportError as e:
            raise ImportError("pip install 'datasets>=2.14'") from e

        ds = self._load_with_fallback(
            hf_dataset_name, split, cache_dir, HFImage, DownloadMode
        )

        # Disable automatic PIL decode so images arrive as raw bytes dicts
        # {"bytes": ..., "path": ...}. We decode in __getitem__ via _to_pil(),
        # which gives per-sample error control and avoids bulk decode failures.
        # For our custom-loaded datasets (ViTextVQA), the "image" column may be a
        # plain string path (Value("string")), which _to_pil() handles directly.
        # We still attempt the cast but skip it on failure rather than crashing.
        if "image" in ds.column_names:
            try:
                ds = ds.cast_column("image", HFImage(decode=False))
            except Exception:
                pass  # string-path column: _to_pil(str) handles it in __getitem__

        self._ds = ds
        print(f"[VietnameseDataset] Loaded {len(self._ds):,} samples (split={split})")

    # ------------------------------------------------------------------
    # Robust dataset loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_with_fallback(hf_name, split, cache_dir, HFImage, DownloadMode):
        """
        Three-attempt loading strategy:

        Pass 1 — use cached files (fast, normal path).
        Pass 2 — if cache is corrupted (ValueError: Expected object or value,
                  or any DatasetGenerationError), force a fresh download and
                  regenerate the cache.
        Pass 3 — if still failing, load as streaming and materialise to a list.
                  Slower but bypasses the Parquet/Arrow cache layer entirely.
        """
        from datasets import load_dataset

        # ViTextVQA stores QA pairs in JSON files (ViTextVQA_{split}.json) and
        # images in a separate ViTextVQA_images.zip.  The standard load_dataset
        # path treats the top-level JSON dict as a single row, giving "1 sample".
        # We must parse the JSON manually to extract individual QA rows.
        if "ViTextVQA" in hf_name:
            try:
                import json as _json
                from huggingface_hub import hf_hub_download
                from datasets import Dataset as HFDataset

                # HF repo uses "dev" for what we call "validation"
                vt_split = {"validation": "dev", "valid": "dev", "val": "dev"}.get(split, split)
                json_filename = f"ViTextVQA_{vt_split}.json"

                local_json = hf_hub_download(
                    repo_id=hf_name,
                    repo_type="dataset",
                    filename=json_filename,
                    cache_dir=cache_dir,
                )

                with open(local_json, encoding="utf-8-sig") as f:
                    raw = _json.load(f)

                # Extract the list of QA items from whatever top-level wrapper exists.
                if isinstance(raw, list):
                    rows = raw
                elif isinstance(raw, dict):
                    # Try common wrapper keys used by TextVQA-style datasets
                    rows = None
                    for key in ("data", "questions", "annotations", "items", "samples"):
                        if key in raw and isinstance(raw[key], list):
                            rows = raw[key]
                            break
                    if rows is None:
                        # Columnar format: {field: [v0, v1, ...], ...}
                        list_cols = {k: v for k, v in raw.items() if isinstance(v, list)}
                        lengths = {len(v) for v in list_cols.values()}
                        if list_cols and len(lengths) == 1:
                            n = next(iter(lengths))
                            rows = [{k: v[i] for k, v in list_cols.items()} for i in range(n)]
                        else:
                            rows = [raw]  # last resort: 1 row = whole dict
                else:
                    rows = [raw]

                # Try to download and extract the images zip (one-time setup).
                # Each row gets an "image" field set to the local file path so
                # _to_pil() can open it directly without needing vi_image_dir.
                image_paths: dict = {}
                try:
                    zip_local = hf_hub_download(
                        repo_id=hf_name,
                        repo_type="dataset",
                        filename="ViTextVQA_images.zip",
                        cache_dir=cache_dir,
                    )
                    image_root = os.path.join(
                        cache_dir or os.path.expanduser("~/.cache/huggingface/datasets"),
                        "vitextvqa_images",
                    )
                    os.makedirs(image_root, exist_ok=True)
                    if not any(os.scandir(image_root)):
                        print("[ViTextVQA] Extracting ViTextVQA_images.zip (one-time) ...")
                        with zipfile.ZipFile(zip_local) as zf:
                            zf.extractall(image_root)
                    for _root, _, _files in os.walk(image_root):
                        for _fn in _files:
                            if _fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                                image_paths[_fn] = os.path.join(_root, _fn)
                    print(f"[ViTextVQA] {len(image_paths):,} images available in {image_root}")
                except Exception as _img_err:
                    print(
                        f"[ViTextVQA] Could not download ViTextVQA_images.zip: {_img_err}\n"
                        f"  → Set vi_image_dir in your config to a local directory with the images."
                    )

                # Normalise each row: resolve image path and ensure image_name field.
                for row in rows:
                    img_ref = (
                        row.get("image")
                        or row.get("image_name")
                        or row.get("image_id")
                        or ""
                    )
                    image_filename = os.path.basename(str(img_ref)) if img_ref else ""
                    row["image_name"] = image_filename
                    if image_filename and image_filename in image_paths:
                        row["image"] = image_paths[image_filename]   # resolved local path
                    else:
                        row["image"] = None   # will use image_dir fallback in __getitem__

                return HFDataset.from_list(rows)
            except Exception as e:
                _raise_load_error(hf_name, split, e)

        # OpenViVQA has a non-standard JSON structure with a BOM and separate
        # image zips. Handle it manually. KTVIC is Parquet-backed on HF, so let
        # datasets.load_dataset handle it via the standard path below.
        if "OpenViVQA" in hf_name:
            try:
                import json
                from huggingface_hub import hf_hub_download
                from datasets import Dataset as HFDataset

                is_openvivqa = "OpenViVQA" in hf_name
                # OpenViVQA annotation files are named with the VLSP 2023
                # split convention: vlsp2023_train/dev/test_data.json.
                openvivqa_split = {
                    "validation": "dev",
                    "valid": "dev",
                    "val": "dev",
                }.get(split, split)
                repo_file_path = f"vlsp2023_{openvivqa_split}_data.json"

                try:
                    local_file_path = hf_hub_download(
                        repo_id=hf_name,
                        repo_type="dataset",
                        filename=repo_file_path,
                        cache_dir=cache_dir,
                    )
                except Exception as e:
                    # If download fails (e.g., for a split that doesn't exist),
                    # raise a controlled error.
                    _raise_load_error(hf_name, split, e)

                with open(local_file_path, encoding="utf-8-sig") as f:
                    json_data = json.load(f)

                if is_openvivqa and isinstance(json_data, dict):
                    images = json_data.get("images", {})
                    annotations = json_data.get("annotations", {})
                    if isinstance(annotations, dict):
                        json_data = list(annotations.values())
                    elif isinstance(annotations, list):
                        json_data = annotations
                    else:
                        raise ValueError(
                            "OpenViVQA annotations must be a dict or list, "
                            f"got {type(annotations).__name__}"
                        )

                    zip_file_path = hf_hub_download(
                        repo_id=hf_name,
                        repo_type="dataset",
                        filename=f"{openvivqa_split}-images.zip",
                        cache_dir=cache_dir,
                    )
                    image_root = os.path.join(
                        cache_dir or os.path.expanduser("~/.cache/huggingface/datasets"),
                        "openvivqa_images",
                        openvivqa_split,
                    )
                    os.makedirs(image_root, exist_ok=True)
                    if not any(os.scandir(image_root)):
                        with zipfile.ZipFile(zip_file_path) as zf:
                            zf.extractall(image_root)

                    image_paths = {}
                    for root, _, filenames in os.walk(image_root):
                        for filename in filenames:
                            image_paths[filename] = os.path.join(root, filename)

                    for row in json_data:
                        image_id = str(row.get("image_id", ""))
                        image_name = images.get(image_id, "")
                        if image_name:
                            row["image_name"] = image_name
                            row["image"] = image_paths.get(
                                image_name, os.path.join(image_root, image_name)
                            )

                return HFDataset.from_list(json_data)
            except Exception as e:
                _raise_load_error(hf_name, split, e)

        def _is_cache_corruption(exc: Exception) -> bool:
            cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
            root  = str(cause or exc)
            _CACHE_CORRUPTION_HINTS = ("ArrowInvalid", "EOF", "Overflow", "Expected object or value")
            return any(hint in root for hint in _CACHE_CORRUPTION_HINTS)

        def _do_load(download_mode=None, streaming=False):
            kwargs = dict(
                split=split,
                cache_dir=cache_dir or None,
            )
            if download_mode is not None:
                kwargs["download_mode"] = download_mode
            if streaming:
                kwargs["streaming"] = True
            return load_dataset(hf_name, **kwargs)

        # Pass 1: normal cached load
        try:
            return _do_load()
        except Exception as e1:
            if not _is_cache_corruption(e1):
                _raise_load_error(hf_name, split, e1)
            print(
                f"[VietnameseDataset] Cache appears corrupted ({type(e1.__cause__ or e1).__name__}: "
                f"{str(e1.__cause__ or e1)[:120]}). "
                f"Retrying with force_redownload ..."
            )

        # Pass 2: force fresh download (clears bad cache entries)
        try:
            return _do_load(download_mode=DownloadMode.FORCE_REDOWNLOAD)
        except Exception as e2:
            if not _is_cache_corruption(e2):
                _raise_load_error(hf_name, split, e2)
            print(
                f"[VietnameseDataset] force_redownload also failed. "
                f"Falling back to streaming mode (slower, no disk cache) ..."
            )

        # Pass 3: streaming → materialise to a regular dataset
        try:
            from datasets import Dataset as HFDataset
            iter_ds = _do_load(streaming=True)
            print("[VietnameseDataset] Streaming mode active — loading all samples into memory ...")
            rows = list(iter_ds)
            ds = HFDataset.from_list(rows)
            print(f"[VietnameseDataset] Materialised {len(ds):,} samples from stream.")
            return ds
        except Exception as e3:
            _raise_load_error(hf_name, split, e3)


    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._ds[idx]
        question   = self._parse_question(sample)
        answers    = self._parse_answers(sample)
        image_obj  = sample.get("image")

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

        # If the HF dataset does not embed image bytes (e.g. ViTextVQA stores
        # only image_name and expects images in a local directory), fall back to
        # loading from self.image_dir.
        if image_obj is None and self.image_dir:
            image_name = self._parse_image_name(sample, idx)
            candidate = os.path.join(self.image_dir, image_name)
            if os.path.isfile(candidate):
                image_obj = candidate
            else:
                print(
                    f"[VietnameseDataset] Image not found at '{candidate}' for idx={idx}. "
                    f"image_dir='{self.image_dir}', image_name='{image_name}'."
                )

        if image_obj is not None:
            try:
                image = self._to_pil(image_obj)
            except Exception as e:
                raise ValueError(
                    f"[VietnameseDataset] Cannot decode image for idx={idx} "
                    f"(type={type(image_obj).__name__}): {e}\n"
                    f"  If images are not embedded in the HF dataset, set "
                    f"'vi_image_dir' in your config to a local images directory."
                ) from e

            try:
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values[0]   # (3, H, W)
            except Exception as e:
                print(f"[VietnameseDataset] CLIP preprocess failed for idx={idx}: {e}")

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
                    print(f"[VietnameseDataset] Pix2Struct preprocess failed for idx={idx}: {e}")

        if pixel_values is None:
            # Raise early with a diagnostic that names the root cause, rather
            # than silently returning a dict without pixel_values and letting
            # the collator raise a less-informative error.
            if image_obj is None:
                hint = (
                    f"The dataset '{self.__class__.__name__}' returned no image "
                    f"for idx={idx} (HF 'image' column is None/missing). "
                )
                if self.image_dir is None:
                    hint += (
                        "Set 'vi_image_dir' in your model config or trainer args "
                        "to a directory containing the images (e.g. TextVQA train images)."
                    )
                else:
                    hint += f"Image not found under image_dir='{self.image_dir}'."
            else:
                hint = f"CLIP preprocessing returned None for idx={idx} (image_obj type={type(image_obj).__name__})."
            raise ValueError(f"[VietnameseDataset] Missing pixel_values: {hint}")

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
            result["_image_name"] = self._parse_image_name(sample, idx)
            result["_image"]      = self._to_pil(image_obj) if image_obj is not None else None

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(image_obj: Any) -> Image.Image:
        """Convert various image representations to PIL.Image.Image."""
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")
        if isinstance(image_obj, bytes):
            return Image.open(io.BytesIO(image_obj)).convert("RGB")
        if isinstance(image_obj, str):
            return Image.open(image_obj).convert("RGB")
        if isinstance(image_obj, dict):
            # datasets library stores images as {"bytes": b"...", "path": "..."}
            if "bytes" in image_obj and image_obj["bytes"]:
                return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            if "path" in image_obj and image_obj["path"]:
                return Image.open(image_obj["path"]).convert("RGB")
        raise ValueError(f"Cannot convert image object of type {type(image_obj)} to PIL")

    # --- Abstract methods for subclasses to implement ---

    def _parse_question(self, sample: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _parse_answers(self, sample: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    def _parse_image_name(self, sample: Dict[str, Any], idx: int) -> str:
        raise NotImplementedError


class KTVICDataset(VietnameseMultimodalDataset):
    """Stage 1: Image Captioning (ai-enthusiasm-community/KTVIC)"""
    def _parse_question(self, sample: Dict[str, Any]) -> str:
        return "Mô tả hình ảnh này một cách chi tiết."

    def _parse_answers(self, sample: Dict[str, Any]) -> List[str]:
        captions = (
            sample.get("captions")
            or sample.get("caption_vi")
            or sample.get("segment_caption_vi")
            or []
        )
        return captions if isinstance(captions, list) else [str(captions)]

    def _parse_image_name(self, sample: Dict[str, Any], idx: int) -> str:
        img_field = sample.get("image")
        if hasattr(img_field, 'filename') and img_field.filename:
            return os.path.basename(img_field.filename)
        return str(sample.get("image_uid", sample.get("id", idx)))


class OpenViVQADataset(VietnameseMultimodalDataset):
    """Stage 2: General VQA (uit-nlp/OpenViVQA-dataset)"""
    def _parse_question(self, sample: Dict[str, Any]) -> str:
        return sample.get("question", "")

    def _parse_answers(self, sample: Dict[str, Any]) -> List[str]:
        answer = sample.get("answer", "")
        return [answer] if isinstance(answer, str) else []

    def _parse_image_name(self, sample: Dict[str, Any], idx: int) -> str:
        img_field = sample.get("image")
        if hasattr(img_field, 'filename') and img_field.filename:
            return os.path.basename(img_field.filename)
        return str(sample.get("image_name", sample.get("question_id", idx)))


class ViTextVQADataset(VietnameseMultimodalDataset):
    """Stage 3: Scene-Text VQA (minhquan6203/ViTextVQA)"""
    def _parse_question(self, sample: Dict[str, Any]) -> str:
        return sample.get("question", "")

    def _parse_answers(self, sample: Dict[str, Any]) -> List[str]:
        return sample.get("answers", [])

    def _parse_image_name(self, sample: Dict[str, Any], idx: int) -> str:
        return str(sample.get("image_name", idx))
