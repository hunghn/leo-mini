"""
lmms-eval adapter for LEO-MINI.

Registers the "leomini" model name with the lmms-eval framework so that
the evaluator can run:

    lmms-eval --model leomini \\
              --model_args pretrained=/path/to/llm,stage3_weights=/path/to/ckpt \\
              --tasks mme,pope,...

This module must be imported before lmms-eval is invoked so that the
@register_model decorator fires.  scripts/run_eval_leomini.py handles this
automatically for subprocess-based evaluation.

model_args keys (all optional except pretrained):
    pretrained       — path to LLM checkpoint passed to LlamaForCausalLM
    stage3_weights   — path to stage3_adapter_weights.pt (projector+CoTR+MMoE)
    n_visual         — number of visual tokens N^V (default 64)
    load_in_4bit     — "True"/"False", enables 4-bit quantisation
    device           — "cuda" / "cpu" (default "cuda")
    batch_size       — inference batch size (default 1)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# Ensure repo root is importable when called from scripts/run_eval_leomini.py
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_here))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
    from lmms_eval.api.instance import Instance
    _LMMS_EVAL_AVAILABLE = True
except ImportError:
    _LMMS_EVAL_AVAILABLE = False
    # Provide stubs so the rest of the module can be imported for type-checking
    class lmms:  # type: ignore[no-redef]
        pass
    def register_model(name):  # type: ignore[no-redef]
        return lambda cls: cls
    Instance = object  # type: ignore[assignment]

from transformers import AutoProcessor, CLIPImageProcessor

from ..models.leo_mini import LeoMini, IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


if _LMMS_EVAL_AVAILABLE:
    @register_model("leomini")
    class LeoMiniAdapter(lmms):
        """
        lmms-eval model adapter for LEO-MINI.

        Implements generate_until (VQA / captioning) and loglikelihood (POPE,
        multiple-choice benchmarks).
        """

        def __init__(
            self,
            pretrained: str,
            stage3_weights: Optional[str] = None,
            n_visual: int = 64,
            load_in_4bit: bool = False,
            device: str = "cuda",
            batch_size: int = 1,
            **kwargs,
        ) -> None:
            super().__init__()

            load_in_4bit = _to_bool(load_in_4bit)
            self._device     = device
            self._batch_size = int(batch_size)

            self._model = LeoMini.from_pretrained(
                llm_path=pretrained,
                stage3_weights=stage3_weights if stage3_weights else None,
                enable_stage3_modules=bool(stage3_weights),
                n_visual=int(n_visual),
                load_in_4bit=load_in_4bit,
            )
            self._model.eval()
            if not load_in_4bit:
                self._model = self._model.to(device)

            self._tokenizer = self._model.tokenizer
            self._clip_proc  = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
            self._p2s_proc = AutoProcessor.from_pretrained(
                "google/pix2struct-large"
            )

        # ------------------------------------------------------------------
        # lmms-eval required properties
        # ------------------------------------------------------------------

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def batch_size(self) -> int:
            return self._batch_size

        @property
        def device(self) -> str:
            return self._device

        # ------------------------------------------------------------------
        # Input preparation
        # ------------------------------------------------------------------

        def _prepare(
            self,
            text: str,
            image: Optional[Image.Image],
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
            """
            Tokenise text and encode image for a single sample.
            Returns (input_ids, attention_mask, pixel_values, pix2struct_inputs).
            """
            # Inject image placeholder if an image is present and not already in text
            if image is not None and IMAGE_TOKEN not in text:
                text = f"{IMAGE_TOKEN}\n{text}"

            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            input_ids      = enc.input_ids.clone()
            attention_mask = enc.attention_mask

            # Replace <image> special token id with IMAGE_TOKEN_INDEX sentinel
            img_tok_id = self._tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            if img_tok_id is not None:
                input_ids[input_ids == img_tok_id] = IMAGE_TOKEN_INDEX

            pixel_values      = None
            pix2struct_inputs = None
            if image is not None:
                pixel_values = self._clip_proc(
                    images=image, return_tensors="pt"
                ).pixel_values
                p2s = self._p2s_proc(
                    images=image,
                    text="",
                    return_tensors="pt",
                    max_patches=576,
                )
                pix2struct_inputs = {
                    "flattened_patches": p2s.flattened_patches,
                    "attention_mask":    p2s.attention_mask,
                }

            return input_ids, attention_mask, pixel_values, pix2struct_inputs

        @staticmethod
        def _extract_image(req: Instance) -> Optional[Image.Image]:
            """Pull the first PIL image out of lmms-eval request metadata."""
            meta = getattr(req, "metadata", None) or {}
            img  = meta.get("image") or getattr(req, "doc", {}).get("image")
            if isinstance(img, list):
                img = img[0] if img else None
            if img is not None and not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            return img

        def _to_device(self, t):
            if t is None:
                return None
            if isinstance(t, dict):
                return {k: v.to(self._device) for k, v in t.items()}
            return t.to(self._device)

        # ------------------------------------------------------------------
        # generate_until — used by VQA, captioning, open-ended benchmarks
        # ------------------------------------------------------------------

        def generate_until(self, requests: List[Instance]) -> List[str]:
            results = []
            for req in requests:
                context, gen_kwargs = req.args[0], req.args[1]
                image = self._extract_image(req)

                input_ids, attn_mask, pv, p2s = self._prepare(context, image)
                input_ids = self._to_device(input_ids)
                attn_mask = self._to_device(attn_mask)
                pv        = self._to_device(pv)
                p2s       = self._to_device(p2s)

                max_new = int(gen_kwargs.get("max_new_tokens", 256))
                until   = gen_kwargs.get("until", [self._tokenizer.eos_token])

                out_ids = self._model.generate(
                    input_ids=input_ids,
                    pixel_values=pv,
                    pix2struct_inputs=p2s,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
                # LeoMini.generate() calls self.llm.generate(inputs_embeds=...)
                # internally. HuggingFace returns ONLY the newly generated token IDs
                # when inputs_embeds is used — there is no input prefix to strip.
                # Slicing by input_ids.shape[1] would discard the answer entirely.
                text = self._tokenizer.decode(out_ids[0], skip_special_tokens=True)
                for stop in until:
                    if stop and stop in text:
                        text = text[: text.index(stop)]
                results.append(text.strip())
            return results

        # ------------------------------------------------------------------
        # loglikelihood — used by POPE, multiple-choice benchmarks
        # ------------------------------------------------------------------

        def loglikelihood(
            self, requests: List[Instance]
        ) -> List[Tuple[float, bool]]:
            results = []
            for req in requests:
                context, continuation = req.args[0], req.args[1]
                image = self._extract_image(req)

                full_text = context + continuation
                input_ids, attn_mask, pv, p2s = self._prepare(full_text, image)

                # Build labels: -100 for context, token ids for continuation
                ctx_len   = len(self._tokenizer.encode(context, add_special_tokens=False))
                labels    = input_ids.clone()
                # +1 accounts for BOS if present
                mask_end  = min(ctx_len + 1, labels.shape[1])
                labels[0, :mask_end] = -100

                input_ids = self._to_device(input_ids)
                attn_mask = self._to_device(attn_mask)
                labels    = self._to_device(labels)
                pv        = self._to_device(pv)
                p2s       = self._to_device(p2s)

                with torch.inference_mode():
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        pixel_values=pv,
                        pix2struct_inputs=p2s,
                        labels=labels,
                    )

                # outputs.loss is mean NLL over continuation tokens
                n_cont       = (labels[0] != -100).sum().item()
                log_lkl      = -outputs.loss.item() * max(n_cont, 1)
                results.append((log_lkl, True))
            return results

        def loglikelihood_rolling(
            self, requests: List[Instance]
        ) -> List[float]:
            raise NotImplementedError(
                "loglikelihood_rolling is not required for standard benchmarks."
            )
