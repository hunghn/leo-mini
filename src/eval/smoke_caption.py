"""
Smoke test: does the model see the image at all?

Loads a Stage-1/2 (or Stage-3) checkpoint and generates free-form Vietnamese
captions for a handful of images. Look at each image next to its caption:
  - Captions describe actual image content  → visual alignment works; the gap
    is scene-text reading specifically (OCR expert / resolution).
  - Captions are generic/unrelated to the image → visual alignment is broken
    upstream (projector / Stage-1-2 training), before any VQA metric applies.

Usage:
    python -m src.eval.smoke_caption \
        --model_path checkpoints/qwen2_5_3b_vi/stage2/llm_checkpoint \
        --image_dir <dir với ảnh .jpg> --n 5

    # hoặc chỉ định ảnh cụ thể:
    python -m src.eval.smoke_caption \
        --model_path ... --images a.jpg b.jpg c.jpg

    # thêm --stage3_weights <pt> để test checkpoint Stage 3 thay vì Stage 2
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPImageProcessor

from ..data.vitextvqa_dataset import _qwen2_prompt_only
from ..models.leo_mini import LeoMini, IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from ..models.vision_experts import Pix2StructExpert


def main() -> None:
    parser = argparse.ArgumentParser(description="Vi-LEO-MINI captioning smoke test")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--projector_path", default=None,
                        help="Auto-derived as sibling projector_weights.pt if omitted")
    parser.add_argument("--stage3_weights", default=None)
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--n", type=int, default=5, help="Number of images from --image_dir")
    parser.add_argument("--question", default="Mô tả hình ảnh này một cách chi tiết.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--vision_experts", nargs="+", default=["clip", "pix2struct"])
    parser.add_argument("--n_visual", type=int, default=128)
    parser.add_argument("--pix2struct_model_name", default="google/pix2struct-large")
    args = parser.parse_args()

    # Resolve image list
    image_paths = list(args.images or [])
    if args.image_dir:
        found = sorted(
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        image_paths.extend(found[: args.n])
    if not image_paths:
        print("ERROR: no images. Pass --images or --image_dir.", file=sys.stderr)
        sys.exit(1)

    # Resolve projector like vi_evaluator does (stage3_weights also carries it)
    projector_path = args.projector_path
    if projector_path is None:
        candidate = os.path.join(
            os.path.dirname(os.path.normpath(args.model_path)), "projector_weights.pt"
        )
        if os.path.isfile(candidate):
            projector_path = candidate
            print(f"[Smoke] Auto-derived projector_path: {projector_path}")
        elif not args.stage3_weights:
            print("ERROR: no projector weights found — model would be blind by "
                  "construction. Pass --projector_path or --stage3_weights.",
                  file=sys.stderr)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeoMini.from_pretrained(
        llm_path=args.model_path,
        projector_path=projector_path,
        stage3_weights=args.stage3_weights,
        enable_stage3_modules=bool(args.stage3_weights),
        n_visual=args.n_visual,
        vision_experts=args.vision_experts,
        pix2struct_model_name=args.pix2struct_model_name,
    )
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer
    clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    has_p2s = any(isinstance(e, Pix2StructExpert) for e in model.vision.experts)
    p2s_proc = AutoProcessor.from_pretrained(args.pix2struct_model_name) if has_p2s else None

    prompt = _qwen2_prompt_only(args.question)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    for path in image_paths:
        image = Image.open(path).convert("RGB")

        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"][0]
        input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        pixel_values = clip_proc(images=image, return_tensors="pt").pixel_values.to(device)

        pix2struct_inputs = None
        if p2s_proc is not None:
            p2s = p2s_proc(images=image, text="", return_tensors="pt", max_patches=576)
            pix2struct_inputs = {
                "flattened_patches": p2s.flattened_patches.to(device),
                "attention_mask": p2s.attention_mask.to(device),
            }

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pix2struct_inputs=pix2struct_inputs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        caption = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        print(f"\n=== {os.path.basename(path)} ===")
        print(f"  Q: {args.question}")
        print(f"  → {caption}")


if __name__ == "__main__":
    main()
