"""
Precompute OCR text for every image in a directory → JSON map for OCR-in-prompt.

Output format (consumed by VietnameseMultimodalDataset via ocr_json=...):
    {"9.jpg": "NHÀ THUỐC | BẢO TÂM AN | 0945166799", "22.jpg": "...", ...}

Reading order: boxes are sorted top-to-bottom then left-to-right, joined
with " | ". Low-confidence detections are dropped.

Images are loaded via PIL with EXIF orientation applied (~1% of ViTextVQA
photos are stored rotated 90°) and small images are upscaled before OCR.

Engines:
    --engine easyocr   pip install easyocr
    --engine paddle    pip install "paddleocr<3" paddlepaddle    (CPU)
                       pip install "paddleocr<3" paddlepaddle-gpu (GPU)

Try both on a sample first and inspect the JSON before the full run:
    python scripts/precompute_ocr.py --image_dir <dir> --limit 20 \
        --engine easyocr --output /tmp/ocr_easy.json
    python scripts/precompute_ocr.py --image_dir <dir> --limit 20 \
        --engine paddle  --output /tmp/ocr_paddle.json

Resume-safe: already-processed images in --output are skipped, progress is
flushed every --save_every images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Upscale images whose longest side is below this before OCR — small photos
# (e.g. 600×336) carry shop-sign text only a few pixels tall.
MIN_LONG_SIDE = 1200


def load_image(path: str):
    """PIL load → EXIF transpose → RGB → optional ×2 upscale → numpy array."""
    import numpy as np
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) < MIN_LONG_SIDE:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    return np.array(img)


def make_reader(engine: str, gpu: bool, langs):
    """Return read(np_img) -> List[(box_4pts, text, conf)]."""
    if engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(langs, gpu=gpu)

        def read(np_img):
            return reader.readtext(np_img, detail=1)

    elif engine == "paddle":
        from paddleocr import PaddleOCR
        reader = PaddleOCR(lang="vi", use_angle_cls=True, show_log=False,
                           use_gpu=gpu)

        def read(np_img):
            result = reader.ocr(np_img, cls=True)
            lines = result[0] if result and result[0] else []
            # Paddle line format: [box_4pts, (text, conf)]
            return [(box, text, conf) for box, (text, conf) in lines]

    else:
        raise ValueError(f"Unknown engine: {engine}")
    return read


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute OCR map for OCR-in-prompt")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", default="data/ocr/vitextvqa_ocr.json")
    parser.add_argument("--engine", choices=["easyocr", "paddle"], default="easyocr")
    parser.add_argument("--langs", nargs="+", default=["vi"],
                        help="easyocr only: language codes (default: vi)")
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N pending images (for a quick trial)")
    args = parser.parse_args()

    image_dir = os.path.expanduser(args.image_dir)
    files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        print(f"ERROR: no images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ocr_map: dict = {}
    if os.path.isfile(args.output):
        with open(args.output, encoding="utf-8") as f:
            ocr_map = json.load(f)
        print(f"[OCR] Resuming: {len(ocr_map):,} images already done in {args.output}")

    pending = [f for f in files if f not in ocr_map]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[OCR] engine={args.engine} | {len(files):,} images total, {len(pending):,} to process")
    if not pending:
        return

    read = make_reader(args.engine, gpu=not args.cpu, langs=args.langs)

    def flush() -> None:
        tmp = args.output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ocr_map, f, ensure_ascii=False)
        os.replace(tmp, args.output)

    n_err = 0
    for i, fname in enumerate(pending, 1):
        try:
            results = read(load_image(os.path.join(image_dir, fname)))
            kept = [
                (box, text) for box, text, conf in results
                if conf >= args.min_confidence and text.strip()
            ]
            # Reading order: top-to-bottom, then left-to-right (bbox = 4 corner
            # points; use the top-left corner, y quantised to 40px rows so
            # near-horizontal neighbours stay on one line).
            kept.sort(key=lambda bt: (int(bt[0][0][1]) // 40, bt[0][0][0]))
            ocr_map[fname] = " | ".join(text.strip() for _, text in kept)
        except Exception as e:
            n_err += 1
            ocr_map[fname] = ""
            if n_err <= 5:
                print(f"[OCR] WARN {fname}: {type(e).__name__}: {e}")

        if i % args.save_every == 0 or i == len(pending):
            flush()
            n_with_text = sum(1 for v in ocr_map.values() if v)
            print(f"[OCR] {i}/{len(pending)} done ({n_with_text:,} with text, {n_err} errors)")

    flush()
    print(f"[OCR] Finished → {args.output} ({len(ocr_map):,} entries)")


if __name__ == "__main__":
    main()
