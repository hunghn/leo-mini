"""
Precompute OCR text for every image in a directory → JSON map for OCR-in-prompt.

Output format (consumed by VietnameseMultimodalDataset via ocr_json=...):
    {"9.jpg": "NHÀ THUỐC | BẢO TÂM AN | 0945166799", "22.jpg": "...", ...}

Reading order: boxes are sorted top-to-bottom then left-to-right, joined
with " | ". Low-confidence detections are dropped.

Usage (on the GPU machine, ~1-2h for 16.7K ViTextVQA images):
    pip install easyocr
    python scripts/precompute_ocr.py \
        --image_dir ~/.cache/huggingface/datasets/vitextvqa_images/st_images \
        --output data/ocr/vitextvqa_ocr.json

Resume-safe: already-processed images in --output are skipped, progress is
flushed every --save_every images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute OCR map for OCR-in-prompt")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", default="data/ocr/vitextvqa_ocr.json")
    parser.add_argument("--langs", nargs="+", default=["vi"],
                        help="EasyOCR language codes (default: vi)")
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N pending images (for a quick trial)")
    args = parser.parse_args()

    try:
        import easyocr
    except ImportError:
        print("ERROR: pip install easyocr", file=sys.stderr)
        sys.exit(1)

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
    print(f"[OCR] {len(files):,} images total, {len(pending):,} to process")
    if not pending:
        return

    reader = easyocr.Reader(args.langs, gpu=not args.cpu)

    def flush() -> None:
        tmp = args.output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ocr_map, f, ensure_ascii=False)
        os.replace(tmp, args.output)

    n_err = 0
    for i, fname in enumerate(pending, 1):
        path = os.path.join(image_dir, fname)
        try:
            # detail=1 → list of (bbox, text, confidence)
            results = reader.readtext(path, detail=1)
            kept = [
                (box, text) for box, text, conf in results
                if conf >= args.min_confidence and text.strip()
            ]
            # Reading order: top-to-bottom, then left-to-right (bbox = 4 corner
            # points; use the top-left corner, y quantised to 20px rows so
            # near-horizontal neighbours stay on one line).
            kept.sort(key=lambda bt: (int(bt[0][0][1]) // 20, bt[0][0][0]))
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
