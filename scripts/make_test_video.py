#!/usr/bin/env python3
"""Build a short MP4 test clip for the video channel demo.

The competition set contains NO videos, so the video path cannot be validated on
label data. This script synthesises a clip from given stills purely to exercise
the video pipeline (uniform frame sampling -> per-frame scoring -> aggregation).

It is a PIPELINE SMOKE TEST, not a benchmark result, and outputs are labelled as
such in reports.

Usage:
  python scripts/make_test_video.py --images data/images/001.png data/images/057.png \
      --out demo_assets/smoke_test.mp4 --seconds 4 --fps 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_resized(path: str, size=(640, 640)) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img.shape[:2]
    s = min(size[0] / w, size[1] / h)
    return cv2.resize(img, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="demo_assets/smoke_test.mp4")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--canvas", type=int, default=640)
    args = ap.parse_args()

    frames_src = [load_resized(p, (args.canvas, args.canvas)) for p in args.images]
    n_frames = max(2, int(args.seconds * args.fps))
    H = W = args.canvas

    def place(img):
        canvas = np.full((H, W, 3), 245, np.uint8)
        h, w = img.shape[:2]
        y, x = (H - h) // 2, (W - w) // 2
        canvas[y:y + h, x:x + w] = img
        return canvas

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter could not open (codec unavailable)")

    per = max(1, n_frames // len(frames_src))
    written = 0
    for img in frames_src:
        base = place(img)
        for i in range(per):
            # slight linear brightness drift so temporal aggregation is non-trivial
            f = base.astype(np.float32) * (1.0 + 0.03 * (i / max(1, per - 1)) - 0.015)
            vw.write(np.clip(f, 0, 255).astype(np.uint8))
            written += 1
    vw.release()
    print(f"[out] {out_path}  frames={written}  fps={args.fps}  ~{written/args.fps:.1f}s")
    print("[note] synthetic pipeline smoke test - NOT a benchmark sample")


if __name__ == "__main__":
    main()
