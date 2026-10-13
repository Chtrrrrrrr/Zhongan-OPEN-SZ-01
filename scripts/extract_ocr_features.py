#!/usr/bin/env python3
"""t6 step 1 - extract OCR text + text-consistency features for document samples.

    python repo/scripts/extract_ocr_features.py --out work/improve/ocr_text.jsonl

Label isolation
---------------
This script never reads a label:
  * the sample list comes from ``data/meta.csv`` and only the ``seq`` and ``content``
    columns are copied out (``src/l1c_ocr.py::iter_document_rows``);
  * ``data/labels.csv`` is never opened;
  * each output record carries ``labels_used: false`` and the column list it used.
The hard evidence is reproduced in the evaluation report: re-running the very same
extraction with a *label-scrambled* copy of meta.csv yields bit-identical features.

Output (JSONL, one record per document sample)
----------------------------------------------
seq, content, image_path, text, mean_score, n_chars, n_boxes, features{...},
boxes[{box,text,score}], backend{...}, elapsed_ms, labels_used
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WS_DEFAULT = REPO.parent
sys.path.insert(0, str(REPO / "src"))

import l1c_ocr as L          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="t6 OCR feature extraction")
    ap.add_argument("--out", default="work/improve/ocr_text.jsonl")
    ap.add_argument("--workspace", default=str(WS_DEFAULT))
    ap.add_argument("--meta", default="data/meta.csv")
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--content", default="document")
    ap.add_argument("--limit", type=int, default=0, help="0 = all document samples")
    args = ap.parse_args(argv)

    ws = Path(args.workspace)
    out = Path(args.out)
    if not out.is_absolute():
        out = ws / out
    out.parent.mkdir(parents=True, exist_ok=True)

    if not L.ocr_available():
        print("[fatal] OCR backend unavailable; see work/improve/ocr_requirements.txt",
              file=sys.stderr)
        return 2
    backend = L.backend_info()
    print(f"[backend] {json.dumps(backend, ensure_ascii=False)}", flush=True)

    rows = L.iter_document_rows(ws / args.meta, args.content,
                               require_images_dir=ws / args.images_dir)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[samples] {len(rows)} '{args.content}' samples from {args.meta} "
          f"(columns read: {list(L.USED_META_COLUMNS)}; data/labels.csv not opened)",
          flush=True)

    t_start = time.perf_counter()
    n_done, n_failed = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            seq = int(row["seq"])
            p = ws / row["image_path"] if not Path(row["image_path"]).is_absolute() \
                else Path(row["image_path"])
            try:
                r = L.extract_features(p)
            except Exception as e:
                n_failed += 1
                rec = {"seq": seq, "content": row["content"],
                       "image_path": str(p.relative_to(ws)).replace("\\", "/"),
                       "text": "", "mean_score": 0.0, "n_chars": 0, "n_boxes": 0,
                       "features": {k: 0.0 for k in L.FEATURE_NAMES},
                       "boxes": [], "backend": backend, "elapsed_ms": None,
                       "labels_used": False, "error": f"{type(e).__name__}: {e}"}
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                print(f"[{i:3d}/{len(rows)}] seq={seq:3d} FAILED {e}", flush=True)
                continue
            boxes = r["ocr"]["items"]
            rec = {
                "seq": seq,
                "content": row["content"],
                "image_path": str(p.relative_to(ws)).replace("\\", "/"),
                "text": "\n".join(b["text"] for b in boxes),
                "mean_score": float(r["features"]["mean_score"]),
                "n_chars": int(r["features"]["n_chars"]),
                "n_boxes": int(r["features"]["n_boxes"]),
                "features": {k: float(v) for k, v in r["features"].items()},
                "boxes": boxes,
                "backend": backend,
                "elapsed_ms": r["elapsed_ms"],
                "labels_used": False,
                "meta_columns_read": list(L.USED_META_COLUMNS),
            }
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            n_done += 1
            if i % 10 == 0 or i == len(rows):
                print(f"[{i:3d}/{len(rows)}] seq={seq:3d} boxes={rec['n_boxes']:3d} "
                      f"chars={rec['n_chars']:4d} mean_score={rec['mean_score']:.4f} "
                      f"{rec['elapsed_ms']:.0f} ms", flush=True)

    print(f"\n[save] {out}  ({out.stat().st_size} bytes)")
    print(f"[done] {n_done} records, {n_failed} failures, "
          f"{time.perf_counter() - t_start:.1f} s")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
