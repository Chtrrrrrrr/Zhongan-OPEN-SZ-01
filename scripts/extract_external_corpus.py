#!/usr/bin/env python3
"""Extract the decision heads' feature blocks for an external labelled corpus.

The competition data alone (119 samples, 63 "real" of which nearly all are phone
photos of paper) cannot support a decision rule that generalises to real claim SCENE
photos - the manual's own audit (ch.2.3) and risk R2 say so, and the measurement in
docs/known-issue-scene-photo-fpr.md shows the consequence (100% of real claim photos
flagged). This script turns an external labelled corpus into the exact same feature
blocks the heads consume, so it can be added to the TRAINING folds of the nested-CV
protocol (scripts/train_with_external_reals.py).

The label of each folder is declared by `--labels` ("dir=0|1", 0 = real, 1 = fake) so a
folder's role is explicit rather than inferred from its name.

Outputs under --out:
  l1.npy       [N, 47]     L1 forensic features
  l1b.npy      [N, 20]     L1b domain features
  clip.npy     [N, 1024]   CLIP global
  patch.npy    [N, 16]     patch-token statistics (when the spec asks for them)
  tiles.npy    object[N]   per-scale tile features, aggregated by the spec's rule
  meta.json    paths, labels, folders, per-image geometry

Usage
  python scripts/extract_external_corpus.py --out models/external_corpus \
      --labels ../testset/dsh_testset/real=0,../testset/dsh_testset/real_synthbuster=0,\
../testset/dsh_testset/fake_ai=1,../testset/dsh_testset/fake_dataset=1,\
../testset/dsh_testset/fake_tamper=1
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from l0_input import ParsedInput, _png_chunk_list, perceptual_hash   # noqa: E402
from l1_forensics import extract_l1_features, feature_vector        # noqa: E402
from l1b_domain import extract_l1b_features, feature_vector_b       # noqa: E402
from l2_clip import CLIPFeatureExtractor                            # noqa: E402
from l3_localization import patch_token_features                    # noqa: E402
from feature_spec import FeatureSpec                                # noqa: E402
from tile_clip_sweep import resolve_clip                            # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def decode_capped(path: Path, max_decode_side: int | None):
    """Decode an image, optionally capping the decoded long side.

    Peak memory and tile cost scale with decoded pixels and this corpus contains
    4,928 px originals; the deployed L1b pass already bounds itself to a 1,600 px long
    side, so a 2,400 px decode cap leaves every feature's own preprocessing bound intact.
    """
    import cv2
    from PIL import Image
    try:
        data = path.read_bytes()
    except OSError as e:
        return None, f"read failed: {e}"
    try:
        pil = Image.open(io.BytesIO(data))
        exif = bool(pil.getexif())
        rgb = np.asarray(pil.convert("RGB"))
    except Exception as e:
        return None, f"decode failed: {e}"
    h, w = rgb.shape[:2]
    if max_decode_side and max(h, w) > max_decode_side:
        s = max_decode_side / max(h, w)
        rgb = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    bgr_full = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    del rgb
    long_side = max(bgr_full.shape[:2])
    scale = min(1.0, 512.0 / long_side)
    bgr = (cv2.resize(bgr_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
           if scale < 1.0 else bgr_full.copy())
    parsed = ParsedInput(ok=True, media_type="image", raw_bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest(),
                         phash=perceptual_hash(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                         orig_width=bgr_full.shape[1], orig_height=bgr_full.shape[0],
                         png_chunks=_png_chunk_list(data), exif_present=exif,
                         bgr=bgr, bgr_full=bgr_full, resize_scale=scale)
    return parsed, None


def parse_label_arg(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        d, lab = part.rsplit("=", 1)
        out.append((Path(d.strip()), int(lab)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="dir=0|1,... (0=real, 1=fake)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-spec", default="models/feature_spec.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-decode-side", type=int, default=2400)
    ap.add_argument("--max-per-folder", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip folders already present in meta.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec.load(ROOT / args.feature_spec)
    print(f"[spec] {json.dumps(spec.to_dict(), ensure_ascii=False)}")

    groups = parse_label_arg(args.labels)
    items = []
    for folder, label in groups:
        files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in EXTS]
        if args.max_per_folder:
            files = files[:args.max_per_folder]
        print(f"[corpus] {label} {folder}: {len(files)} files")
        items += [(p, label, folder.name) for p in files]

    done_folders = set()
    if args.resume and (out / "meta.json").exists():
        prev = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        done_folders = {f for f in prev.get("folders", [])}
        print(f"[resume] already extracted folders: {sorted(done_folders)}")
        items = [it for it in items if it[2] not in done_folders]
        print(f"[resume] {len(items)} files remaining")

    ext = CLIPFeatureExtractor(resolve_clip(""), device=args.device, input_size=224)
    blocks = {"l1": [], "l1b": [], "clip": [], "patch": [], "tiles": []}
    labels, paths, folders = [], [], []
    skipped = 0
    t0 = time.perf_counter()
    for i, (p, label, folder) in enumerate(items):
        parsed, err = decode_capped(p, args.max_decode_side)
        if parsed is None:
            print(f"  [skip] {p.name}: {err}", flush=True)
            skipped += 1
            continue
        bgr_full = parsed.bgr_full
        detail = spec.tile_detail(ext, bgr_full)
        blocks["l1"].append(feature_vector(extract_l1_features(parsed)))
        blocks["l1b"].append(feature_vector_b(extract_l1b_features(parsed)))
        blocks["clip"].append(ext(parsed.bgr).astype(np.float32))
        if spec.patch_tokens:
            blocks["patch"].append(
                patch_token_features(ext, bgr_full, max_side=spec.patch_max_side))
        blocks["tiles"].append(spec.aggregate_detail(detail))
        labels.append(label)
        paths.append(str(p))
        folders.append(folder)
        del parsed, detail
        if (i + 1) % 25 == 0:
            gc.collect()
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(items)}  {el:.0f}s  ({el/(i+1)*1000:.0f} ms/image)", flush=True)

    if not labels:
        print("[corpus] nothing extracted")
        return 0

    prev = {}
    if args.resume and (out / "meta.json").exists():
        prev = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        for k in blocks:
            f = out / f"{k}.npy"
            if f.exists():
                old = np.load(f, allow_pickle=(k == "tiles"))
                blocks[k] = list(old) + blocks[k]
        labels = list(prev.get("labels", [])) + labels
        paths = list(prev.get("paths", [])) + paths
        folders = list(prev.get("folders", [])) + folders

    for k, v in blocks.items():
        if not v:
            continue
        arr = np.array(v, dtype=object) if k == "tiles" else np.stack(v)
        np.save(out / f"{k}.npy", arr, allow_pickle=(k == "tiles"))
    (out / "meta.json").write_text(json.dumps(
        {"n": len(labels), "labels": labels, "paths": paths, "folders": folders,
         "skipped": skipped, "feature_spec": spec.to_dict(),
         "max_decode_side": args.max_decode_side,
         "extract_s": round(time.perf_counter() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    n_real = sum(1 for l in labels if l == 0)
    print(f"[corpus] {len(labels)} images (real={n_real} fake={len(labels)-n_real}, "
          f"{skipped} skipped) in {time.perf_counter()-t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
