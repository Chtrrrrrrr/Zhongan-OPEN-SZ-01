#!/usr/bin/env python3
"""External holdout validation on the 1,537-image真假测试素材集 (robustness check).

`data/` holds the 119 competition samples the heads are fit on. The testset under
`--testset` is a DIFFERENT, larger collection (1,040 real / 497 fake, 13 generators,
all converted to PNG) that the heads never see. Because the heads are trained on the
competition samples, this set is not a clean holdout for *model selection* - it is a
robustness and false-positive-frequency check: how often does the deployed decision
rule, with the thresholds frozen from the competition calibration, flag a real
photograph, and how well does it rank AI-generated material it has never seen.

Cost note: (re)extracting features for all 1,537 images is the expensive part, so the
per-image features are cached under --cache and reused across decision configurations.

Usage
  python scripts/external_holdout.py --testset ..\\testset\\dsh_testset --stage extract
  python scripts/external_holdout.py --testset ..\\testset\\dsh_testset --stage evaluate
"""
from __future__ import annotations

import argparse
import csv
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

from l0_input import (ParsedInput, _png_chunk_list, parse_image,  # noqa: E402
                      perceptual_hash)
import models_path as mp                                          # noqa: E402
from l1_forensics import extract_l1_features, feature_vector      # noqa: E402
from l1b_domain import extract_l1b_features, feature_vector_b     # noqa: E402
from l2_clip import CLIPFeatureExtractor                          # noqa: E402
from tile_clip_sweep import AGGS, CONFIGS, tiles_for, resolve_clip  # noqa: E402
from extract_clip_patch_features import patch_stats               # noqa: E402

CELL = "C7"          # 5x5 @ 0.30 overlap, max_side 1600
AGG = "meanmax"      # mean (+) max over tiles


def iter_testset(testset: Path, per_folder_limit: int = 0):
    """(relative path, label) for every image in the testset folder layout.

    ``per_folder_limit`` takes a deterministic prefix of each folder (files are sorted),
    which keeps a bounded run reproducible instead of sampling at random.
    """
    groups = {"real": 0, "real_synthbuster": 0, "fake_ai": 1, "fake_dataset": 1,
              "fake_tamper": 1}
    for folder, label in groups.items():
        d = testset / folder
        if not d.is_dir():
            continue
        n = 0
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                yield p, label, folder
                n += 1
                if per_folder_limit and n >= per_folder_limit:
                    break


def decode_capped(path: Path, max_decode_side: int | None):
    """Decode an image, optionally capping the decoded long side.

    Peak RSS and tile cost scale with the decoded pixel count, and this testset has
    4,928 px originals whose full-resolution decode dwarfs everything else; the L1/L1b
    noise lines are measured on a long side of 1,600 in the deployed pipeline anyway
    (see l1b_domain), so a 2,400 px cap leaves every feature's own preprocessing bound
    intact while keeping memory flat over 1,500 images.
    """
    import cv2
    from PIL import Image
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return None, f"read failed: {e}"
    try:
        pil = Image.open(io.BytesIO(data))
        pil = pil.convert("RGB")
        rgb = np.asarray(pil)
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
    if scale < 1.0:
        bgr = cv2.resize(bgr_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        bgr = bgr_full.copy()
    from l0_input import perceptual_hash
    parsed = ParsedInput(ok=True, media_type="image", raw_bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest(),
                         phash=perceptual_hash(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
                         orig_width=bgr_full.shape[1], orig_height=bgr_full.shape[0],
                         png_chunks=_png_chunk_list(data),
                         exif_present=bool(pil_exif(data)),
                         bgr=bgr, bgr_full=bgr_full, resize_scale=scale)
    return parsed, None


def pil_exif(data: bytes):
    from PIL import Image
    try:
        return Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return {}


def stage_extract(args):
    import cv2
    import gc
    testset = Path(args.testset)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    for p, label, folder in iter_testset(testset, args.per_folder_limit):
        items.append((p, label, folder))
    print(f"[testset] {len(items)} images "
          f"(real={sum(1 for _, l, _ in items if l == 0)}, "
          f"fake={sum(1 for _, l, _ in items if l == 1)})", flush=True)

    feats = {"l1": [], "l1b": [], "clip": [], "patch": [], "tiles": []}
    labels, rels, folders = [], [], []
    ext = CLIPFeatureExtractor(resolve_clip(""), device=args.device, input_size=224)
    grid, overlap, max_side = CONFIGS[CELL][0]
    t0 = time.perf_counter()
    skipped = 0
    for i, (p, label, folder) in enumerate(items):
        parsed, err = decode_capped(p, args.max_decode_side)
        if parsed is None:
            print(f"  [skip] {p.name}: {err}", flush=True)
            skipped += 1
            continue
        f1 = extract_l1_features(parsed)
        fb = extract_l1b_features(parsed)
        cf = ext(parsed.bgr)
        pt = patch_stats(ext, parsed.bgr_full) if args.patch_max_side else None
        ts = tiles_for(parsed.bgr_full, grid, overlap, max_side)
        tf = ext.batch(ts, batch_size=len(ts)).astype(np.float32)

        feats["l1"].append(feature_vector(f1))
        feats["l1b"].append(feature_vector_b(fb))
        feats["clip"].append(cf.astype(np.float32))
        if pt is not None:
            feats["patch"].append(pt)
        feats["tiles"].append(tf)
        labels.append(label)
        rels.append(str(p.relative_to(testset)).replace("\\", "/"))
        folders.append(folder)
        del parsed, f1, fb, cf, ts, tf
        if (i + 1) % 25 == 0:
            gc.collect()
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(items)}  {el:.0f}s  ({el/(i+1)*1000:.0f} ms/image)", flush=True)

    for k, v in feats.items():
        if not v:
            continue
        arr = np.array(v, dtype=object) if k == "tiles" else np.stack(v)
        np.save(cache / f"{k}.npy", arr, allow_pickle=(k == "tiles"))
    (cache / "meta.json").write_text(json.dumps(
        {"n": len(labels), "labels": labels, "rel": rels, "folder": folders,
         "skipped": skipped, "cell": CELL, "aggregation": AGG,
         "max_decode_side": args.max_decode_side,
         "scales": [{"grid": g, "overlap": o, "max_side": m} for (g, o, m) in CONFIGS[CELL]],
         "extract_s": round(time.perf_counter() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[extract] {len(labels)} images ({skipped} skipped) in "
          f"{time.perf_counter()-t0:.0f}s -> {cache}")
    return 0


def stage_evaluate(args):
    """Score the testset with the heads+threshold frozen from the competition data."""
    import joblib
    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    y = np.array(meta["labels"])
    tiles = np.load(cache / "tiles.npy", allow_pickle=True)
    agg = np.stack([AGGS[AGG](np.asarray(t, dtype=np.float32)) for t in tiles]).astype(np.float64)
    Xf = np.concatenate([np.load(cache / "l1.npy"), np.load(cache / "l1b.npy"),
                         np.load(cache / "patch.npy")], axis=1).astype(np.float64)
    Xc = np.concatenate([Xf, np.load(cache / "clip.npy"), agg], axis=1)
    print(f"[features] forensic={Xf.shape} combined={Xc.shape} n={len(y)}")

    fp = ROOT / args.forensic_head
    cp = ROOT / args.combined_head
    ej = ROOT / args.fusion_json
    if not (fp.exists() and cp.exists() and ej.exists()):
        raise SystemExit(f"frozen heads missing: {fp} / {cp} / {ej}")
    fpack, cpack = joblib.load(fp), joblib.load(cp)
    fusion = json.loads(ej.read_text(encoding="utf-8"))
    w = {k: v for k, v in fusion["weights"].items() if v > 0}
    temps = fusion.get("temperatures", {})
    thr_high = fusion["thr_high"]

    def logit(pack, X):
        z = pack["clf"].decision_function(pack["scaler"].transform(X))
        return z

    zf, zc = logit(fpack, Xf), logit(cpack, Xc)
    names = [n for n in w]
    wn = np.array([w[n] for n in names], dtype=np.float64)
    wn = wn / wn.sum()

    def fuse(zmap):
        ps = np.stack([1.0 / (1.0 + np.exp(-zmap[n] / temps.get(n, 1.0))) for n in names], axis=1)
        ps = np.clip(ps, 1e-6, 1 - 1e-6)
        L = np.log(ps / (1 - ps))
        return (L * wn).sum(axis=1)

    fused = fuse({"forensic": zf, "combined": zc})
    pred = (fused >= thr_high).astype(int)

    def block(sel):
        yy, pp = y[sel], pred[sel]
        tp = int(((yy == 1) & (pp == 1)).sum())
        tn = int(((yy == 0) & (pp == 0)).sum())
        fp_ = int(((yy == 0) & (pp == 1)).sum())
        fn = int(((yy == 1) & (pp == 0)).sum())
        from sklearn.metrics import roc_auc_score
        return {"n": int(len(yy)), "real": int((yy == 0).sum()), "fake": int((yy == 1).sum()),
                "accuracy": round((tp + tn) / max(1, len(yy)), 4),
                "fpr": round(fp_ / max(1, int((yy == 0).sum())), 4),
                "tpr": round(tp / max(1, int((yy == 1).sum())), 4),
                "confusion": {"tp": tp, "tn": tn, "fp": fp_, "fn": fn},
                "auc": round(float(roc_auc_score(yy, fused[sel])), 4)
                if len(set(yy.tolist())) > 1 else None}

    allsel = np.arange(len(y), dtype=int)
    overall = block(allsel)
    by_folder = {}
    for folder in sorted(set(meta["folder"])):
        sel = np.array([i for i, f in enumerate(meta["folder"]) if f == folder])
        by_folder[folder] = block(sel)

    # generator-level breakdown (from the file name prefix for fake_ai)
    by_gen = {}
    for i, rel in enumerate(meta["rel"]):
        if not rel.startswith("fake_ai/"):
            continue
        parts = Path(rel).name.split("_")
        gen = parts[1] if len(parts) > 1 else "unknown"
        by_gen.setdefault(gen, []).append(i)
    gen_rows = {}
    for gen, idxs in sorted(by_gen.items()):
        sel = np.array(idxs)
        gen_rows[gen] = {"n": int(len(sel)),
                         "tpr": round(float(pred[sel].mean()), 4),
                         "auc": None}

    payload = {
        "report": "external_holdout",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "testset": args.testset,
        "decision": {"members": names, "weights": w, "temperatures": temps,
                     "thr_high": thr_high,
                     "note": "heads and threshold FROZEN from the competition calibration; "
                             "nothing is re-fit or re-tuned on this set"},
        "overall": overall,
        "by_folder": by_folder,
        "fake_ai_by_generator": gen_rows,
        "errors": {
            "false_positives": [meta["rel"][i] for i in allsel
                                if y[i] == 0 and pred[i] == 1][:60],
            "false_negatives": [meta["rel"][i] for i in allsel
                                if y[i] == 1 and pred[i] == 0][:60],
            "n_false_positives": int(((y == 0) & (pred == 1)).sum()),
            "n_false_negatives": int(((y == 1) & (pred == 0)).sum()),
        },
        "protocol_note": ("The heads are trained on the 119 competition samples, so this set "
                          "is a robustness / FPR-frequency check, not a clean model-selection "
                          "holdout. Because it contains only 2 real claim photographs, the "
                          "real/real_synthbuster rows should be read as 'camera photos in "
                          "general', not as claim documents."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "by_folder": by_folder,
                      "fake_ai_by_generator": gen_rows}, ensure_ascii=False, indent=1))
    print(f"[save] {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--cache", default="models/external_cache")
    ap.add_argument("--stage", choices=("extract", "evaluate"), default="evaluate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-decode-side", type=int, default=2400,
                    help="cap the decoded long side to keep peak memory flat over ~1.5k images")
    ap.add_argument("--patch-max-side", type=int, default=1600,
                    help="long side used for the patch-token pass (0 disables it)")
    ap.add_argument("--per-folder-limit", type=int, default=0,
                    help="deterministic prefix of each testset folder (0 = all)")
    ap.add_argument("--models-dir", default=None,
                    help="directory holding the deployment artifacts; default resolution "
                         "prefers repo/models (the frozen copy) and falls back per file to "
                         "<workspace>/models - see scripts/models_path.py")
    ap.add_argument("--forensic-head", default=None)
    ap.add_argument("--combined-head", default=None)
    ap.add_argument("--fusion-json", default=None)
    ap.add_argument("--out", default="reports/external_holdout.json")
    args = ap.parse_args()

    # ---- one place decides which artifact copy is loaded ---------------------
    md, md_source = mp.resolve_models_dir(args.models_dir, workspace=ROOT.parent)
    print(mp.describe(md, md_source))
    print(mp.assert_expected_fusion(md))
    for attr, name in (("forensic_head", "forensic_lr.joblib"),
                       ("combined_head", "combined_lr.joblib"),
                       ("fusion_json", "fusion.json")):
        if getattr(args, attr):
            continue
        p, src = mp.resolve_artifact(md, name, workspace=ROOT.parent)
        if src != "frozen":
            print(f"[models] {name}: {p}  (source={src})")
        setattr(args, attr, str(p))

    return stage_extract(args) if args.stage == "extract" else stage_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
