#!/usr/bin/env python3
"""Cross-generator generalisation test (manual ch.10.2 / 10.3 D2, 12.1).

Why this script exists: every shard of AIGC-Detection-Benchmark is a stratified
sample of the SAME 18 sources, so "hold out a shard" is a same-generator holdout,
not a cross-generator test. The honest protocol is to REMOVE whole generators from
training and score them afterwards.

Protocol
  - extract frozen CLIP features once for all shards (with the same common-codec
    pre-processing used by training: JPEG q90 for every sample), cache to disk;
  - for each fold: hold out a set of generators, train the linear probe on the
    remaining generators only, then report AUC per HELD-OUT generator
    (plus a same-generator control trained/tested with a normal split);
  - the real class is shared; only generated sources are rotated out.

Output: reports/cross_generator.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

GEN = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "CycleGAN", 4: "DALLE2", 5: "GauGAN",
       6: "GLIDE", 7: "Midjourney", 8: "ProGAN", 9: "SD14", 10: "SD15", 11: "SDXL",
       12: "StarGAN", 13: "StyleGAN", 14: "StyleGAN2", 15: "VQDM",
       16: "WhichFaceIsReal", 17: "Wukong"}

# generators held out entirely (chosen to span families: diffusion + GAN + Chinese)
FOLDS = {
    "diffusion_holdout": ["Midjourney", "SDXL"],
    "gan_holdout": ["BigGAN", "StyleGAN2"],
    "chinese_and_glide": ["Wukong", "GLIDE"],
}


def resolve_clip(name: str) -> str:
    p = Path(name)
    if p.is_dir() and (p / "config.json").is_file():
        return str(p)
    for root in (WS / "models" / "hf_cache", ROOT / "models" / "hf_cache"):
        snaps = root / ("models--" + name.replace("/", "--")) / "snapshots"
        if snaps.is_dir():
            for d in sorted(snaps.iterdir(), reverse=True):
                if (d / "config.json").is_file():
                    return str(d)
    return name


def extract_or_load(cache: Path, shards, reencode_q: int = 90):
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        print(f"[cache] {cache} {z['feats'].shape}")
        return z["feats"], z["labels"], z["gens"]
    from l2_clip import CLIPFeatureExtractor

    ext = CLIPFeatureExtractor(resolve_clip("openai/clip-vit-large-patch14"),
                               device="cuda", input_size=224)
    feats, labels, gens = [], [], []
    for sp in shards:
        df = pd.read_parquet(sp, columns=["image", "label", "generator"])
        raw_col = df["image"].to_numpy()
        lab_col = df["label"].to_numpy()
        gen_col = df["generator"].to_numpy()
        buf_i, buf_l, buf_g = [], [], []

        def flush():
            if not buf_i:
                return
            bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in buf_i]
            feats.append(ext.batch(bgr, batch_size=len(bgr)))
            labels.extend(buf_l)
            gens.extend(buf_g)
            buf_i.clear()
            buf_l.clear()
            buf_g.clear()

        for i in tqdm(range(len(df)), desc=f"extract {Path(sp).name}", mininterval=10):
            try:
                img = np.array(Image.open(io.BytesIO(raw_col[i]["bytes"])).convert("RGB"))
            except Exception:
                continue
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, reencode_q])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            buf_i.append(img)
            buf_l.append(int(lab_col[i]))
            buf_g.append(int(gen_col[i]))
            if len(buf_i) >= 24:
                flush()
        flush()
    F = np.concatenate(feats, 0)
    y = np.array(labels, dtype=np.int64)
    g = np.array(gens, dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=F, labels=y, gens=g)
    print(f"[cache] wrote {cache} {F.shape}")
    return F, y, g


def fit_eval(Ftr, ytr, Fte, yte):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Ftr), ytr)
    p = clf.predict_proba(sc.transform(Fte))[:, 1]
    return clf, sc, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--cache", default="models/clip_bench_feats.npz")
    ap.add_argument("--out", default="reports/cross_generator.json")
    ap.add_argument("--max-per-shard", type=int, default=0,
                    help="0 = all; otherwise cap each shard (host RAM guard)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    shards = args.shards or [str(p) for p in sorted((WS / "data" / "external").glob("*.parquet"))]
    shards = shards[:3]  # shards 0-2 are the training set used for the deployed probe
    if args.max_per_shard:
        shards = shards
    print(f"[data] {len(shards)} shards: {[Path(s).name for s in shards]}")

    F, y, g = extract_or_load(WS / args.cache, shards)
    if args.max_per_shard:
        keep = np.arange(0, len(y))
        rng = np.random.RandomState(0)
        sel = []
        for gen in np.unique(g):
            idx = keep[g == gen]
            if len(idx) > args.max_per_shard:
                idx = rng.choice(idx, args.max_per_shard, replace=False)
            sel.append(idx)
        sel = np.sort(np.concatenate(sel))
        F, y, g = F[sel], y[sel], g[sel]
    print(f"[data] {len(y)} samples, generators: "
          f"{ {GEN[k]: int((g == k).sum()) for k in np.unique(g)} }")

    results = {"protocol": {
        "task": "cross-generator generalisation of the frozen-CLIP linear probe",
        "why": ("all benchmark shards contain the SAME 18 sources, so a shard holdout is "
                "NOT a cross-generator test; whole generators are removed from training here"),
        "features": "frozen CLIP-L/14 pooled, common-codec preprocessing (JPEG q90) for every sample",
        "classifier": "L2 logistic regression on standardised features",
    }, "dataset": {
        "n_samples": int(len(y)),
        "generator_counts": {GEN[k]: int((g == k).sum()) for k in np.unique(g)},
    }, "folds": {}}

    # same-generator control: stratified 5-fold over all data
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(F, y):
        _, _, p = fit_eval(F[tr], y[tr], F[te], y[te])
        oof[te] = p
    results["same_generator_control"] = {
        "auc": round(float(roc_auc_score(y, oof)), 4),
        "note": "5-fold OOF over all data; every generator appears in training",
    }

    gen_names = {k: GEN[k] for k in np.unique(g) if k != 0}
    for fold_name, held in FOLDS.items():
        held_ids = [k for k, v in gen_names.items() if v in held]
        if not held_ids:
            continue
        # Test set = the held-out generators PLUS the shared Real class, otherwise the
        # held-out set would contain a single class and AUC/FPR would be undefined.
        mask_gen_hold = np.isin(g, held_ids)
        mask_real = g == 0
        mask_hold = mask_gen_hold | mask_real
        mask_train = ~mask_gen_hold  # Real stays available for training
        _, _, p_hold = fit_eval(F[mask_train], y[mask_train], F[mask_hold], y[mask_hold])
        real_hold = y[mask_hold] == 0
        # operating threshold from the REAL samples of the held-out evaluation set,
        # with the same 2-false-positive budget used for the 119-sample evaluation
        thr = float(np.sort(p_hold[real_hold])[::-1][min(2, int(real_hold.sum()) - 1)]) \
            if real_hold.sum() else 0.5
        fold = {
            "held_out_generators": held,
            "n_train": int(mask_train.sum()),
            "n_holdout": int(mask_hold.sum()),
            "n_real_in_holdout": int(real_hold.sum()),
            "overall_auc_on_holdout": round(float(roc_auc_score(y[mask_hold], p_hold)), 4),
            "threshold_from_real_2fp_budget": round(thr, 4),
            # per held-out generator the meaningful number is recall at that threshold;
            # AUC is undefined inside a single-class generator subset
            "per_generator_recall_at_threshold": {},
        }
        for k in held_ids:
            m = mask_gen_hold[mask_hold] & (g[mask_hold] == k)
            if m.sum() < 8:
                fold["per_generator_recall_at_threshold"][GEN[k]] = {
                    "n": int(m.sum()), "recall": None, "note": "too few samples"}
                continue
            fold["per_generator_recall_at_threshold"][GEN[k]] = {
                "n": int(m.sum()),
                "recall": round(float((p_hold[m] >= thr).mean()), 4),
            }
        pred = (p_hold >= thr).astype(int)
        fold["holdout_at_2fp_real_budget"] = {
            "fpr": round(float(pred[real_hold].mean()), 4),
            "tpr": round(float(pred[y[mask_hold] == 1].mean()), 4),
            "accuracy": round(float((pred == y[mask_hold]).mean()), 4),
        }
        results["folds"][fold_name] = fold
        print(f"[{fold_name}] held={held} auc={fold['overall_auc_on_holdout']} "
              f"recall={ {k: v['recall'] for k, v in fold['per_generator_recall_at_threshold'].items()} } "
              f"thr={thr:.3f}")

    out = WS / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
