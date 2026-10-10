#!/usr/bin/env python3
"""Robustness matrix under the OUT-OF-FOLD protocol (manual ch.14.1).

Why OOF: the deployed heads are fit on all 119 samples, so scoring the clean set
with them is in-sample and saturates (everything far from the threshold), which
hides the perturbation effect. Here each sample is scored by a head that never saw
it (5-fold, identical split/seed to scripts/evaluate_system.py), and the FROZEN
calibration + thresholds from models/fusion.json are applied unchanged.

The point is not that nothing degrades, but *which* cue degrades first.

Perturbations: JPEG q95/85/70/50, long-side downscale to 1024 / 512 then back,
screen-capture simulation, double JPEG, light noise, motion blur.

Usage: python scripts/robustness_matrix.py [--limit N] [--perturbations ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import AIGCDetector, PipelineConfig               # noqa: E402
from fusion import FusionEngine, ModelScore                     # noqa: E402
from l0_input import parse_bytes                                # noqa: E402
from l1_forensics import extract_l1_features, feature_vector     # noqa: E402
from l1b_domain import extract_l1b_features, feature_vector_b    # noqa: E402
from l2_clip_tiles import extract_tiled_features                 # noqa: E402


def enc_jpeg(bgr, q):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return buf.tobytes() if ok else None


def enc_png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else None


def perturb(name: str, bgr: np.ndarray, orig_bytes: bytes | None = None):
    h, w = bgr.shape[:2]
    if name == "none":
        return bgr, (orig_bytes if orig_bytes is not None else enc_png(bgr))
    if name.startswith("jpeg_q"):
        data = enc_jpeg(bgr, int(name.split("q")[1]))
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), data
    if name.startswith("downscale_"):
        side = int(name.split("_")[1])
        s = side / max(h, w)
        small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return back, enc_png(back)
    if name == "screenshot":
        s = 0.6
        small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        data = enc_jpeg(small, 80)
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), data
    if name == "double_jpeg":
        b1 = cv2.imdecode(np.frombuffer(enc_jpeg(bgr, 90), np.uint8), cv2.IMREAD_COLOR)
        d2 = enc_jpeg(b1, 75)
        return cv2.imdecode(np.frombuffer(d2, np.uint8), cv2.IMREAD_COLOR), d2
    if name == "noise_sigma3":
        rng = np.random.RandomState(0)
        out = np.clip(bgr.astype(np.float32) + rng.normal(0, 3, bgr.shape), 0, 255).astype(np.uint8)
        return out, enc_png(out)
    if name == "motion_blur":
        k = np.zeros((7, 7), np.float32)
        k[3, :] = 1.0 / 7
        out = cv2.filter2D(bgr, -1, k)
        return out, enc_png(out)
    raise ValueError(name)


PERTURBATIONS = ["none", "jpeg_q95", "jpeg_q85", "jpeg_q70", "jpeg_q50",
                 "downscale_1024", "downscale_512", "screenshot", "double_jpeg",
                 "noise_sigma3", "motion_blur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(WS))
    ap.add_argument("--perturbations", nargs="+", default=PERTURBATIONS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", default="reports/robustness_matrix.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ws = Path(args.workspace)
    md = ws / "models"

    seqs, y = [], []
    with open(ws / "data" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)
    paths = [ws / "data" / "images" / f"{s:03d}.png" for s in seqs]
    if args.limit:
        paths, y, seqs = paths[:args.limit], y[:args.limit], seqs[:args.limit]
    n = len(paths)
    print(f"[data] {n} samples, {len(args.perturbations)} perturbations", flush=True)

    # frozen calibration + thresholds (fitted on clean nested-CV scores)
    eng = FusionEngine.load(str(md / "fusion.json"))
    thr = eng.thr_high
    members = list(eng.weights)
    print(f"[frozen] members={members} thr_high={thr:.4f} temps="
          f"{ {k: round(v.temperature, 3) for k, v in eng.scalers.items()} }", flush=True)

    # clean features to fit the fold heads on
    X1 = np.loadtxt(ws / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(md / "l1b_feats_119.npy")
    Fg = np.load(md / "clip_feats_119.npy")
    Ft = np.load(md / "clip_tiles_119.npy")
    if args.limit:
        X1, Xb, Fg, Ft = X1[:n], Xb[:n], Fg[:n], Ft[:n]
    XF = np.concatenate([X1, Xb], 1)
    XC = np.concatenate([X1, Xb, Fg, Ft], 1)

    clip_snap = md / "hf_cache" / "models--openai--clip-vit-large-patch14" / "snapshots"
    snaps = sorted([d for d in clip_snap.iterdir() if d.is_dir()]) if clip_snap.is_dir() else []
    det = AIGCDetector(PipelineConfig(clip_dir=str(snaps[-1]) if snaps else None,
                                      device=args.device))

    def extract(bgr, raw):
        parsed = parse_bytes(raw, keep_full=True)
        if not parsed.ok:
            raise RuntimeError(parsed.error)
        f1 = feature_vector(extract_l1_features(parsed))
        fb = feature_vector_b(extract_l1b_features(parsed))
        cf = det.clip(parsed.bgr)
        tf = extract_tiled_features(det.clip, parsed.bgr_full)
        return np.concatenate([f1, fb]), np.concatenate([f1, fb, cf, tf])

    def logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # fold heads fitted ONLY on clean training-fold features
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    folds = []
    for tr, va in skf.split(XF, y):
        scf = StandardScaler().fit(XF[tr])
        clff = LogisticRegression(C=1.0, max_iter=5000).fit(scf.transform(XF[tr]), y[tr])
        scc = StandardScaler().fit(XC[tr])
        clfc = LogisticRegression(C=1.0, max_iter=5000).fit(scc.transform(XC[tr]), y[tr])
        folds.append((va, (scf, clff), (scc, clfc)))
    print("[folds] heads fitted on clean training folds", flush=True)

    originals = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    raw_originals = [p.read_bytes() for p in paths]

    rows = []
    for pname in args.perturbations:
        # 1) extract features for the perturbed inputs
        FX = np.zeros((n, XF.shape[1]))
        FC = np.zeros((n, XC.shape[1]))
        for i, bgr in enumerate(originals):
            pbgr, raw = perturb(pname, bgr, raw_originals[i])
            FX[i], FC[i] = extract(pbgr, raw)
        # 2) score each sample with the head that never saw it
        preds = np.zeros(n, dtype=int)
        for va, (scf, clff), (scc, clfc) in folds:
            lf = logit(clff.predict_proba(scf.transform(FX[va]))[:, 1])
            lc = logit(clfc.predict_proba(scc.transform(FC[va]))[:, 1])
            for k, i in enumerate(va):
                dec = eng.fuse([ModelScore("forensic", float(lf[k])),
                                ModelScore("combined", float(lc[k]))])
                preds[i] = 1 if dec.fused_score >= thr else 0
        tp = int(((y == 1) & (preds == 1)).sum())
        tn = int(((y == 0) & (preds == 0)).sum())
        fp = int(((y == 0) & (preds == 1)).sum())
        fn = int(((y == 1) & (preds == 0)).sum())
        row = {"perturbation": pname,
               "accuracy": round((tp + tn) / n, 4),
               "fpr": round(fp / max(1, int((y == 0).sum())), 4),
               "tpr": round(tp / max(1, int((y == 1).sum())), 4),
               "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}
        rows.append(row)
        print(f"[{pname:14s}] acc={row['accuracy']:.4f} fpr={row['fpr']:.4f} "
              f"tpr={row['tpr']:.4f} (fp={fp}, fn={fn})", flush=True)

    base = next((r for r in rows if r["perturbation"] == "none"), None)
    for r in rows:
        if base:
            r["delta_accuracy_vs_clean"] = round(r["accuracy"] - base["accuracy"], 4)
            r["delta_tpr_vs_clean"] = round(r["tpr"] - base["tpr"], 4)

    out = ws / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": ("out-of-fold: each sample scored by a head fitted on the clean "
                     "training fold; frozen calibration and thresholds from fusion.json"),
        "threshold_high": round(float(thr), 4),
        "members": members,
        "n_samples": n,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
