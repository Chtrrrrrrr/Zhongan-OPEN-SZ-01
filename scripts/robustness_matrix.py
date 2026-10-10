#!/usr/bin/env python3
"""Robustness matrix (manual ch.14.1).

Re-encodes / resizes / degrades every sample, then re-scores it with the SAME
frozen pipeline and thresholds, and reports how accuracy, FPR and TPR move.
The point is not that nothing degrades, but to learn *which* cue fails first
(the manual's stated purpose).

Perturbations: JPEG q95/85/70/50, long-side downscale to 1024 / 512 then back,
screen-capture simulation (downscale + JPEG), light Gaussian noise, motion blur,
and a 2x re-encode chain.

Usage: python scripts/robustness_matrix.py [--limit 119]
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

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import AIGCDetector, PipelineConfig          # noqa: E402
from fusion import ModelScore                              # noqa: E402
from l0_input import parse_bytes                           # noqa: E402
from l1_forensics import extract_l1_features, feature_vector   # noqa: E402
from l1b_domain import extract_l1b_features, feature_vector_b  # noqa: E402
from l2_clip_tiles import extract_tiled_features           # noqa: E402


def enc_jpeg(bgr, q):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return buf.tobytes() if ok else None


def enc_png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else None


def perturb(name: str, bgr: np.ndarray):
    """Return (perturbed_bgr, encoded_bytes) for a named perturbation."""
    h, w = bgr.shape[:2]
    if name == "none":
        return bgr, enc_png(bgr)
    if name.startswith("jpeg_q"):
        q = int(name.split("q")[1])
        data = enc_jpeg(bgr, q)
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
    if name == "noise_sigma3":
        rng = np.random.RandomState(0)
        out = np.clip(bgr.astype(np.float32) + rng.normal(0, 3, bgr.shape), 0, 255).astype(np.uint8)
        return out, enc_png(out)
    if name == "motion_blur":
        k = np.zeros((7, 7), np.float32)
        k[3, :] = 1.0 / 7
        out = cv2.filter2D(bgr, -1, k)
        return out, enc_png(out)
    if name == "double_jpeg":
        d1 = enc_jpeg(bgr, 90)
        b1 = cv2.imdecode(np.frombuffer(d1, np.uint8), cv2.IMREAD_COLOR)
        d2 = enc_jpeg(b1, 75)
        return cv2.imdecode(np.frombuffer(d2, np.uint8), cv2.IMREAD_COLOR), d2
    raise ValueError(name)


PERTURBATIONS = ["none", "jpeg_q95", "jpeg_q85", "jpeg_q70", "jpeg_q50",
                 "downscale_1024", "downscale_512", "screenshot",
                 "noise_sigma3", "motion_blur", "double_jpeg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(WS))
    ap.add_argument("--perturbations", nargs="+", default=PERTURBATIONS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
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
        paths, y = paths[:args.limit], y[:args.limit]
    print(f"[data] {len(paths)} samples, {len(args.perturbations)} perturbations")

    clip_snap = md / "hf_cache" / "models--openai--clip-vit-large-patch14" / "snapshots"
    snaps = sorted([d for d in clip_snap.iterdir() if d.is_dir()]) if clip_snap.is_dir() else []
    cfg = PipelineConfig(
        clip_dir=str(snaps[-1]) if snaps else None,
        clip_probe=str(md / "clip_probe_norm.pt"),
        forensic_lr=str(md / "forensic_lr.joblib"),
        combined_lr=str(md / "combined_lr.joblib"),
        fusion_json=str(md / "fusion.json"),
        device=args.device,
    )
    det = AIGCDetector(cfg)
    thr = det.fusion.thr_high
    print(f"[frozen] threshold thr_high = {thr:.4f}, members = {list(det.fusion.weights)}")

    orig = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]

    def score(bgr_full: np.ndarray, raw: bytes) -> float:
        parsed = parse_bytes(raw, keep_full=True)
        if not parsed.ok:
            return 0.0
        f1 = feature_vector(extract_l1_features(parsed))
        fb = feature_vector_b(extract_l1b_features(parsed))
        cf = det.clip(parsed.bgr)
        tf = extract_tiled_features(det.clip, parsed.bgr_full)
        s = []
        if det.forensic_pack is not None:
            s.append(ModelScore("forensic", det._pack_logit(det.forensic_pack, np.concatenate([f1, fb]))))
        if det.combined_pack is not None:
            s.append(ModelScore("combined",
                                det._pack_logit(det.combined_pack, np.concatenate([f1, fb, cf, tf]))))
        return det.fusion.fuse(s).fused_score

    rows = []
    for pname in args.perturbations:
        preds = np.zeros(len(paths), dtype=int)
        for i, bgr in enumerate(orig):
            try:
                pbgr, raw = perturb(pname, bgr)
                sc = score(pbgr, raw)
                preds[i] = 1 if sc >= thr else 0
            except Exception as e:
                print(f"[warn] {pname} sample {i}: {e}")
        tp = int(((y == 1) & (preds == 1)).sum())
        tn = int(((y == 0) & (preds == 0)).sum())
        fp = int(((y == 0) & (preds == 1)).sum())
        fn = int(((y == 1) & (preds == 0)).sum())
        row = {"perturbation": pname,
               "accuracy": round((tp + tn) / len(y), 4),
               "fpr": round(fp / max(1, int((y == 0).sum())), 4),
               "tpr": round(tp / max(1, int((y == 1).sum())), 4),
               "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}
        rows.append(row)
        print(f"[{pname:14s}] acc={row['accuracy']:.4f} fpr={row['fpr']:.4f} "
              f"tpr={row['tpr']:.4f} (fp={fp}, fn={fn})", flush=True)

    base = rows[0] if rows and rows[0]["perturbation"] == "none" else None
    for r in rows:
        if base:
            r["delta_accuracy_vs_clean"] = round(r["accuracy"] - base["accuracy"], 4)
            r["delta_tpr_vs_clean"] = round(r["tpr"] - base["tpr"], 4)

    out = ws / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "note": "same frozen pipeline and threshold; only the input is perturbed",
        "threshold_high": round(float(thr), 4),
        "n_samples": len(paths),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
