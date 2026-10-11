#!/usr/bin/env python3
"""Video-channel validation under controlled labels (audit finding F8).

Nothing in the brief is a video, and no labelled corpus with BOTH real and
AI-generated video was reachable offline, so an external video benchmark could not
be built. What CAN be done honestly is a controlled channel validation:

  * take N real stills and N AI stills from the brief (labels known);
  * encode each still into a short clip (same codec/fps/duration for every class,
    so the only difference is the still's provenance);
  * run the production video path (10-frame uniform sampling -> per-frame evidence
    -> mean/P90 aggregation) and report the video-level AUC/accuracy.

This validates that the aggregation actually works and that per-frame evidence
carries through to a video verdict. It is explicitly NOT an external benchmark and
is labelled as such in the output. A genuine external validation needs a public
real+AI video corpus (see the gap note in the output).

Output: reports/video_channel_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import AIGCDetector, PipelineConfig   # noqa: E402


def resolve_clip(p: Path) -> Path:
    if (p / "config.json").is_file():
        return p
    if p.is_dir():
        for d in sorted([x for x in p.iterdir() if x.is_dir()], reverse=True):
            if (d / "config.json").is_file():
                return d
    return p


def make_clip(still: np.ndarray, out_path: Path, seconds: float, fps: int,
              canvas: int = 640) -> None:
    """Encode a still into a clip that FILLS the canvas.

    Deliberately no letterboxing: padding with a flat colour would inject a
    near-white/flat-region artefact whose area depends on the still's aspect ratio,
    which would confound the forensic features and make the comparison meaningless.
    """
    h, w = still.shape[:2]
    s = max(canvas / w, canvas / h)
    big = cv2.resize(still, (max(canvas, int(round(w * s))), max(canvas, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0 = (big.shape[0] - canvas) // 2
    x0 = (big.shape[1] - canvas) // 2
    base = np.ascontiguousarray(big[y0:y0 + canvas, x0:x0 + canvas])
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas, canvas))
    n = max(2, int(seconds * fps))
    for i in range(n):
        f = base.astype(np.float32) * (1.0 + 0.02 * (i / max(1, n - 1)) - 0.01)
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/video_channel_validation.json")
    args = ap.parse_args()
    ws = Path(str(WS))
    md = ws / "models"

    seqs, labels = [], []
    with open(ws / "data" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            labels.append(r["label"])
    ai = [s for s, l in zip(seqs, labels) if l == "ai"][:args.per_class]
    real = [s for s, l in zip(seqs, labels) if l == "real"][:args.per_class]

    cfg = PipelineConfig(
        clip_dir=str(resolve_clip(md / "hf_cache" / "models--openai--clip-vit-large-patch14" / "snapshots")),
        clip_probe=str(md / "clip_probe_norm.pt"),
        forensic_lr=str(md / "forensic_lr.joblib"),
        clip_domain_lr=str(md / "clip_domain_lr.joblib"),
        combined_lr=str(md / "combined_lr.joblib"),
        fusion_json=str(md / "fusion.json"),
        device=args.device,
    )
    det = AIGCDetector(cfg)

    tmp = Path(tempfile.mkdtemp(prefix="opensz01_video_"))
    rows = []
    try:
        for tag, group in (("ai", ai), ("real", real)):
            for seq in group:
                still = cv2.imread(str(ws / "data" / "images" / f"{seq:03d}.png"), cv2.IMREAD_COLOR)
                p = tmp / f"{tag}_{seq:03d}.mp4"
                make_clip(still, p, args.seconds, args.fps)
                t0 = time.perf_counter()
                rep = det.detect_video(str(p))
                ms = (time.perf_counter() - t0) * 1000
                rows.append({"seq": seq, "label": tag, "fused_score": rep.fused_score,
                             "verdict": rep.verdict, "frames": rep.versions.get("frames"),
                             "latency_ms": round(ms, 1)})
                print(f"[{tag}] seq={seq:3d} score={rep.fused_score:.4f} "
                      f"verdict={rep.verdict} {ms:.0f} ms", flush=True)
    finally:
        for f in tmp.glob("*.mp4"):
            f.unlink(missing_ok=True)
        tmp.rmdir()

    from sklearn.metrics import roc_auc_score
    y = np.array([1 if r["label"] == "ai" else 0 for r in rows])
    sc = np.array([r["fused_score"] for r in rows])
    lat = np.array([r["latency_ms"] for r in rows])
    auc = float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else None
    # threshold at the 2-false-positive budget over the real clips
    real_sc = np.sort(sc[y == 0])[::-1]
    thr = float(real_sc[min(2, len(real_sc) - 1)]) if len(real_sc) else 0.5
    pred = (sc >= thr).astype(int)
    out = {
        "report": "video_channel_validation",
        "caveat": ("CONTROLLED validation, not an external benchmark. The brief contains no "
                   "videos and no labelled real+AI video corpus was reachable offline, so both "
                   "classes were synthesised from the labelled STILLS (identical codec, fps, "
                   "duration and canvas for both classes; only the still's provenance differs)."),
        "gap": ("a genuine external video validation still requires a public corpus with both "
               "real and AI-generated video (the AI-only collections found on the mirror have "
               "no matching real class); this is recorded as an open item"),
        "protocol": {"per_class": args.per_class, "seconds": args.seconds, "fps": args.fps,
                     "video_path": "10-frame uniform sampling -> per-frame forensic+global-CLIP "
                                   "evidence -> mean/P90 aggregation (production path)"},
        "video_level": {
            "n": len(rows), "auc": (round(auc, 4) if auc is not None else None),
            "threshold_from_2fp_budget": round(thr, 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "tpr": round(float(pred[y == 1].mean()), 4),
            "fpr": round(float(pred[y == 0].mean()), 4),
        },
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                       "p95": round(float(np.percentile(lat, 95)), 1),
                       "budget_ms": 5000,
                       "p95_within_budget": bool(np.percentile(lat, 95) <= 5000)},
        "per_clip": rows,
    }
    p = ws / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[video] AUC={out['video_level']['auc']} acc={out['video_level']['accuracy']} "
          f"tpr={out['video_level']['tpr']} fpr={out['video_level']['fpr']} "
          f"P95={out['latency_ms']['p95']}ms")
    print(f"[save] {p}")


if __name__ == "__main__":
    main()
