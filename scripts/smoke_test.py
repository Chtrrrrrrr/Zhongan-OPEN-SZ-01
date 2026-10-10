#!/usr/bin/env python3
"""End-to-end smoke test: cold start, warm image path, video path, budget check.

Prints a compact PASS/FAIL table (no base64 payloads) and writes
reports/smoke_test.json. This is the script to run before a live demo.

Usage: python scripts/smoke_test.py [--video demo_assets/smoke_test.mp4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT.parent))
    ap.add_argument("--image", default="data/images/001.png")
    ap.add_argument("--image-real", default="data/images/057.png")
    ap.add_argument("--video", default="demo_assets/smoke_test.mp4")
    ap.add_argument("--clip-probe", default="models/clip_probe_norm.pt")
    ap.add_argument("--npr", default="",
                    help="NPR weights path; EMPTY by default because the NPR branch was "
                         "ablated out (see docs/results.md section 9)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/smoke_test.json")
    args = ap.parse_args()

    ws = Path(args.workspace)
    md = ws / "models"

    cfg = PipelineConfig(
        clip_dir=str(resolve_clip(md / "hf_cache" / "models--openai--clip-vit-large-patch14" / "snapshots")),
        clip_probe=str(md / "clip_probe_norm.pt"),
        forensic_lr=str(md / "forensic_lr.joblib"),
        clip_domain_lr=str(md / "clip_domain_lr.joblib"),
        combined_lr=str(md / "combined_lr.joblib"),
        fusion_json=str(md / "fusion.json"),
        npr_weights=(str(md / "npr.pt") if args.npr else None),
        device=args.device,
    )
    for p in (cfg.clip_dir, cfg.clip_probe, cfg.forensic_lr, cfg.combined_lr, cfg.fusion_json):
        if p and not Path(p).exists():
            print(f"[warn] missing: {p}")

    t0 = time.perf_counter()
    det = AIGCDetector(cfg)
    results = {}

    # ---- cold start: first inference includes model load + cudnn autotune ----
    tc = time.perf_counter()
    rep_cold, _ = det.detect_image(str(ws / args.image))
    cold_ms = (time.perf_counter() - tc) * 1000
    results["cold_start_ms"] = round(cold_ms, 1)

    # ---- warm image path ----
    hot = []
    for p in (args.image, args.image_real):
        t = time.perf_counter()
        rep, _ = det.detect_image(str(ws / p))
        hot.append((p, rep, (time.perf_counter() - t) * 1000))
    results["warm_image_ms"] = [round(h[2], 1) for h in hot]

    # ---- video path ----
    vid_path = ws / args.video
    if vid_path.exists():
        tv = time.perf_counter()
        repv = det.detect_video(str(vid_path))
        vid_ms = (time.perf_counter() - tv) * 1000
        results["video"] = {"ms": round(vid_ms, 1), "verdict": repv.verdict,
                            "confidence": repv.confidence, "duration_s": repv.duration_s}
    else:
        results["video"] = {"skipped": f"{vid_path} not found (run make_test_video.py)"}

    results["budget_ms"] = 5000
    results["image_p95_within_budget"] = bool(max(results["warm_image_ms"]) <= 5000)
    results["video_within_budget"] = bool(results.get("video", {}).get("ms", 9999) <= 5000)

    print("\n===== OPEN-SZ-01 smoke test =====")
    print(f"cold start (load + first infer): {results['cold_start_ms']:.0f} ms")
    print(f"warm image latency:              {results['warm_image_ms']} ms")
    for p, rep, ms in hot:
        print(f"  - {p}: {rep.verdict_cn} (conf={rep.confidence:.3f}) in {ms:.0f} ms")
    v = results.get("video", {})
    print(f"video: {v.get('verdict', v.get('skipped'))} conf={v.get('confidence', '-')} "
          f"in {v.get('ms', '-')} ms")
    print(f"image budget  ({results['budget_ms']}ms): "
          f"{'PASS' if results['image_p95_within_budget'] else 'FAIL'}")
    print(f"video budget  ({results['budget_ms']}ms): "
          f"{'PASS' if results['video_within_budget'] else 'FAIL'}")
    print("note: cold start is reported separately; the service warms up at startup "
          "so live requests see the warm path (manual ch.16.2).")

    out = ws / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
