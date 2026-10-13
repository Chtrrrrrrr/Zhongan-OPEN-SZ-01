#!/usr/bin/env python3
"""End-to-end smoke test: cold start, warm image path, video path, budget check.

Prints a compact PASS/FAIL table (no base64 payloads) and writes
reports/smoke_test.json. This is the script to run before a live demo.

The warm image row is measured over EVERY labelled sample, not over one or two
hand-picked files: a 2-sample "P95" is just a max, and the acceptance criterion is a
P95 over a batch. Per-stage medians are reported alongside so a regression can be
attributed to a layer instead of to "the pipeline".

Usage: python scripts/smoke_test.py [--video demo_assets/smoke_test.mp4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from detector import AIGCDetector, PipelineConfig   # noqa: E402
from models_path import (assert_expected_fusion, describe,  # noqa: E402
                         resolve_artifact, resolve_models_dir)


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
    ap.add_argument("--workspace", default=None,
                    help="workspace root holding data/ (default: auto-detect from this "
                         "script's location, then its parent)")
    ap.add_argument("--models-dir", default=None,
                    help="directory holding the frozen deployment artifacts. Default "
                         "resolution prefers <repo>/models (the frozen copy); see "
                         "scripts/models_path.py")
    ap.add_argument("--data-dir", default="data",
                    help="directory holding labels.csv + images/ (the labelled set)")
    ap.add_argument("--image", default="data/images/001.png")
    ap.add_argument("--image-real", default="data/images/057.png")
    ap.add_argument("--video", default="demo_assets/smoke_test.mp4")
    ap.add_argument("--clip-probe", default=None,
                    help="trained CLIP probe; default resolves against the models dir")
    ap.add_argument("--npr", default="",
                    help="NPR weights path; EMPTY by default because the NPR branch was "
                         "ablated out (see docs/results.md section 9)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: <repo>/reports/smoke_test.json)")
    ap.add_argument("--budget-ms", type=int, default=5000)
    ap.add_argument("--target-ms", type=int, default=4500,
                    help="engineering target the iteration steers to (manual 表 16-2 "
                         "stage allocations sum to 4500 ms); P95 is compared against both")
    args = ap.parse_args()

    # ---- locate the labelled data -------------------------------------------
    # Auto-detect instead of assuming the caller's cwd is the workspace root.
    if args.workspace:
        ws = Path(args.workspace)
    else:
        # A candidate works only if it holds BOTH the labels and the images: `repo/`
        # ships data/labels.csv but not the PNGs (they are git-ignored), so a
        # labels-only test would pick the repo and then fail on the first image.
        cands = [ROOT, ROOT.parent]

        def usable(c: Path) -> bool:
            return ((c / args.data_dir / "labels.csv").is_file()
                    and (c / args.data_dir / "images").is_dir())

        ws = next((c for c in cands if usable(c)), None)
        if ws is None:
            ws = next((c for c in cands
                       if (c / args.data_dir / "labels.csv").is_file()), ROOT.parent)
    ws = ws.resolve()

    # ---- locate the FROZEN deployment artifacts -----------------------------
    # Prefer <repo>/models; the workspace copy is stale (different thresholds and
    # different joblib bytes) and would make these numbers disagree with the docs.
    md, md_source = resolve_models_dir(args.models_dir, workspace=ws)
    print(describe(md, md_source))
    print(assert_expected_fusion(md))

    def artifact(name: str, explicit: str | None = None) -> Path:
        """Frozen copy first; fall back to <workspace>/models only if not shipped.

        The frozen repository copy does not carry the large working artefacts
        (clip_probe_norm.pt, clip_domain_lr.joblib, hf_cache/), so a single
        directory choice is not enough - every fallback is printed.
        """
        if explicit:
            return Path(explicit)
        p, src = resolve_artifact(md, name, workspace=ws)
        if src != "frozen":
            print(f"[models] {name}: {p}  (source={src})")
        return p

    clip_probe = artifact("clip_probe_norm.pt", args.clip_probe)
    forensic_lr = artifact("forensic_lr.joblib")
    clip_domain_lr = artifact("clip_domain_lr.joblib")
    combined_lr = artifact("combined_lr.joblib")
    feature_spec = artifact("feature_spec.json")
    fusion_json = artifact("fusion.json")
    hf_cache = artifact("hf_cache/models--openai--clip-vit-large-patch14/snapshots")

    cfg = PipelineConfig(
        clip_dir=str(resolve_clip(hf_cache)),
        clip_probe=str(clip_probe),
        forensic_lr=str(forensic_lr),
        clip_domain_lr=str(clip_domain_lr),
        combined_lr=str(combined_lr),
        feature_spec=str(feature_spec),
        fusion_json=str(fusion_json),
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

    # ---- warm image path: every labelled sample ----
    hot = []
    for p in (args.image, args.image_real):
        t = time.perf_counter()
        rep, _ = det.detect_image(str(ws / p))
        hot.append((p, rep, (time.perf_counter() - t) * 1000))
    results["warm_image_ms"] = [round(h[2], 1) for h in hot]

    per_sample = []
    labels_path = ws / args.data_dir / "labels.csv"
    if labels_path.exists():
        seqs, y = [], []
        with open(labels_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seqs.append(int(r["seq"]))
                y.append(1 if r["label"] == "ai" else 0)
        stage_acc = {}
        pred, truth = [], []
        for s, lab in zip(seqs, y):
            p = ws / args.data_dir / "images" / f"{s:03d}.png"
            t = time.perf_counter()
            rep, _ = det.detect_image(str(p))
            ms = (time.perf_counter() - t) * 1000
            per_sample.append({"seq": s, "label": lab, "verdict": rep.verdict, "ms": round(ms, 1)})
            pred.append(1 if rep.verdict == "suspect" else 0)
            truth.append(lab)
            for k, v in (rep.latency_ms or {}).items():
                stage_acc.setdefault(k, []).append(v)
        pred, truth = np.array(pred), np.array(truth)
        lat = np.array([r["ms"] for r in per_sample])
        tp = int(((pred == 1) & (truth == 1)).sum())
        tn = int(((pred == 0) & (truth == 0)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        results["warm_latency_all_samples"] = {
            "n": len(per_sample),
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "max_ms": round(float(lat.max()), 1),
            "mean_ms": round(float(lat.mean()), 1),
            "budget_ms": args.budget_ms,
            "target_ms": args.target_ms,
            "p95_within_budget": bool(np.percentile(lat, 95) <= args.budget_ms),
            "p95_within_target": bool(np.percentile(lat, 95) <= args.target_ms),
            "stage_p50_ms": {k: round(float(np.percentile(v, 50)), 1)
                             for k, v in stage_acc.items()},
            "stage_p95_ms": {k: round(float(np.percentile(v, 95)), 1)
                             for k, v in stage_acc.items()},
        }
        # decision metrics measured with the FROZEN deployment thresholds: this is the
        # "what does the shipped service output today" row, not an out-of-fold estimate
        results["frozen_threshold_decision"] = {
            "accuracy": round((tp + tn) / len(truth), 4),
            "fpr": round(fp / max(1, int((truth == 0).sum())), 4),
            "tpr": round(tp / max(1, int((truth == 1).sum())), 4),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "note": "in-sample for these heads (they are fit on all labelled samples); "
                    "use reports/deployment_metrics.json for the out-of-fold numbers",
        }
        results["per_sample"] = per_sample

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

    results["budget_ms"] = args.budget_ms
    results["target_ms"] = args.target_ms
    warm = results.get("warm_latency_all_samples")
    results["image_p95_within_budget"] = bool(
        warm["p95_within_budget"] if warm else max(results["warm_image_ms"]) <= args.budget_ms)
    results["image_p95_within_target"] = bool(
        warm["p95_within_target"] if warm else max(results["warm_image_ms"]) <= args.target_ms)
    results["video_within_budget"] = bool(results.get("video", {}).get("ms", 9999) <= args.budget_ms)

    print("\n===== OPEN-SZ-01 smoke test =====")
    print(f"cold start (load + first infer): {results['cold_start_ms']:.0f} ms")
    print(f"warm image latency (spot):       {results['warm_image_ms']} ms")
    for p, rep, ms in hot:
        print(f"  - {p}: {rep.verdict_cn} (conf={rep.confidence:.3f}) in {ms:.0f} ms")
    if warm:
        print(f"warm latency, all {warm['n']} labelled samples: "
              f"P50={warm['p50_ms']:.0f} P95={warm['p95_ms']:.0f} max={warm['max_ms']:.0f} ms")
        for k in sorted(warm["stage_p50_ms"]):
            print(f"    stage {k:16s} P50={warm['stage_p50_ms'][k]:8.1f} "
                  f"P95={warm['stage_p95_ms'][k]:8.1f}")
        d = results["frozen_threshold_decision"]
        print(f"frozen-threshold decision: acc={d['accuracy']} fpr={d['fpr']} tpr={d['tpr']} "
              f"{d['confusion']}")
    v = results.get("video", {})
    print(f"video: {v.get('verdict', v.get('skipped'))} conf={v.get('confidence', '-')} "
          f"in {v.get('ms', '-')} ms")
    print(f"image budget  ({results['budget_ms']}ms): "
          f"{'PASS' if results['image_p95_within_budget'] else 'FAIL'}"
          f"   target ({results['target_ms']}ms): "
          f"{'PASS' if results['image_p95_within_target'] else 'FAIL'}")
    print(f"video budget  ({results['budget_ms']}ms): "
          f"{'PASS' if results['video_within_budget'] else 'FAIL'}")
    print("note: cold start is reported separately; the service warms up at startup "
          "so live requests see the warm path (manual ch.16.2).")

    out = Path(args.out) if args.out else (ROOT / "reports" / "smoke_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
