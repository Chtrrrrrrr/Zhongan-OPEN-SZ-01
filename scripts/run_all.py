#!/usr/bin/env python3
"""One-command reproduction of the OPEN-SZ-01 demo results.

Runs the whole pipeline with the exact ordering used for the reported numbers:

  step 1  L1 + L1b forensic features        -> features.csv, models/l1b_feats.npy
  step 2  CLIP features (global + tiles)    -> models/clip_feats*.npy, clip_tiles*.npy
  step 3  CLIP probe training (external)    -> models/clip_probe.pt (+_norm)
  step 4  full evaluation on 119 samples    -> reports/metrics.json + reports/*.html
  step 5  ablations (features / fusion)     -> reports/*_ablation.json
  step 6  error analysis + figures          -> reports/error_analysis.csv, assets/*.png
  step 7  smoke test + video channel        -> reports/smoke_test.json, video report

Usage:
  python scripts/run_all.py                      # everything
  python scripts/run_all.py --steps 1,2,4        # only some steps
  python scripts/run_all.py --skip-clip-train    # reuse an existing probe
  python scripts/run_all.py --with-robustness    # add the (slow) robustness matrix

Everything is written under the workspace so the package is self-contained.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo/
WS = ROOT.parent                                    # workspace root


def run(cmd: list[str], desc: str) -> None:
    print(f"\n===== {desc} =====\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(WS))
    print(f"----- {desc}: exit={r.returncode} ({time.perf_counter() - t0:.1f}s) -----",
          flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1,2,3,4,5,6,7")
    ap.add_argument("--skip-clip-train", action="store_true")
    ap.add_argument("--with-robustness", action="store_true",
                    help="also run the robustness matrix (about 30-40 min on GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",")}
    py = sys.executable

    if "1" in steps:
        run([py, "repo/scripts/extract_l1b_features.py", "--data-dir", "data",
             "--features", "features.csv", "--out-dir", "models"],
            "step 1/7  L1 (47) + L1b (20) forensic features")

    if "2" in steps:
        run([py, "repo/scripts/extract_clip_features.py", "--data-dir", "data",
             "--out-dir", "models", "--device", args.device],
            "step 2/7  CLIP global + tiled features for the 119 samples")

    if "3" in steps and not args.skip_clip_train:
        shards = sorted((WS / "data" / "external").glob("*.parquet"))
        if not shards:
            raise SystemExit("no parquet shards in data/external - see docs/reproduce.md")
        run([py, "repo/scripts/train_clip_probe.py",
             "--shards", *[str(s) for s in shards],
             "--out", "models/clip_probe_norm.pt", "--device", args.device,
             "--epochs", str(args.epochs), "--reencode-q", "90"],
            "step 3/7  CLIP probe on the 18-generator benchmark (common-codec)")

    if "4" in steps:
        run([py, "repo/scripts/evaluate_system.py", "--device", args.device,
             "--clip-probe", "models/clip_probe_norm.pt"],
            "step 4/7  nested-CV evaluation + latency + per-sample reports")

    if "5" in steps:
        run([py, "repo/scripts/experiment_features.py"], "step 5a/7  feature-set ablation")
        run([py, "repo/scripts/experiment_fusion.py"], "step 5b/7  fusion-rule ablation")

    if "6" in steps:
        run([py, "repo/scripts/diagnose_errors.py"], "step 6a/7  error analysis")
        run([py, "repo/scripts/make_figures.py"], "step 6b/7  figures")

    if "7" in steps:
        (WS / "demo_assets").mkdir(exist_ok=True)
        run([py, "repo/scripts/make_test_video.py",
             "--images", "data/images/001.png", "data/images/057.png",
             "--out", "demo_assets/smoke_test.mp4"], "step 7a/7  video smoke clip")
        run([py, "repo/scripts/smoke_test.py", "--device", args.device],
            "step 7b/7  cold/warm image + video budget check")

    if args.with_robustness:
        run([py, "repo/scripts/robustness_matrix.py", "--device", args.device],
            "extra  robustness matrix (manual ch.14.1)")

    print("\nALL REQUESTED STEPS COMPLETED")
    print("Key artifacts: features.csv, models/*.pt|joblib|json, reports/metrics.json, "
          "reports/*.html, repo/assets/*.png, demo_assets/smoke_test.mp4")


if __name__ == "__main__":
    main()
