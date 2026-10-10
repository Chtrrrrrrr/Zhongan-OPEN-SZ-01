#!/usr/bin/env python3
"""One-command reproduction of the OPEN-SZ-01 demo results.

Runs the whole pipeline with the exact ordering used for the reported numbers:

  step 1  L1 forensic features            -> features.csv
  step 2  CLIP probe training (external)  -> models/clip_probe.pt
  step 3  full evaluation on 119 samples  -> reports/metrics.json + reports/*.html
  step 4  video-channel smoke test        -> demo_assets/smoke_test.mp4 + report

Usage:
  python scripts/run_all.py                 # full run
  python scripts/run_all.py --skip-clip-train   # reuse an existing probe
  python scripts/run_all.py --steps 3,4         # only some steps

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
    ap.add_argument("--steps", default="1,2,3,4")
    ap.add_argument("--skip-clip-train", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",")}
    py = sys.executable

    if "1" in steps:
        run([py, "repo/scripts/extract_features.py", "--data-dir", "data",
             "--out", "features.csv"], "step 1/4  L1 forensic features")

    if "2" in steps and not args.skip_clip_train:
        shards = sorted((WS / "data" / "external").glob("*.parquet"))
        if not shards:
            raise SystemExit("no parquet shards in data/external - see docs/reproduce.md")
        run([py, "repo/scripts/train_clip_probe.py",
             "--shards", *[str(s) for s in shards],
             "--out", "models/clip_probe.pt", "--device", args.device,
             "--epochs", str(args.epochs)], "step 2/4  CLIP linear probe (external 18-generator data)")

    if "3" in steps:
        run([py, "repo/scripts/evaluate_system.py", "--device", args.device],
            "step 3/4  full evaluation on the 119 samples + latency benchmark")

    if "4" in steps:
        (WS / "demo_assets").mkdir(exist_ok=True)
        run([py, "repo/scripts/make_test_video.py",
             "--images", "data/images/001.png", "data/images/057.png",
             "--out", "demo_assets/smoke_test.mp4"], "step 4/4  video smoke clip")
        run([py, "repo/scripts/predict.py", "--video", "demo_assets/smoke_test.mp4",
             "--out", "reports"], "step 4b  video channel inference")

    print("\nALL REQUESTED STEPS COMPLETED")
    print("Artifacts: features.csv, models/*.pt|joblib|json, reports/metrics.json, "
          "reports/*.html, demo_assets/smoke_test.mp4")


if __name__ == "__main__":
    main()
