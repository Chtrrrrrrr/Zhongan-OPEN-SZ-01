#!/usr/bin/env python3
"""One-command reproduction of the OPEN-SZ-01 demo results.

Runs the whole pipeline with the exact ordering used for the reported numbers:

  step 1  L1 + L1b forensic features        -> features.csv, models/l1b_feats.npy
  step 2  CLIP features (global + tiles)    -> models/clip_feats*.npy, clip_tiles*.npy
  step 2b patch-token statistics + the evidence sweep + the frozen feature spec
                                            -> models/clip_patch.npy, feature_spec.json,
                                               clip_tiles_spec.npy, reports/tile_clip_sweep.json
  step 3  CLIP probe training (external)    -> models/clip_probe.pt (+_norm)
  step 4  full evaluation on 119 samples    -> reports/metrics.json + reports/*.html
  step 5  ablations (features / fusion)     -> reports/*_ablation.json
  step 6  error analysis + figures          -> reports/error_analysis.csv, assets/*.png
  step 7  smoke test + video channel        -> reports/smoke_test.json, video report
  step 8  metrics pack (manual 12.1/13.1/15.1)
                                            -> reports/calibration.json,
                                               reports/member_analysis.json,
                                               reports/threshold_stability.json,
                                               reports/stratified_analysis.json
  step 9  per-stage latency + GPU memory    -> reports/resource_profile.json (manual 16.2)
  step 10 compliance pack                   -> reports/cross_generator.json,
                                               reports/aeroblade_ablation.json,
                                               reports/meta_proxy_attempt.json,
                                               data/eval_protocol_frozen.json
  step 11 multi-round verification          -> reports/verify_matrix.json,
                                               models/deployment_metrics.json
  step 12 real-scene-photo fix (optional)   -> models/external_corpus/,
                                               reports/external_reals_fix.json,
                                               reports/fusion_tuning_reals.json,
                                               models/fix_metrics.json (NOT archived; the in-repo
                                               stand-in is reports/fix_metrics_restored.json)
  step 13 external holdout (optional)       -> reports/external_holdout.json

Usage:
  python scripts/run_all.py                      # everything
  python scripts/run_all.py --steps 1,2,4        # only some steps
  python scripts/run_all.py --steps 8            # just the metrics pack (fast, no GPU ops
                                                 # beyond feature reuse; rebuilds the OOF
                                                 # matrix under models/ on first run)
  python scripts/run_all.py --skip-clip-train    # reuse an existing probe
  python scripts/run_all.py --with-robustness    # add the (slow) robustness matrix
  python scripts/run_all.py --testset <dir>      # also run the external holdout
  python scripts/run_all.py --steps 12 --testset <dir>
                                                 # rebuild the real-scene-photo fix from
                                                 # the external corpus (~55 min on GPU)

Everything is written under the workspace so the package is self-contained.

Which artifact copy is loaded
-----------------------------
This script REGENERATES artifacts into the workspace `models/` (including a fresh
`models/fusion.json`), which is why the workspace copy is not the frozen deployment
copy. The read-only verification steps (7 smoke test, 13 external holdout) resolve
their artifacts through `scripts/models_path.py`, which prefers the frozen
`<repo>/models` copy and logs `source=repo` / `source=workspace-fallback` per file.
Pass `--models-dir <dir>` to point those steps at an explicit copy.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo/
WS = ROOT.parent                                    # workspace root
sys.path.insert(0, str(ROOT / "scripts"))

import models_path as mp                            # noqa: E402


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
    ap.add_argument("--steps", default="1,2,2b,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--skip-clip-train", action="store_true")
    ap.add_argument("--with-robustness", action="store_true",
                    help="also run the robustness matrix (about 30-40 min on GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sweep-configs", default="C0,C5,C6,C7,C8,C9,M1,M2,M3,M5",
                    help="tile-CLIP cells for the evidence sweep (step 2b)")
    ap.add_argument("--deploy-cell", default="C8",
                    help="sweep cell frozen into models/feature_spec.json (step 2b)")
    ap.add_argument("--deploy-aggregation", default="meanmax")
    ap.add_argument("--verify-rounds", type=int, default=6)
    ap.add_argument("--testset", default="",
                    help="optional external 真假测试素材集 for the robustness holdout")
    ap.add_argument("--scene-photo-fix", action="store_true",
                    help="run step 12: rebuild the real-scene-photo false-positive fix "
                         "(external corpus extraction + retrain + freeze, ~55 min on GPU). "
                         "Needs --testset.")
    ap.add_argument("--official-weight", type=float, default=300.0,
                    help="sample_weight for competition rows when external material is "
                         "added to training (step 12)")
    ap.add_argument("--models-dir", default=None,
                    help="artifact directory for the READ-ONLY verification steps (7 smoke test, "
                         "13 external holdout); default resolution prefers the frozen "
                         "<repo>/models copy - see scripts/models_path.py")
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",")}
    py = sys.executable

    # State up front which fusion.json the verification steps will honour.
    _md, _src = mp.resolve_models_dir(args.models_dir, workspace=WS)
    print(mp.describe(_md, _src))
    print(mp.assert_expected_fusion(_md))
    _fwd = ["--models-dir", args.models_dir] if args.models_dir else []

    if "1" in steps:
        run([py, "repo/scripts/extract_l1b_features.py", "--data-dir", "data",
             "--features", "features.csv", "--out-dir", "models"],
            "step 1/12  L1 (47) + L1b (20) forensic features")

    if "2" in steps:
        run([py, "repo/scripts/extract_clip_features.py", "--data-dir", "data",
             "--out-dir", "models", "--device", args.device],
            "step 2/12  CLIP global + tiled features for the 119 samples")

    if "2b" in steps:
        run([py, "repo/scripts/extract_clip_patch_features.py", "--data-dir", "data",
             "--out-dir", "models", "--device", args.device],
            "step 2b/12a  CLIP patch-token statistics (training-free clue family)")
        run([py, "repo/scripts/tile_clip_sweep.py", "--data-dir", "data",
             "--cache", "models/tile_cache", "--configs", args.sweep_configs,
             "--aggregations", "base,meanmax", "--seeds", "42,7,123,2024",
             "--out", "repo/reports/tile_clip_sweep.json", "--device", args.device],
            "step 2b/12b  tile-CLIP evidence sweep (accuracy at the FP budget vs latency)")
        run([py, "repo/scripts/build_feature_spec.py", "--cell", args.deploy_cell,
             "--aggregation", args.deploy_aggregation, "--patch",
             "--cache", "models/tile_cache", "--models-dir", "models"],
            "step 2b/12c  freeze models/feature_spec.json + clip_tiles_spec.npy")

    if "3" in steps and not args.skip_clip_train:
        shards = sorted((WS / "data" / "external").glob("*.parquet"))
        if not shards:
            raise SystemExit("no parquet shards in data/external - see docs/reproduce.md")
        run([py, "repo/scripts/train_clip_probe.py",
             "--shards", *[str(s) for s in shards],
             "--out", "models/clip_probe_norm.pt", "--device", args.device,
             "--epochs", str(args.epochs), "--reencode-q", "90"],
            "step 3/12  CLIP probe on the 18-generator benchmark (common-codec)")

    if "4" in steps:
        run([py, "repo/scripts/evaluate_system.py", "--device", args.device,
             "--clip-probe", "models/clip_probe_norm.pt",
             "--feature-spec", "models/feature_spec.json"],
            "step 4/12  nested-CV evaluation + latency + per-sample reports")

    if "5" in steps:
        run([py, "repo/scripts/experiment_features.py"], "step 5a/12  feature-set ablation")
        run([py, "repo/scripts/experiment_fusion.py"], "step 5b/12  fusion-rule ablation")

    if "6" in steps:
        run([py, "repo/scripts/diagnose_errors.py"], "step 6a/12  error analysis")
        run([py, "repo/scripts/make_figures.py"], "step 6b/12  figures")

    if "7" in steps:
        (WS / "demo_assets").mkdir(exist_ok=True)
        run([py, "repo/scripts/make_test_video.py",
             "--images", "data/images/001.png", "data/images/057.png",
             "--out", "demo_assets/smoke_test.mp4"], "step 7a/12  video smoke clip")
        run([py, "repo/scripts/smoke_test.py", "--device", args.device, *_fwd],
            "step 7b/12  cold/warm image + video budget check")

    if "8" in steps:
        # the whole metrics pack shares one out-of-fold score matrix: the first script
        # builds and caches it under models/, the rest reuse it (they all reproduce the
        # fold protocol of scripts/evaluate_system.py)
        run([py, "repo/scripts/calibration_analysis.py"],
            "step 8a/12  calibration: ECE (10/15 bins) + reliability + guardrails "
            "FPR@TPR=95% / TPR@FPR=1%")
        run([py, "repo/scripts/member_analysis.py"],
            "step 8b/12  member correlation, decision disagreement, FP/FN overlap, "
            "leave-one-out ΔTPR")
        run([py, "repo/scripts/threshold_stability.py"],
            "step 8c/12  threshold bootstrap CI + sensitivity + hold-out FPR")
        run([py, "repo/scripts/stratified_analysis.py"],
            "step 8d/12  per-content-class / per-capture-mode breakdown (proxy labels)")
        run([py, "repo/scripts/protocol_reconciliation.py"],
            "step 8e/12  frozen-split vs out-of-fold protocol reconciliation + signature")

    if "9" in steps:
        run([py, "repo/scripts/resource_profile.py", "--device", args.device],
            "step 9/12  per-stage latency + GPU memory + manual 16.2 budget table")

    if "10" in steps:
        run([py, "repo/scripts/build_meta.py"],
            "step 10a  D2/D3 stratification key attempt + validity check")
        run([py, "repo/scripts/aeroblade_ablation.py"],
            "step 10b  AEROBLADE ablation (optional member, manual 12.1)")
        run([py, "repo/scripts/cross_generator_test.py", "--max-per-shard", "200"],
            "step 10c  cross-generator test: whole generators removed from training")
        run([py, "repo/scripts/stratified_by_meta.py"],
            "step 10d  per-stratum metrics from the manual labels (shortcut test)")
        run([py, "repo/scripts/video_channel_validation.py"],
            "step 10e  controlled video-channel validation (30 clips)")
        run([py, "repo/scripts/freeze_eval_protocol.py"],
            "step 10f  freeze the evaluation protocol + signature")
        run([py, "repo/scripts/freeze_eval_protocol.py", "--verify"],
            "step 10g  verify the evaluation-protocol freeze")

    if "11" in steps:
        run([py, "repo/scripts/calibrate_deployment.py", "--data-dir", "data",
             "--models-dir", "models", "--rounds", str(args.verify_rounds),
             "--member-set", "combined_forensic"],
            "step 11a/12  freeze heads + thresholds, multi-round out-of-fold metrics")
        run([py, "repo/scripts/verify_matrix.py", "--rounds", str(args.verify_rounds),
             "--data-dir", "data", "--cache", "models/tile_cache",
             "--out", "repo/reports/verify_matrix.json"],
            "step 11b/12  verification matrix (config x aggregation x member set)")

    if args.scene_photo_fix:
        if not args.testset:
            raise SystemExit("--scene-photo-fix needs --testset <dir> for the corpus")
        ts = args.testset
        run([py, "repo/scripts/extract_external_corpus.py",
             "--out", "models/external_corpus",
             "--labels", f"{ts}/real=0,{ts}/real_synthbuster=0,{ts}/fake_ai=1,"
                         f"{ts}/fake_dataset=1,{ts}/fake_tamper=1",
             "--feature-spec", "models/feature_spec.json", "--device", args.device,
             "--resume"],
            "step 12a/13  external corpus features (real + generated, current spec)")
        run([py, "repo/scripts/train_with_external_reals.py",
             "--corpus", "models/external_corpus", "--rounds", str(args.verify_rounds),
             "--variants", "baseline_official_only",
             "--weight-grid", str(int(args.official_weight)),
             "--extra-seeds", str(args.verify_rounds),
             "--out", "repo/reports/external_reals_fix.json"],
            "step 12b/13  before/after: external material in the training folds only")
        run([py, "repo/scripts/tune_fusion_for_reals.py",
             "--corpus", "models/external_corpus",
             "--official-weights", f"100,{int(args.official_weight)},1000",
             "--out", "repo/reports/fusion_tuning_reals.json"],
            "step 12c/13  fusion weights x official up-weight (claim-photo safety)")
        run([py, "repo/scripts/freeze_external_fix.py",
             "--corpus", "models/external_corpus", "--corpus-mode", "both",
             "--official-weight", str(int(args.official_weight)),
             "--rounds", str(args.verify_rounds), "--threshold", "official"],
            "step 12d/13  freeze heads + thresholds, write models/fix_metrics.json")

    if args.testset:
        run([py, "repo/scripts/external_holdout.py", "--testset", args.testset,
             "--cache", "models/external_cache", "--stage", "extract",
             "--device", args.device, *_fwd],
            "step 13a  external testset feature extraction")
        run([py, "repo/scripts/external_holdout.py", "--testset", args.testset,
             "--cache", "models/external_cache", "--stage", "evaluate",
             "--out", "repo/reports/external_holdout.json", *_fwd],
            "step 13b  external holdout evaluation (frozen thresholds)")

    if args.with_robustness:
        run([py, "repo/scripts/robustness_matrix.py", "--device", args.device],
            "extra  robustness matrix (manual ch.14.1)")

    print("\nALL REQUESTED STEPS COMPLETED")
    print("Key artifacts: features.csv, models/*.pt|joblib|json, reports/metrics.json, "
          "reports/*.html, repo/assets/*.png, demo_assets/smoke_test.mp4")
    print("Metrics pack: reports/calibration.json, reports/member_analysis.json, "
          "reports/threshold_stability.json, reports/stratified_analysis.json, "
          "reports/resource_profile.json, reports/protocol_reconciliation.json")
    print("Iteration pack: reports/tile_clip_sweep.json, reports/verify_matrix.json, "
          "models/deployment_metrics.json, models/feature_spec.json")
    print("Compliance pack: reports/cross_generator.json, reports/aeroblade_ablation.json, "
          "reports/meta_proxy_attempt.json, data/eval_protocol_frozen.json")


if __name__ == "__main__":
    main()
