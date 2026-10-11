#!/usr/bin/env python3
"""Run the verification matrix (config x aggregation x member set) and tabulate it.

Every cell is the same honest protocol: stratified 5-fold x N rounds, per-fold
threshold from the nested inner-CV out-of-fold scores of the training fold, heads fit
on the training fold only. The point of the matrix is to separate real movement from
CV noise before anything is frozen into the deployed version.

Usage
  python scripts/verify_matrix.py --rounds 6
  python scripts/verify_matrix.py --cells C0:base,C7:meanmax --weights deployed,no_clip_domain
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WEIGHT_SETS = {
    "deployed": {"combined": 0.60, "forensic": 0.25},                # archived deployment
    "with_clip_domain": {"combined": 0.60, "forensic": 0.25, "clip_domain": 0.15},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="C0:base,C0:meanmax,C7:base,C7:meanmax,C7:lse",
                    help="comma list of config:aggregation")
    ap.add_argument("--weights", default="deployed,with_clip_domain")
    ap.add_argument("--patch", default="yes,no", help="comma list: yes/no")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache", default="models/tile_cache")
    ap.add_argument("--out-dir", default="reports_local/matrix")
    ap.add_argument("--out", default="reports_local/verify_matrix.json")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    weight_sets = [w.strip() for w in args.weights.split(",") if w.strip()]
    patches = [p.strip() == "yes" for p in args.patch.split(",") if p.strip()]

    rows = []
    for cell in cells:
        cfg, agg = cell.split(":")
        for wname in weight_sets:
            if wname == "deployed" and cfg != "C0" and wname in weight_sets and len(weight_sets) == 1:
                pass
            for use_patch in patches:
                tag = f"{cfg}_{agg}_{wname}_patch{int(use_patch)}"
                out = out_dir / f"{tag}.json"
                if out.exists():
                    blob = json.loads(out.read_text(encoding="utf-8"))
                else:
                    cmd = [sys.executable, str(ROOT / "scripts" / "verify_rounds.py"),
                           "--data-dir", str(ROOT / args.data_dir),
                           "--cache", str(ROOT / args.cache),
                           "--config", cfg, "--aggregation", agg,
                           "--rounds", str(args.rounds), "--n-folds", str(args.n_folds),
                           "--fp-budget", str(args.fp_budget),
                           "--weights", json.dumps(WEIGHT_SETS[wname]),
                           "--out", str(out)]
                    if not use_patch:
                        cmd.append("--no-patch")
                    t0 = time.perf_counter()
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode != 0:
                        print(f"[{tag}] FAILED rc={proc.returncode}\n{proc.stderr[-1500:]}")
                        continue
                    blob = json.loads(out.read_text(encoding="utf-8"))
                    blob["_seconds"] = round(time.perf_counter() - t0, 1)
                r = blob["rules"]
                row = {
                    "tag": tag, "cell": cfg, "aggregation": agg, "weights": wname,
                    "patch": use_patch,
                    "members": blob["fusion"]["weights"],
                    "full_auc": blob["fusion"]["full_oof_auc_mean"],
                    "acc_mean": r["fused_threshold"]["acc_mean"],
                    "acc_min": r["fused_threshold"]["acc_min"],
                    "acc_max": r["fused_threshold"]["acc_max"],
                    "fpr_mean": r["fused_threshold"]["fpr_mean"],
                    "tpr_mean": r["fused_threshold"]["tpr_mean"],
                    "oracle_acc": r["oracle_2fp"]["acc_mean"],
                    "two_hits_acc": r["two_hits_review"]["acc_mean"],
                    "loo": blob["fusion"]["leave_one_out_auc"],
                    "per_member": {k: v["auc_mean"]
                                   for k, v in blob["per_member_oof_auc"].items()},
                }
                rows.append(row)
                print(f"[{tag:38s}] auc={row['full_auc']:.4f} "
                      f"acc={row['acc_mean']:.4f} [{row['acc_min']:.4f},{row['acc_max']:.4f}] "
                      f"fpr={row['fpr_mean']:.4f} tpr={row['tpr_mean']:.4f} "
                      f"oracle={row['oracle_acc']:.4f} loo={row['loo']}", flush=True)

    rows.sort(key=lambda r: (-r["acc_mean"], -r["full_auc"]))
    payload = {
        "report": "verify_matrix",
        "rounds": args.rounds, "n_folds": args.n_folds, "fp_budget": args.fp_budget,
        "protocol": ("stratified k-fold x rounds; per-fold nested inner-CV threshold; "
                     "heads fit on the training fold only; no held-out selection"),
        "rows": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out}\n\n=== ranked by honest accuracy ===")
    for r in rows:
        print(f"  {r['tag']:38s} acc={r['acc_mean']:.4f} fpr={r['fpr_mean']:.4f} "
              f"auc={r['full_auc']:.4f} oracle={r['oracle_acc']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
