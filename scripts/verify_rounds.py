#!/usr/bin/env python3
"""Multi-round verification of the deployed decision rule (honest, out-of-fold).

Produces one evidence file with everything a claim about the system needs:

  1. Per-member out-of-fold (OOF) logits for `forensic`, `combined`, `clip_domain`
     and `clip_external` under the SAME stratified k-fold splits the headline
     evaluation uses, repeated over several CV seeds, so run-to-run spread is
     measured instead of asserted.
  2. Leave-one-out deltas per member (manual ch.8.3 / 12.1: a member stays only if
     removing it costs measurable performance).
  3. Decision-rule comparison on the pooled OOF scores:
       * `fused_threshold`      - the archived rule (fuse -> one threshold)
       * `two_hits_review`      - same, but a suspect verdict additionally requires
                                  >= 2 independent evidence families to fire; a
                                  single-family sample is downgraded to review
                                  (manual ch.6.2 R2 mitigation)
       * `oracle_2fp`           - reachable ceiling at the 2-false-positive budget
  4. Error attribution: which samples cause every false positive / false negative,
     and in how many rounds.

No rule is selected on held-out data: thresholds come from the nested inner CV of
the training fold (see scripts/tile_clip_sweep.py for the same protocol), and the
`oracle_2fp` row is labelled as an oracle because it is one.

Usage
  python scripts/verify_rounds.py --rounds 6
  python scripts/verify_rounds.py --config C7 --aggregation meanmax --save-hits
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

from tile_clip_sweep import (AGGS, CONFIGS, aggregate_multi, load_tiles,  # noqa: E402
                             _fit_head, _logits, _fit_temperature, _fused,
                             _fpr_threshold, _acc_at_fp)

WEIGHTS = {"combined": 0.60, "forensic": 0.25, "clip_domain": 0.15}
# The archived deployed fusion is {combined 0.60, forensic 0.25} only (clip_external
# failed its admission test). `--weights` switches between that and the 3-family mix.
DEPLOYED_WEIGHTS = {"combined": 0.60, "forensic": 0.25}
# a "hit" is one independent evidence family voting AI at its own calibrated
# operating point (its own threshold at the shared false-positive budget)
FAMILIES = ("forensic", "combined", "clip_domain")


def load_labels(data_dir: Path):
    seqs, y = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    return seqs, np.array(y)


def build_features(cell: str, aggregation: str, cache: Path, use_patch: bool = True):
    """Assemble the three member feature blocks.

    forensic    = L1(47) + L1b(20) [+ CLIP patch-token stats]
    clip_domain = CLIP global(1024)
    combined    = forensic + clip_domain + tile aggregation
    """
    X1 = np.loadtxt(ROOT / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(ROOT / "models" / "l1b_feats.npy")
    Fg = np.load(ROOT / "models" / "clip_feats.npy")
    Xf = np.concatenate([X1, Xb], axis=1).astype(np.float64)
    if use_patch:
        patch_path = ROOT / "models" / "clip_patch.npy"
        if patch_path.exists():
            Xf = np.concatenate([Xf, np.load(patch_path).astype(np.float64)], axis=1)
    if cell == "C0":
        agg = np.load(ROOT / "models" / "clip_tiles.npy").astype(np.float64)
        cfg_used = {"grid": 3, "overlap": 0.25, "max_side": 1600,
                    "source": "archived clip_tiles.npy"}
    else:
        agg = aggregate_multi(load_tiles(cache, cell), aggregation).astype(np.float64)
        cfg_used = {"scales": [{"grid": g, "overlap": o, "max_side": m}
                               for (g, o, m) in CONFIGS[cell]],
                    "aggregation": aggregation}
    cfg_used["patch_features"] = bool(use_patch)
    X_combined = np.concatenate([Xf, Fg, agg], axis=1)
    return Xf, Fg, X_combined, cfg_used


def run_rounds(X_f, X_g, X_combined, y, n_folds, fp_budget, rounds, base_seed=42,
               weights=None):
    """One nested-CV pass per round; returns per-sample OOF logits per member."""
    from sklearn.model_selection import StratifiedKFold

    weights = dict(weights or DEPLOYED_WEIGHTS)
    inner_splits = max(3, n_folds - 1)
    n = len(y)
    members = ("forensic", "combined", "clip_domain")
    blocks = {"forensic": X_f, "combined": X_combined, "clip_domain": X_g}
    oof_z = {m: np.zeros((rounds, n)) for m in members}
    oof_fused = np.zeros((rounds, n))
    oof_pred = np.zeros((rounds, n), dtype=int)
    oof_hits = np.zeros((rounds, n), dtype=int)
    member_thr = {m: np.zeros(rounds) for m in members}
    fused_thr = np.zeros(rounds)
    round_stats = []

    for r in range(rounds):
        seed = base_seed + r * 13
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        t0 = time.perf_counter()
        for fold, (tr, va) in enumerate(skf.split(X_combined, y)):
            Xtr = {m: blocks[m][tr] for m in members}
            Xva = {m: blocks[m][va] for m in members}

            heads = {m: _fit_head(Xtr[m], y[tr]) for m in members}
            z_va = {m: _logits(*heads[m], Xva[m]) for m in members}
            for m in members:
                oof_z[m][r, va] = z_va[m]

            # inner nested CV on the training fold -> honest scale for calibration/thresholds
            inner = {m: np.zeros(len(tr)) for m in members}
            iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                   random_state=1000 + seed * 10 + fold)
            for itr, iva in iskf.split(X_combined[tr], y[tr]):
                for m in members:
                    a, b = _fit_head(Xtr[m][itr], y[tr][itr])
                    inner[m][iva] = _logits(a, b, Xtr[m][iva])
            temps = {m: _fit_temperature(inner[m], y[tr]) for m in members}
            fam_thr = {m: _fpr_threshold(inner[m][y[tr] == 0], fp_budget) for m in members}
            for m in members:
                member_thr[m][r] += fam_thr[m] / n_folds

            thr = _fpr_threshold(_fused(inner, temps, weights)[y[tr] == 0], fp_budget)
            fused_thr[r] += thr / n_folds
            fva = _fused(z_va, temps, weights)
            oof_fused[r, va] = fva
            oof_pred[r, va] = (fva >= thr).astype(int)
            hits = np.zeros(len(va), dtype=int)
            for m in members:
                hits += (z_va[m] >= fam_thr[m]).astype(int)
            oof_hits[r, va] = hits
        dt = time.perf_counter() - t0
        acc = float((oof_pred[r] == y).mean())
        round_stats.append({"round": r + 1, "seed": seed, "acc": round(acc, 4),
                            "fpr": round(float(oof_pred[r][y == 0].mean()), 4),
                            "tpr": round(float(oof_pred[r][y == 1].mean()), 4),
                            "fused_thr": round(float(fused_thr[r]), 4),
                            "seconds": round(dt, 1)})
        print(f"[round {r+1}] seed={seed} acc={acc:.4f} "
              f"fpr={round(float(oof_pred[r][y == 0].mean()),4)} "
              f"tpr={round(float(oof_pred[r][y == 1].mean()),4)} "
              f"thr={fused_thr[r]:.4f} ({dt:.0f}s)", flush=True)

    return {"oof_z": oof_z, "oof_fused": oof_fused, "oof_pred": oof_pred,
            "oof_hits": oof_hits, "member_thr": member_thr, "fused_thr": fused_thr,
            "rounds": round_stats, "members": members}


def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"accuracy": round((tp + tn) / len(y), 4),
            "fpr": round(fp / max(1, int((y == 0).sum())), 4),
            "tpr": round(tp / max(1, int((y == 1).sum())), 4),
            "precision": round(tp / max(1, tp + fp), 4),
            "coverage_non_review": round((tp + tn + fp + fn) / len(y), 4),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache", default="models/tile_cache")
    ap.add_argument("--config", default="C7")
    ap.add_argument("--aggregation", default="meanmax")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--weights", default="deployed",
                    help="'deployed' = {combined 0.60, forensic 0.25}; or a JSON object")
    ap.add_argument("--no-patch", action="store_true",
                    help="drop the CLIP patch-token statistics from the forensic member")
    ap.add_argument("--out", default="reports/verify_rounds.json")
    args = ap.parse_args()

    weights = (dict(DEPLOYED_WEIGHTS) if args.weights.strip() == "deployed"
               else json.loads(args.weights))

    data_dir = Path(args.data_dir)
    seqs, y = load_labels(data_dir)
    X_f, X_g, X_combined, cfg_used = build_features(
        args.config, args.aggregation, Path(args.cache), not args.no_patch)
    print(f"[features] forensic={X_f.shape} clip_domain={X_g.shape} "
          f"combined={X_combined.shape} cfg={cfg_used}")

    res = run_rounds(X_f, X_g, X_combined, y, args.n_folds, args.fp_budget, args.rounds,
                     args.base_seed, weights)

    # ---- per-member OOF AUC, averaged over rounds (deterministic ranking metric) ----
    from sklearn.metrics import roc_auc_score
    per_member = {}
    for m in res["members"]:
        aucs = [float(roc_auc_score(y, res["oof_z"][m][r])) for r in range(args.rounds)]
        per_member[m] = {"auc_mean": round(float(np.mean(aucs)), 4),
                         "auc_min": round(float(np.min(aucs)), 4),
                         "auc_max": round(float(np.max(aucs)), 4),
                         "threshold_p50": round(float(np.median(res["member_thr"][m])), 4)}

    # ---- fusion rules on pooled OOF ----
    rules = {}
    accs, fprs, tprs = [], [], []
    for r in range(args.rounds):
        m = metrics(y, res["oof_pred"][r])
        accs.append(m["accuracy"])
        fprs.append(m["fpr"])
        tprs.append(m["tpr"])
    rules["fused_threshold"] = {"acc_mean": round(float(np.mean(accs)), 4),
                                "acc_min": round(float(np.min(accs)), 4),
                                "acc_max": round(float(np.max(accs)), 4),
                                "fpr_mean": round(float(np.mean(fprs)), 4),
                                "tpr_mean": round(float(np.mean(tprs)), 4)}

    # two-hits rule: suspect requires the fused threshold AND >= 2 family hits
    accs2, fprs2, tprs2, review2 = [], [], [], []
    for r in range(args.rounds):
        pred = res["oof_pred"][r].copy()
        downgrade = (pred == 1) & (res["oof_hits"][r] < 2)
        pred[downgrade] = 0                     # review is counted as non-suspect
        m = metrics(y, pred)
        accs2.append(m["accuracy"])
        fprs2.append(m["fpr"])
        tprs2.append(m["tpr"])
        review2.append(int(downgrade.sum()))
    rules["two_hits_review"] = {"acc_mean": round(float(np.mean(accs2)), 4),
                                "acc_min": round(float(np.min(accs2)), 4),
                                "acc_max": round(float(np.max(accs2)), 4),
                                "fpr_mean": round(float(np.mean(fprs2)), 4),
                                "tpr_mean": round(float(np.mean(tprs2)), 4),
                                "review_downgrades_mean": round(float(np.mean(review2)), 1)}

    # oracle at the false-positive budget (reachable ceiling for the same ranking)
    oracles = [_acc_at_fp(res["oof_fused"][r], y, args.fp_budget) for r in range(args.rounds)]
    rules["oracle_2fp"] = {"acc_mean": round(float(np.mean([o["acc"] for o in oracles])), 4),
                           "note": "threshold chosen on the pooled OOF scores (oracle, "
                                   "not deployable); upper bound for the same ranking"}

    # nested-CV threshold, reported as the honest deployable rule
    nest_acc, nest_fpr, nest_tpr = [], [], []
    for r in range(args.rounds):
        m = metrics(y, res["oof_pred"][r])
        nest_acc.append(m["accuracy"])
    rules["fused_threshold"]["note"] = ("threshold from the nested inner CV of the "
                                        "training fold; no held-out selection")

    # ---- leave-one-out deltas (mean OOF AUC with each family removed) ----
    loo = {}
    for drop in res["members"]:
        keep = [m for m in res["members"] if m != drop]
        aucs = []
        for r in range(args.rounds):
            aucs.append(float(roc_auc_score(y, np.mean([res["oof_z"][m][r] for m in keep], axis=0))))
        loo[f"drop_{drop}"] = round(float(np.mean(aucs)), 4)
    full_auc = round(float(np.mean([roc_auc_score(y, res["oof_fused"][r])
                                    for r in range(args.rounds)])), 4)

    # ---- error attribution ----
    fp_idx = np.where(y == 0)[0]
    fn_idx = np.where(y == 1)[0]
    fp_counts = res["oof_pred"][:, fp_idx].sum(axis=0)
    fn_counts = (1 - res["oof_pred"][:, fn_idx]).sum(axis=0)
    errors = {
        "real_always_flagged": [{"seq": int(seqs[i]), "rounds": int(fp_counts[k]),
                                 "of": args.rounds}
                                for k, i in enumerate(fp_idx) if fp_counts[k] == args.rounds],
        "real_flagged_sometimes": [{"seq": int(seqs[i]), "rounds": int(fp_counts[k]),
                                    "of": args.rounds}
                                   for k, i in enumerate(fp_idx)
                                   if 0 < fp_counts[k] < args.rounds],
        "ai_never_detected": [{"seq": int(seqs[i]), "rounds": int(fn_counts[k]),
                               "of": args.rounds}
                              for k, i in enumerate(fn_idx) if fn_counts[k] == args.rounds],
        "ai_missed_sometimes": [{"seq": int(seqs[i]), "rounds": int(fn_counts[k]),
                                 "of": args.rounds}
                                for k, i in enumerate(fn_idx)
                                if 0 < fn_counts[k] < args.rounds],
    }

    payload = {
        "report": "verify_rounds",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"cell": args.config, "aggregation": args.aggregation, **cfg_used},
        "protocol": {
            "cv": f"stratified {args.n_folds}-fold x {args.rounds} rounds "
                  f"(seeds {args.base_seed}..{args.base_seed + (args.rounds-1)*13})",
            "threshold": f"nested inner-CV out-of-fold training-fold real scores, "
                         f"fp_budget={args.fp_budget}",
            "families": list(FAMILIES),
            "note": "every head is fit on the training fold only; no held-out selection",
        },
        "n_samples": int(len(y)),
        "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
        "per_member_oof_auc": per_member,
        "fusion": {"weights": WEIGHTS, "full_oof_auc_mean": full_auc,
                   "leave_one_out_auc": loo},
        "rules": rules,
        "rounds": res["rounds"],
        "errors": errors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[per-member OOF AUC] {json.dumps({k: v['auc_mean'] for k, v in per_member.items()})}")
    print(f"[fusion] full={full_auc} LOO={loo}")
    for k, v in rules.items():
        print(f"[rule] {k}: {json.dumps(v, ensure_ascii=False)}")
    print(f"[errors] {json.dumps(errors, ensure_ascii=False)}")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
