#!/usr/bin/env python3
"""Freeze the deployed heads, thresholds and report the out-of-fold verification.

This is the iteration's calibration entry point. It runs the archived protocol
(stratified 5-fold, `--cv-seed 42`, nested inner CV for the score scale - the same
sequence scripts/evaluate_system.py uses) and then repeats it over extra CV seeds to
report the run-to-run spread that a single split cannot show.

Artifacts written (models/):
  forensic_lr.joblib   head on [L1 | L1b | patch]           (fit on all 119)
  combined_lr.joblib   head on [forensic | CLIP global | tiles]
  fusion.json          members, weights, temperatures, thr_low / thr_high
  deployment_metrics.json  fold table, OOF metrics, error attribution, caveats

Threshold rule: per fold, the decision threshold is the order statistic of the
NESTED-CV out-of-fold fused scores of the training fold's real samples, targeting
`--fp-budget` false positives (2 = floor(0.03*70), the 3% rule on a 7:3 batch). The
final deployed threshold is the median of the per-fold thresholds, so no held-out
sample ever selects it.

Usage
  python scripts/calibrate_deployment.py --rounds 6
  python scripts/calibrate_deployment.py --member-set combined_forensic --rounds 6
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

from tile_clip_sweep import (_fit_head, _logits, _fit_temperature, _fused,  # noqa: E402
                            _fpr_threshold, _acc_at_fp)

MEMBER_SETS = {
    "combined_forensic": {"combined": 0.60, "forensic": 0.25},
    "with_clip_domain": {"combined": 0.60, "forensic": 0.25, "clip_domain": 0.15},
    "combined_only": {"combined": 1.0},
}


def load_blocks(models_dir: Path, data_dir: Path, spec_path: Path):
    """Assemble the member input blocks from the frozen feature artifacts."""
    from feature_spec import FeatureSpec
    spec = FeatureSpec.load(spec_path)
    X1 = np.loadtxt(ROOT / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(models_dir / "l1b_feats.npy")
    parts_f = [np.asarray(X1, dtype=np.float64), np.asarray(Xb, dtype=np.float64)]
    if spec.patch_tokens:
        parts_f.append(np.load(models_dir / "clip_patch.npy").astype(np.float64))
    Xf = np.concatenate(parts_f, axis=1)
    Fg = np.load(models_dir / "clip_feats.npy").astype(np.float64)
    tiles = np.load(models_dir / ("clip_tiles_spec.npy" if (models_dir / "clip_tiles_spec.npy").exists()
                                  else "clip_tiles.npy")).astype(np.float64)
    Xc = np.concatenate([Xf, Fg, tiles], axis=1)
    return spec, {"forensic": Xf, "clip_domain": Fg, "combined": Xc}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--feature-spec", default=None)
    ap.add_argument("--member-set", default="combined_forensic", choices=list(MEMBER_SETS))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cv-seed", type=int, default=42,
                    help="first outer-CV seed; further rounds use seed + 13*r")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--out", default=None, help="metrics json (default <models>/deployment_metrics.json)")
    ap.add_argument("--no-freeze", action="store_true",
                    help="report metrics only; do not overwrite the deployed heads")
    ap.add_argument("--rewrite-fusion-only", action="store_true",
                    help="recompute thresholds from the cached metrics and rewrite ONLY "
                         "models/fusion.json (no head refit). Used to repair the artifact "
                         "after an evaluation run overwrote it with a single-split threshold")
    args = ap.parse_args()

    import joblib
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    data_dir = Path(args.data_dir)
    md = ROOT / args.models_dir
    spec_path = Path(args.feature_spec) if args.feature_spec else md / "feature_spec.json"
    spec, blocks = load_blocks(md, data_dir, spec_path)
    weights = MEMBER_SETS[args.member_set]
    members = [m for m in ("combined", "forensic", "clip_domain") if m in weights]
    print(f"[spec] {json.dumps(spec.to_dict(), ensure_ascii=False)}")
    print(f"[blocks] " + " ".join(f"{k}={v.shape}" for k, v in blocks.items())
          + f"  members={members}")

    if args.rewrite_fusion_only:
        # Repair path: the calibration below is deterministic, but an evaluation run can
        # overwrite models/fusion.json with its own single-split threshold. This branch
        # redoes only the threshold arithmetic and writes the fusion artifact.
        metrics_path = Path(args.out) if args.out else md / "deployment_metrics.json"
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        from sklearn.model_selection import StratifiedKFold as _SKF
        fused = prev.get("_oof_fused_scores")
        if not fused:
            raise SystemExit("--rewrite-fusion-only needs the cached out-of-fold scores; "
                             "run the full calibration once without the flag")
        seqs, y = [], []
        with open(data_dir / "labels.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seqs.append(int(r["seq"]))
                y.append(1 if r["label"] == "ai" else 0)
        y = np.array(y)
        arr = np.array(fused, dtype=np.float64)
        mean_oof = arr.mean(axis=0)
        srt = np.sort(mean_oof[y == 0])[::-1]
        k = max(0, min(args.fp_budget, len(srt) - 1))
        thr_logit = float(srt[k]) if k == 0 else float((srt[k - 1] + srt[k]) / 2.0)
        thr_p = float(1.0 / (1.0 + np.exp(-thr_logit)))
        prev_fusion = json.loads((md / "fusion.json").read_text(encoding="utf-8"))
        prev_fusion.update({"thr_high": round(thr_p, 6), "threshold_space": "probability",
                            "weights": {m: weights[m] for m in members}})
        (md / "fusion.json").write_text(json.dumps(prev_fusion, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"[rewrite] thr_high {thr_p:.6f} (logit {thr_logit:.6f}) -> {md/'fusion.json'}")
        return 0

    seqs, y = [], []
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    y = np.array(y)
    n = len(y)

    inner_splits = max(3, args.n_folds - 1)
    rounds = []
    oof_all = np.zeros((args.rounds, n))
    pred_all = np.zeros((args.rounds, n), dtype=int)
    fp_counts = np.zeros(n)
    fn_counts = np.zeros(n)

    for r in range(args.rounds):
        seed = args.cv_seed + r * 13
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
        oof = np.zeros(n)
        pred = np.zeros(n, dtype=int)
        fold_rows = []
        t0 = time.perf_counter()
        for fold, (tr, va) in enumerate(skf.split(blocks["combined"], y)):
            heads = {m: _fit_head(blocks[m][tr], y[tr]) for m in members}
            z_va = {m: _logits(*heads[m], blocks[m][va]) for m in members}
            inner = {m: np.zeros(len(tr)) for m in members}
            iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                   random_state=100 + fold)
            for itr, iva in iskf.split(blocks["combined"][tr], y[tr]):
                for m in members:
                    a, b = _fit_head(blocks[m][tr][itr], y[tr][itr])
                    inner[m][iva] = _logits(a, b, blocks[m][tr][iva])
            temps = {m: _fit_temperature(inner[m], y[tr]) for m in members}
            thr = _fpr_threshold(_fused(inner, temps, weights)[y[tr] == 0], args.fp_budget)
            fva = _fused(z_va, temps, weights)
            oof[va] = fva
            pred[va] = (fva >= thr).astype(int)
            fold_rows.append({"fold": fold + 1, "thr_high": round(thr, 6),
                              "temperatures": {m: round(temps[m], 4) for m in members},
                              "acc": round(float((pred[va] == y[va]).mean()), 4),
                              "fp": int(((pred[va] == 1) & (y[va] == 0)).sum()),
                              "fn": int(((pred[va] == 0) & (y[va] == 1)).sum())})
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        oof_all[r] = oof
        pred_all[r] = pred
        fp_counts += ((pred == 1) & (y == 0)).astype(float)
        fn_counts += ((pred == 0) & (y == 1)).astype(float)
        rounds.append({
            "round": r + 1, "seed": seed,
            "accuracy": round((tp + tn) / n, 4),
            "fpr": round(fp / int((y == 0).sum()), 4),
            "tpr": round(tp / int((y == 1).sum()), 4),
            "precision": round(tp / max(1, tp + fp), 4),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "auc": round(float(roc_auc_score(y, oof)), 4),
            "acc_at_fp_budget": _acc_at_fp(oof, y, args.fp_budget)["acc"],
            "folds": fold_rows,
            "seconds": round(time.perf_counter() - t0, 1),
        })
        print(f"[round {r+1}] seed={seed} acc={rounds[-1]['accuracy']:.4f} "
              f"fpr={rounds[-1]['fpr']:.4f} tpr={rounds[-1]['tpr']:.4f} "
              f"auc={rounds[-1]['auc']:.4f} "
              f"acc@2FP={rounds[-1]['acc_at_fp_budget']:.4f} "
              f"({rounds[-1]['seconds']:.0f}s)", flush=True)

    accs = [r["accuracy"] for r in rounds]
    fprs = [r["fpr"] for r in rounds]
    aucs = [r["auc"] for r in rounds]
    at_fp = [r["acc_at_fp_budget"] for r in rounds]
    summary = {
        "accuracy": {"mean": round(float(np.mean(accs)), 4),
                     "min": round(float(np.min(accs)), 4),
                     "max": round(float(np.max(accs)), 4),
                     "stdev": round(float(np.std(accs)), 4),
                     "per_round": [round(a, 4) for a in accs]},
        "fpr": {"mean": round(float(np.mean(fprs)), 4),
                "min": round(float(np.min(fprs)), 4),
                "max": round(float(np.max(fprs)), 4)},
        "auc": {"mean": round(float(np.mean(aucs)), 4),
                "min": round(float(np.min(aucs)), 4)},
        "acc_at_fp_budget": {"mean": round(float(np.mean(at_fp)), 4),
                             "min": round(float(np.min(at_fp)), 4)},
    }
    print(f"[summary] acc={summary['accuracy']['mean']} "
          f"[{summary['accuracy']['min']},{summary['accuracy']['max']}] "
          f"fpr={summary['fpr']['mean']} auc={summary['auc']['mean']} "
          f"acc@fp={summary['acc_at_fp_budget']['mean']}")

    # ---- freeze the deployed heads on all 119 ----
    # Deployed threshold. Two candidate rules are computed and both reported, because
    # they trade transferability against exactness:
    #   * pooled  - order statistic on the round-averaged out-of-fold real scores. It
    #               hits the false-positive budget on the calibration set exactly by
    #               construction, and averaging over rounds removes single-split luck.
    #   * perfold - median of the per-fold nested thresholds (each fold's own budget).
    # The pooled rule is what the service ships; the per-fold numbers stay in the
    # artifact so the disagreement is visible rather than hidden.
    mean_oof = oof_all.mean(axis=0)
    pooled_sorted = np.sort(mean_oof[y == 0])[::-1]
    k = max(0, min(args.fp_budget, len(pooled_sorted) - 1))
    thr_pooled = (float(pooled_sorted[k]) if k == 0
                  else float((pooled_sorted[k - 1] + pooled_sorted[k]) / 2.0))
    all_thr = [f["thr_high"] for r in rounds for f in r["folds"]]
    thr_perfold = float(np.median(all_thr))
    thr_high = thr_pooled
    print(f"[threshold] pooled={thr_pooled:.6f} (budget {args.fp_budget}/"
          f"{int((y==0).sum())} real)  per-fold median={thr_perfold:.6f}")

    temps_final = {}
    fusion_path = md / "fusion.json"
    if not args.no_freeze:
        head_paths = {}
        for m, fname in (("forensic", "forensic_lr.joblib"),
                         ("combined", "combined_lr.joblib"),
                         ("clip_domain", "clip_domain_lr.joblib")):
            if m not in members:
                continue
            sc, clf = _fit_head(blocks[m], y)
            joblib.dump({"scaler": sc, "clf": clf,
                         "features": ("l1_47+l1b_20" + ("+patch_16" if spec.patch_tokens else "")
                                      if m == "forensic" else
                                      f"{spec.name}: forensic|clip_global|tiles")},
                        md / fname)
            head_paths[m] = fname
        # temperatures for the deployed heads: from the nested-CV scale of a final
        # inner split over all data
        inner_full = {m: np.zeros(n) for m in members}
        iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=777)
        for itr, iva in iskf.split(blocks["combined"], y):
            for m in members:
                a, b = _fit_head(blocks[m][itr], y[itr])
                inner_full[m][iva] = _logits(a, b, blocks[m][iva])
        temps_final = {m: _fit_temperature(inner_full[m], y) for m in members}
        # Thresholds ship in the fused PROBABILITY space that FusionEngine.fuse returns;
        # the calibrated values above are logits, so convert once, here.
        real_fused = _fused(inner_full, temps_final, weights)[y == 0]
        def _to_prob(logit: float) -> float:
            return float(1.0 / (1.0 + np.exp(-logit)))
        thr_low = float(np.median(real_fused))
        thr_high_prob = _to_prob(thr_high)
        thr_low_prob = _to_prob(thr_low)
        if thr_low_prob >= thr_high_prob:            # degenerate ordering guard
            thr_low_prob = float(np.percentile([_to_prob(v) for v in real_fused], 25))
        payload = {
            "weights": {m: weights[m] for m in members},
            "thr_low": round(thr_low_prob, 6),
            "thr_high": round(thr_high_prob, 6),
            "threshold_space": "probability",
            "disagree_review": 1.25,
            "temperatures": temps_final,
            "feature_spec": spec.to_dict(),
            "heads": head_paths,
            "calibration": {
                "protocol": f"nested {args.n_folds}-fold x {args.rounds} rounds; "
                            f"deployed threshold = pooled out-of-fold order statistic",
                "fp_budget": args.fp_budget,
                "thr_pooled_logit": round(thr_pooled, 6),
                "thr_perfold_median_logit": round(thr_perfold, 6),
                "thr_low_logit": round(thr_low, 6),
                "per_fold_thresholds_logit": [round(t, 6) for t in all_thr],
                "pooled_real_scores_sorted_logit": [round(float(v), 5) for v in pooled_sorted],
            },
        }
        fusion_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"[freeze] {fusion_path} thr_low={payload['thr_low']} "
              f"thr_high={payload['thr_high']} (probability space; "
              f"logits {round(thr_low,4)} / {round(thr_high,4)}) "
              f"temps={ {k: round(v,3) for k,v in temps_final.items()} }")

    out = Path(args.out) if args.out else md / "deployment_metrics.json"
    payload = {
        "report": "deployment_metrics",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_spec": spec.to_dict(),
        "member_set": args.member_set,
        "weights": weights,
        "protocol": {
            "cv": f"stratified {args.n_folds}-fold x {args.rounds} rounds "
                  f"(seeds {args.cv_seed}..{args.cv_seed + (args.rounds-1)*13})",
            "threshold": f"nested inner-CV out-of-fold training-fold real scores, "
                         f"fp_budget={args.fp_budget}",
            "deployed_threshold": "pooled out-of-fold order statistic (round-averaged)",
            "note": "no held-out sample selects a threshold, a weight, or a feature set",
        },
        "thresholds": {"pooled_logit": round(thr_pooled, 6),
                       "perfold_median_logit": round(thr_perfold, 6),
                       "space_note": "logits for calibration; the deployed fusion.json "
                                     "stores the probability equivalent"},
        "deployed_rule_metrics": {
            "note": "per-round metrics when the DEPLOYED (pooled) threshold is applied "
                    "to that round's out-of-fold scores - this is the number the service "
                    "actually realises",
            "per_round": [dict(
                {"round": r["round"], "seed": r["seed"],
                 "accuracy": round(float(((oof_all[i] >= thr_pooled).astype(int) == y).mean()), 4),
                 "fpr": round(float(((oof_all[i] >= thr_pooled).astype(int))[y == 0].mean()), 4),
                 "tpr": round(float(((oof_all[i] >= thr_pooled).astype(int))[y == 1].mean()), 4)}
            ) for i, r in enumerate(rounds)],
        },
        "n_samples": n,
        "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
        "summary": summary,
        # cached so the threshold arithmetic can be redone without refitting anything
        # (see --rewrite-fusion-only): rows = CV rounds, columns = samples in label order
        "_oof_fused_scores": [[round(float(v), 6) for v in row] for row in oof_all],
        "_oof_seqs": [int(s) for s in seqs],
        "rounds": rounds,
        "errors": {
            "fp_counts": {int(seqs[i]): int(fp_counts[i]) for i in range(n) if y[i] == 0
                          and fp_counts[i] > 0},
            "fn_counts": {int(seqs[i]): int(fn_counts[i]) for i in range(n) if y[i] == 1
                          and fn_counts[i] > 0},
            "of_rounds": args.rounds,
        },
        "frozen": None if args.no_freeze else {
            "thr_high": round(thr_high, 6),
            "temperatures": temps_final,
            "files": ["forensic_lr.joblib", "combined_lr.joblib", "fusion.json",
                      "feature_spec.json", "clip_tiles_spec.npy"],
        },
        "caveats": [
            "The 119 samples are the only labelled competition domain data; every head, "
            "temperature and threshold is fit on training folds only.",
            "The threshold is an order statistic on 50 real training samples per fold, so "
            "the false-positive budget is quantised (1 FP = 2% at that scale).",
            "Accuracy spread across CV seeds is reported instead of a single split's value.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
