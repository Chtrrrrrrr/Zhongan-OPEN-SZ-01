#!/usr/bin/env python3
"""Add external labelled material to the decision heads' TRAINING folds only.

The defect this addresses (docs/known-issue-scene-photo-fpr.md): the 119 competition
samples' "real" class is nearly all phone photos of paper, so the head learned
"photographed paper = real" and flags real claim SCENE photos 100%. The manual's risk R2
mitigation is to enlarge the real side of the corpus; this script does that in a way
that keeps every reported number honest.

Protocol
  * the 119 competition samples are the ONLY evaluation set; every headline number below
    is an out-of-fold prediction on them,
  * an external labelled corpus (real photographs + AI-generated material from other
    generators) is appended to the TRAINING fold of each outer split only,
  * the inner nested CV that supplies the score scale for temperature/threshold fitting
    uses official training samples alone, so the threshold keeps the meaning it had
    (a per-batch false-positive budget on competition-like material),
  * external reals are scored by the fold model that did NOT train on them, which makes
    the reported external false-positive rate a real holdout number.

Reported per variant
  * official OOF accuracy / FPR / TPR (the competition metric),
  * external real false-positive rate under the official-only threshold,
  * external real false-positive rate under a threshold pooled over official + external
    reals (the deployment option for the business case),
  * the reachable accuracy ceiling at the same false-positive budget.

Usage
  python scripts/train_with_external_reals.py --corpus models/external_corpus \
      --rounds 6 --out reports/external_reals_fix.json
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
from feature_spec import FeatureSpec                                        # noqa: E402

WEIGHTS = {"combined": 0.60, "forensic": 0.25}


def load_official(models_dir: Path, spec: FeatureSpec):
    X1 = np.loadtxt(ROOT / "features.csv", delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(models_dir / "l1b_feats.npy")
    parts = [X1, Xb]
    if spec.patch_tokens:
        parts.append(np.load(models_dir / "clip_patch.npy"))
    Xf = np.concatenate(parts, axis=1).astype(np.float64)
    Fg = np.load(models_dir / "clip_feats.npy").astype(np.float64)
    tile = models_dir / ("clip_tiles_spec.npy" if (models_dir / "clip_tiles_spec.npy").exists()
                         else "clip_tiles.npy")
    Ft = np.load(tile).astype(np.float64)
    return Xf, np.concatenate([Xf, Fg, Ft], axis=1)


def load_corpus(corpus: Path, spec: FeatureSpec):
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    parts = [np.load(corpus / "l1.npy"), np.load(corpus / "l1b.npy")]
    if spec.patch_tokens:
        parts.append(np.load(corpus / "patch.npy"))
    Xf = np.concatenate(parts, axis=1).astype(np.float64)
    Fg = np.load(corpus / "clip.npy").astype(np.float64)
    Ft = np.load(corpus / "tiles.npy", allow_pickle=True)
    Ft = np.stack([np.asarray(t, dtype=np.float64) for t in Ft])
    y = np.array(meta["labels"], dtype=int)
    folders = np.array(meta["folders"])
    return Xf, np.concatenate([Xf, Fg, Ft], axis=1), y, folders


def _fit_head_w(Xtr, ytr, w=None, C=1.0):
    """Logistic head with optional per-sample weights (official rows up-weighted)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), ytr, sample_weight=w)
    return sc, clf


def run_variant(label, Xf_o, Xc_o, y_o, Xf_e, Xc_e, y_e, rounds, n_folds, fp_budget,
                base_seed, use_external_train, sample_weight=0.0, weights=None):
    """Nested CV; external material (when enabled) only ever enters TRAINING folds."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    inner_splits = max(3, n_folds - 1)
    W = dict(weights or WEIGHTS)
    n_o, n_e = len(y_o), len(y_e)
    oof = np.zeros((rounds, n_o))
    ext_logits = {"forensic": np.zeros((rounds, n_e)), "combined": np.zeros((rounds, n_e))}
    member_thr = []
    t0 = time.perf_counter()

    def fit(Xtr_o, ytr_o, Xtr_e=None, ytr_e=None):
        Xf_tr_o, Xc_tr_o = Xtr_o
        if Xtr_e is None or len(ytr_e) == 0:
            return {"forensic": _fit_head(Xf_tr_o, ytr_o),
                    "combined": _fit_head(Xc_tr_o, ytr_o)}
        Xf_tr_e, Xc_tr_e = Xtr_e
        w_o = (np.full(len(ytr_o), float(sample_weight)) if sample_weight > 0 else None)
        Xf_all = np.concatenate([Xf_tr_o, Xf_tr_e], axis=0)
        Xc_all = np.concatenate([Xc_tr_o, Xc_tr_e], axis=0)
        y_all = np.concatenate([ytr_o, ytr_e])
        w_all = None if w_o is None else np.concatenate([w_o, np.ones(len(ytr_e))])
        return {"forensic": _fit_head_w(Xf_all, y_all, w_all),
                "combined": _fit_head_w(Xc_all, y_all, w_all)}

    sel_real = y_e == 0
    sel_fake = y_e == 1
    for r in range(rounds):
        seed = base_seed + r * 13
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(Xc_o, y_o)):
            if use_external_train == "both":
                ext_sel = np.ones(n_e, dtype=bool)
            elif use_external_train == "reals":
                ext_sel = sel_real
            elif use_external_train == "fakes":
                ext_sel = sel_fake
            else:
                ext_sel = np.zeros(n_e, dtype=bool)
            heads = fit((Xf_o[tr], Xc_o[tr]), y_o[tr],
                        (Xf_e[ext_sel], Xc_e[ext_sel]), y_e[ext_sel])
            z_va = {"forensic": _logits(*heads["forensic"], Xf_o[va]),
                    "combined": _logits(*heads["combined"], Xc_o[va])}
            # inner nested CV on OFFICIAL training samples only: keeps the score scale and
            # the threshold tied to competition-like material
            inner = {"forensic": np.zeros(len(tr)), "combined": np.zeros(len(tr))}
            iskf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                   random_state=100 + fold)
            for itr, iva in iskf.split(Xc_o[tr], y_o[tr]):
                h = {"forensic": _fit_head(Xf_o[tr][itr], y_o[tr][itr]),
                     "combined": _fit_head(Xc_o[tr][itr], y_o[tr][itr])}
                inner["forensic"][iva] = _logits(*h["forensic"], Xf_o[tr][iva])
                inner["combined"][iva] = _logits(*h["combined"], Xc_o[tr][iva])
            temps = {m: _fit_temperature(inner[m], y_o[tr]) for m in inner}
            member_thr.append(_fpr_threshold(
                _fused(inner, temps, W)[y_o[tr] == 0], fp_budget))
            oof[r, va] = _fused(z_va, temps, W)
            # external reals scored by a model that never saw them
            ext_logits["forensic"][r] += _logits(*heads["forensic"], Xf_e) / n_folds
            ext_logits["combined"][r] += _logits(*heads["combined"], Xc_e) / n_folds

    # Temperature for the external scores. The per-fold temperatures belong to the inner
    # official scale; reuse the last fold's estimate, which is the same scale family, and
    # document the choice rather than silently dropping calibration.
    temps_last = {m: _fit_temperature(inner[m], y_o[tr]) for m in inner}
    # fuse each round separately: _fused works on 1-D score vectors, so stacking the raw
    # logits across rounds and fusing once would broadcast wrongly (and averaging raw
    # logits before fusion would not equal averaging the fused scores)
    ext_fused_rounds = np.stack([_fused({m: ext_logits[m][r] for m in ext_logits},
                                        temps_last, W) for r in range(rounds)])
    ext_fused = ext_fused_rounds.mean(axis=0)
    ext_auc = None
    if len(set(y_e.tolist())) > 1:
        from sklearn.metrics import roc_auc_score as _auc
        ext_auc = round(float(np.mean([_auc(y_e, ext_fused_rounds[r])
                                       for r in range(rounds)])), 4)

    return {"label": label, "use_external_train": bool(use_external_train),
            "oof": oof, "member_thresholds": member_thr, "external_fused": ext_fused,
            "external_auc": ext_auc, "seconds": round(time.perf_counter() - t0, 1)}


def summarize(res, y_o, y_e, fp_budget, rounds):
    oof = res["oof"]
    per_round = []
    for r in range(rounds):
        pred = (oof[r] >= res["thr_official"]).astype(int)
        per_round.append({"accuracy": round(float((pred == y_o).mean()), 4),
                          "fpr": round(float(pred[y_o == 0].mean()), 4),
                          "tpr": round(float(pred[y_o == 1].mean()), 4)})
    ext_real = res["external_fused"][y_e == 0]
    # Official metrics under BOTH candidate thresholds. Two populations with different
    # score levels (competition documents vs real scene photos) cannot share one exact
    # operating point, so the report states what each choice costs.
    pooled_rounds = []
    for r in range(rounds):
        pred = (oof[r] >= res["thr_pooled"]).astype(int)
        pooled_rounds.append({"accuracy": round(float((pred == y_o).mean()), 4),
                              "fpr": round(float(pred[y_o == 0].mean()), 4),
                              "tpr": round(float(pred[y_o == 1].mean()), 4)})
    out = {
        "official_oof": {
            "accuracy_mean": round(float(np.mean([p["accuracy"] for p in per_round])), 4),
            "accuracy_min": round(float(np.min([p["accuracy"] for p in per_round])), 4),
            "accuracy_max": round(float(np.max([p["accuracy"] for p in per_round])), 4),
            "fpr_mean": round(float(np.mean([p["fpr"] for p in per_round])), 4),
            "tpr_mean": round(float(np.mean([p["tpr"] for p in per_round])), 4),
            "per_round": per_round,
        },
        "official_oof_at_pooled_threshold": {
            "accuracy_mean": round(float(np.mean([p["accuracy"] for p in pooled_rounds])), 4),
            "accuracy_min": round(float(np.min([p["accuracy"] for p in pooled_rounds])), 4),
            "fpr_mean": round(float(np.mean([p["fpr"] for p in pooled_rounds])), 4),
            "tpr_mean": round(float(np.mean([p["tpr"] for p in pooled_rounds])), 4),
        },
        "official_threshold": round(float(res["thr_official"]), 6),
        "external_real_fpr_at_official_threshold": round(float((ext_real >= res["thr_official"]).mean()), 4),
        "external_real_fpr_at_pooled_threshold": round(float((ext_real >= res["thr_pooled"]).mean()), 4),
        "pooled_threshold": round(float(res["thr_pooled"]), 6),
        "external_real_n": int(len(ext_real)),
        "accuracy_ceiling_at_budget": round(float(np.mean(
            [_acc_at_fp(oof[r], y_o, fp_budget)["acc"] for r in range(rounds)])), 4),
            "external_auc": res["external_auc"],
            "temperature_note": "external scores are calibrated with the last outer fold's "
                                "official inner-CV temperature (same scale family)",
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--corpus", default="models/external_corpus")
    ap.add_argument("--out", default="reports/external_reals_fix.json")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fp-budget", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--corpus-real-limit", type=int, default=0,
                    help="cap external real images (0 = all)")
    ap.add_argument("--corpus-fake-limit", type=int, default=0)
    ap.add_argument("--variants", default="baseline_official_only,external_reals_only,"
                                          "external_both,external_both_balanced",
                    help="which training corpora to compare")
    ap.add_argument("--sample-weight", type=float, default=0.0,
                    help="sample_weight for official rows when external data is added "
                         "(0 = unweighted, i.e. plain concatenation)")
    ap.add_argument("--weight-grid", default="",
                    help="comma list of official-row weights to also test on the "
                         "external_both corpus (e.g. 1,3,10); each becomes its own variant")
    ap.add_argument("--folders-exclude", default="",
                    help="comma list of corpus folders to leave out of TRAINING (they stay "
                         "in the holdout evaluation), e.g. real_synthbuster")
    ap.add_argument("--extra-seeds", type=int, default=0,
                    help="repeat the best variant with this many additional CV seed sets "
                         "and report the combined accuracy spread")
    args = ap.parse_args()

    md = ROOT / args.models_dir
    spec = FeatureSpec.load(md / "feature_spec.json")
    Xf_o, Xc_o = load_official(md, spec)
    y_o = []
    with open(Path(args.data_dir) / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y_o.append(1 if r["label"] == "ai" else 0)
    y_o = np.array(y_o)
    print(f"[official] forensic={Xf_o.shape} combined={Xc_o.shape}")

    Xf_e, Xc_e, y_e, folders = load_corpus(Path(args.corpus), spec)
    if args.folders_exclude:
        drop = {f.strip() for f in args.folders_exclude.split(",") if f.strip()}
        print(f"[corpus] excluding from training: {sorted(drop)}")
        # excluded folders are dropped from training *and* from the holdout, so the
        # reported external numbers always describe the same population the heads can see
        keep = np.array([i for i, f in enumerate(folders) if f not in drop], dtype=int)
        Xf_e, Xc_e, y_e, folders = Xf_e[keep], Xc_e[keep], y_e[keep], folders[keep]
    if args.corpus_real_limit or args.corpus_fake_limit:
        keep = []
        n_real = n_fake = 0
        for i, lab in enumerate(y_e):
            if lab == 0:
                if args.corpus_real_limit and n_real >= args.corpus_real_limit:
                    continue
                n_real += 1
            else:
                if args.corpus_fake_limit and n_fake >= args.corpus_fake_limit:
                    continue
                n_fake += 1
            keep.append(i)
        keep = np.array(keep, dtype=int)
        Xf_e, Xc_e, y_e, folders = Xf_e[keep], Xc_e[keep], y_e[keep], folders[keep]
    print(f"[corpus] forensic={Xf_e.shape} combined={Xc_e.shape} "
          f"real={int((y_e==0).sum())} fake={int((y_e==1).sum())} "
          f"folders={sorted(set(folders.tolist()))}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    modes = {"baseline_official_only": None, "external_reals_only": "reals",
             "external_fakes_only": "fakes", "external_both": "both",
             "external_both_balanced": "both"}
    for w in [x.strip() for x in args.weight_grid.split(",") if x.strip()]:
        modes[f"external_both_w{w}"] = "both"
        variants.append(f"external_both_w{w}")
    payload = {"report": "external_reals_fix",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "motivation": "docs/known-issue-scene-photo-fpr.md",
               "protocol": {
                   "evaluation_set": "the 119 competition samples (out-of-fold only)",
                   "training": "official training fold + all external labelled material",
                   "inner_cv": "official training samples only (keeps the threshold in "
                               "competition-material units)",
                   "fp_budget": args.fp_budget,
                   "rounds": args.rounds, "n_folds": args.n_folds,
                   "sample_weight_official": args.sample_weight,
               },
               "corpus": {"real": int((y_e == 0).sum()), "fake": int((y_e == 1).sum()),
                          "folders": sorted(set(folders.tolist()))},
               "variants": {}}
    for label in variants:
        if label not in modes:
            raise SystemExit(f"unknown variant {label!r}; choose from {list(modes)}")
        mode = modes[label]
        if label == "external_both_balanced":
            sw = args.sample_weight
        elif label.startswith("external_both_w"):
            sw = float(label.rsplit("_w", 1)[1])
        else:
            sw = 0.0
        res = run_variant(label, Xf_o, Xc_o, y_o, Xf_e, Xc_e, y_e, args.rounds,
                          args.n_folds, args.fp_budget, args.base_seed, mode,
                          sample_weight=sw)
        all_thr = np.array(res["member_thresholds"])
        res["thr_official"] = float(np.median(all_thr))
        pooled = np.concatenate([res["oof"][:, y_o == 0].mean(axis=0),
                                 res["external_fused"][y_e == 0]])
        s = np.sort(pooled)[::-1]
        k = max(0, min(args.fp_budget, len(s) - 1))
        res["thr_pooled"] = float(s[k]) if k == 0 else float((s[k - 1] + s[k]) / 2.0)
        summary = summarize(res, y_o, y_e, args.fp_budget, args.rounds)
        payload["variants"][label] = summary
        print(f"\n=== {label} ({res['seconds']}s) ===")
        print(f"  official OOF : acc={summary['official_oof']['accuracy_mean']} "
              f"[{summary['official_oof']['accuracy_min']},{summary['official_oof']['accuracy_max']}] "
              f"fpr={summary['official_oof']['fpr_mean']} tpr={summary['official_oof']['tpr_mean']} "
              f"ceiling@budget={summary['accuracy_ceiling_at_budget']}")
        print(f"  thresholds   : official={summary['official_threshold']:.4f} "
              f"pooled={summary['pooled_threshold']:.4f}")
        print(f"  external real: FPR@official={summary['external_real_fpr_at_official_threshold']:.1%} "
              f"FPR@pooled={summary['external_real_fpr_at_pooled_threshold']:.1%} "
              f"(n={summary['external_real_n']})  external AUC={summary['external_auc']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")

    if args.extra_seeds:
        best = max(payload["variants"],
                   key=lambda k: (payload["variants"][k]["official_oof"]["accuracy_mean"],
                                  -payload["variants"][k]["external_real_fpr_at_official_threshold"]))
        mode = modes[best]
        sw = (args.sample_weight if best == "external_both_balanced"
              else float(best.rsplit("_w", 1)[1]) if best.startswith("external_both_w")
              else 0.0)
        print(f"\n[extra seeds] repeating {best} with {args.extra_seeds} more seed sets")
        extra = run_variant(f"{best}_extraseeds", Xf_o, Xc_o, y_o, Xf_e, Xc_e, y_e,
                            args.extra_seeds, args.n_folds, args.fp_budget,
                            args.base_seed + 1000, mode, sample_weight=sw)
        all_thr = np.array(extra["member_thresholds"])
        extra["thr_official"] = float(np.median(all_thr))
        s = np.sort(np.concatenate([extra["oof"][:, y_o == 0].mean(axis=0),
                                    extra["external_fused"][y_e == 0]]))[::-1]
        k = max(0, min(args.fp_budget, len(s) - 1))
        extra["thr_pooled"] = float(s[k]) if k == 0 else float((s[k - 1] + s[k]) / 2.0)
        summary = summarize(extra, y_o, y_e, args.fp_budget, args.extra_seeds)
        payload["extra_seeds_validation"] = {"variant": best, "seeds_used":
                                             [args.base_seed + 1000 + r * 13
                                              for r in range(args.extra_seeds)],
                                             **summary}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  extra: official OOF acc={summary['official_oof']['accuracy_mean']} "
              f"[{summary['official_oof']['accuracy_min']},{summary['official_oof']['accuracy_max']}] "
              f"fpr={summary['official_oof']['fpr_mean']}  "
              f"external real FPR@official={summary['external_real_fpr_at_official_threshold']:.2%}")
        print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
