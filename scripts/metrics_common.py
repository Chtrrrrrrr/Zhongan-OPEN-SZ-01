#!/usr/bin/env python3
"""Shared out-of-fold (OOF) evaluation harness for the OPEN-SZ-01 metrics pack.

This module is imported by:

  scripts/calibration_analysis.py   (manual ch.13.1 / 15.1 - ECE, temperature scaling,
                                     FPR@TPR=95%, TPR@FPR=1%)
  scripts/member_analysis.py        (manual ch.12.1 - score correlation, decision
                                     disagreement, FP/FN set overlap, leave-one-out)
  scripts/threshold_stability.py    (manual ch.13.1 - bootstrap threshold CI + sensitivity)
  scripts/stratified_analysis.py    (manual ch.15.1 - per-group diagnostic breakdown)

It exists so that every one of those reports is computed from the *same* OOF score
matrix, i.e. the numbers cannot drift between documents. The protocol reproduces
scripts/evaluate_system.py exactly:

  * stratified 5-fold (StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
  * every trainable head (scaler + LogisticRegression C=1.0, max_iter=5000) is fit on
    the training fold only
  * an inner stratified 4-fold (random_state=100+fold) produces unbiased in-training
    logits used to fit the per-member temperature scalers
  * temperatures fitted on the inner OOF logits are applied to the held-out fold, so
    the cached "calibrated" probability of every sample is out-of-sample

Members (the five score-producing routes considered by the manual, ch.9/12.1):

  forensic        LR on [L1(47) (+) L1b(20)]                     trainable, in the deployed fusion
  combined        LR on [L1(47) (+) L1b(20) (+) CLIP(1024) (+) tiles(3072)]
                  trainable, in the deployed fusion
  clip_domain     LR on CLIP global features (1024)              trainable, ablation channel
  clip_external   frozen linear probe trained on the 18-generator AIGC benchmark
                  (models/clip_probe_norm.pt), applied to codec-normalized CLIP features;
                  never trained on the 119 samples
  npr             frozen NPR ResNet-18 (models/npr.pt) logits, precomputed by
                  scripts/extract_npr_scores.py; trained on the external benchmark shards

Weight tables
-------------
DEPLOYED_WEIGHTS is the frozen production table written into models/fusion.json.
EXTENDED_WEIGHTS is an *analysis-only* table that admits the two ablation channels
(clip_domain, npr) so the manual's leave-one-out experiment can be run over all five
routes; it is reported explicitly as an analysis configuration and never used by the
deployed pipeline. EQUAL_WEIGHTS is the weight-free control (equal weights over the
five routes) used to check that the leave-one-out ranking is not an artefact of the
chosen table.

Nothing here writes into reports/ - each caller owns its own evidence file.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # repo/
WS = ROOT.parent                                        # workspace root
sys.path.insert(0, str(ROOT / "src"))

from fusion import TemperatureScaler  # noqa: E402  (read-only reuse of the shipped scaler)

# ---------------------------------------------------------------- constants
N_FOLDS_DEFAULT = 5
OUTER_SEED = 42                     # identical to evaluate_system.py
INNER_SEED_BASE = 100               # identical to evaluate_system.py
EXT_MIN_AUC = 0.65                  # clip_external admission rule (evaluate_system.py)

FP_BUDGET_DEFAULT = 2               # floor(0.03 * 70) under the stated 7:3 evaluation ratio

# Nominal weight table from the protocol description (evaluate_system.py WEIGHTS).
# The *frozen* deployed table is read from models/fusion.json at run time: clip_external
# failed the training-fold AUC admission rule in 2 of 5 folds, so the shipped table
# contains only combined + forensic (see reports/metrics.json -> final_members).
DEPLOYED_WEIGHTS = {"combined": 0.60, "forensic": 0.25, "clip_external": 0.15}

# Analysis-only: adds the two ablation channels so all five routes can be ranked by
# leave-one-out. The deployed three-route table is always reported next to it.
EXTENDED_WEIGHTS = {"combined": 0.40, "forensic": 0.20, "clip_domain": 0.15,
                    "clip_external": 0.15, "npr": 0.10}

EQUAL_WEIGHTS = {m: 1.0 for m in ("forensic", "combined", "clip_domain",
                                  "clip_external", "npr")}

# Named weight tables used by the analysis scripts (the deployed per-fold path is
# handled separately because its member set varies by fold).
WEIGHT_TABLES = {
    "nominal_3route": DEPLOYED_WEIGHTS,
    "extended_5route": EXTENDED_WEIGHTS,
    "equal_5route": EQUAL_WEIGHTS,
}

MEMBER_ORDER = ["forensic", "combined", "clip_domain", "clip_external", "npr"]

MEMBER_DESCRIPTION = {
    "forensic": "LR on L1(47)+L1b(20) forensic/domain features (trainable, deployed)",
    "combined": "LR on L1(47)+L1b(20)+CLIP global(1024)+CLIP tiles(3072) (trainable, deployed)",
    "clip_domain": "LR on CLIP global features(1024) only (trainable, ablation channel)",
    "clip_external": "frozen CLIP probe trained on the 18-generator benchmark "
                     "(never trained on the 119)",
    "npr": "frozen NPR ResNet-18 logits on external benchmark shards (never trained on the 119)",
}


# ---------------------------------------------------------------- basic IO
def load_labels(data_dir: Path):
    seqs, y = [], []
    with open(Path(data_dir) / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            y.append(1 if r["label"] == "ai" else 0)
    return np.array(seqs), np.array(y)


def load_external_probe(pt: Path):
    """Same reader as evaluate_system.py: {state_dict, scaler:{mean,scale}}."""
    import torch
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    w = sd["weight"].numpy().reshape(-1)
    b = float(sd["bias"].numpy().reshape(-1)[0])
    sc = ck.get("scaler") or {}
    return (w, b,
            np.array(sc.get("mean", np.zeros_like(w)), dtype=np.float64),
            np.array(sc.get("scale", np.ones_like(w)), dtype=np.float64),
            {k: v for k, v in ck.items() if k not in ("state_dict", "scaler")})


def logit(p):
    """Logit with the same 1e-6 clipping scripts/evaluate_system.py uses."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- metrics
def tpr_at_fpr(y, score, target=0.01):
    """TPR at a target FPR.

    Two numbers are returned on purpose (manual ch.13.1 表述纪律: a quantile is not
    the same thing as an FPR):

      interp   - linear interpolation of the ROC curve at the exact target FPR
      empirical- best TPR whose measured FPR is <= target (counts are integral, so with
                 n_real=63 the smallest non-zero FPR is 1/63 = 1.59%)
    """
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    fpr, tpr, _ = roc_curve(y, np.asarray(score, dtype=np.float64))
    interp = float(np.interp(target, fpr, tpr))
    ok = fpr <= target + 1e-12
    empirical = float(tpr[ok].max()) if ok.any() else 0.0
    return {"target_fpr": target, "tpr_interp": round(interp, 4),
            "tpr_empirical": round(empirical, 4),
            "achieved_fpr_empirical": round(float(fpr[ok].max()) if ok.any() else 0.0, 4)}


def fpr_at_tpr(y, score, target=0.95):
    """Smallest measured FPR whose TPR reaches `target` (manual 15.1 guardrail)."""
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    fpr, tpr, _ = roc_curve(y, np.asarray(score, dtype=np.float64))
    ok = tpr >= target - 1e-12
    if not ok.any():
        return {"target_tpr": target, "fpr": None, "reachable": False}
    return {"target_tpr": target, "fpr": round(float(fpr[ok].min()), 4),
            "reachable": True,
            "tpr_at_that_point": round(float(tpr[ok].min()), 4)}


def auc(y, score):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(np.asarray(y), np.asarray(score, dtype=np.float64))), 4)


def ece(y, p, n_bins=10, strategy="uniform"):
    """Expected / maximum calibration error + reliability-diagram bins.

    ECE = sum_b (n_b / N) * |mean(p in b) - mean(y in b)|, MCE = max_b |gap|.
    `strategy="uniform"` uses equal-width bins on [0,1] (the standard definition);
    `"quantile"` uses equal-mass bins, which is more stable when the score
    distribution is heavily skewed - both are reported.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    n = len(y)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "uniform" or len(edges) < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.asarray(edges, dtype=np.float64)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    bins, e, mce = [], 0.0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            bins.append({"index": b, "lo": round(float(edges[b]), 4),
                         "hi": round(float(edges[b + 1]), 4), "count": 0,
                         "mean_confidence": None, "empirical_positive_rate": None,
                         "gap": None})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        gap = abs(conf - acc)
        e += (m.sum() / n) * gap
        mce = max(mce, gap)
        bins.append({"index": b, "lo": round(float(edges[b]), 4),
                     "hi": round(float(edges[b + 1]), 4), "count": int(m.sum()),
                     "mean_confidence": round(conf, 4),
                     "empirical_positive_rate": round(acc, 4),
                     "gap": round(gap, 4)})
    return {"n_bins": len(edges) - 1, "strategy": strategy, "ece": round(float(e), 4),
            "mce": round(float(mce), 4), "n": n, "bins": bins}


def brier(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return round(float(np.mean((p - np.asarray(y, dtype=np.float64)) ** 2)), 4)


def log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)


def confusion(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_at(y, pred):
    c = confusion(y, pred)
    n_real = max(1, int((np.asarray(y) == 0).sum()))
    n_ai = max(1, int((np.asarray(y) == 1).sum()))
    return {"accuracy": round((c["tp"] + c["tn"]) / max(1, len(y)), 4),
            "fpr": round(c["fp"] / n_real, 4),
            "tpr": round(c["tp"] / n_ai, 4),
            "precision": round(c["tp"] / max(1, c["tp"] + c["fp"]), 4),
            "confusion": c}


def fpr_capped_threshold(real_scores, target_fpr=None, fp_budget=None):
    """Order statistic on a pure-real calibration set (identical to evaluate_system.py).

    Admits at most `fp_budget` real samples (or floor(target_fpr * n)) at or above the
    threshold, so the returned threshold is the midpoint between the last admissible and
    the first inadmissible real score.
    """
    s = np.sort(np.asarray(real_scores, dtype=np.float64))[::-1]
    n = len(s)
    if n == 0:
        return 0.5
    allowed = int(np.floor(target_fpr * n)) if fp_budget is None else int(fp_budget)
    allowed = max(0, min(allowed, n - 1))
    if allowed <= 0:
        return float(s[0]) + 1e-9
    if allowed >= n:
        return float(s[-1]) - 1e-9
    return float((s[allowed - 1] + s[allowed]) / 2.0)


# ---------------------------------------------------------------- fusion helpers
def fuse_probabilities(calibrated: dict, weights: dict, members=None):
    """Weighted logit-mean fusion over a subset of members -> array of probabilities.

    Mirrors src/fusion.py `FusionEngine.fuse` (calibrated probability -> logit ->
    weighted mean -> sigmoid) but vectorised over all samples, which is required for
    the leave-one-out and threshold experiments.
    """
    members = list(members or [m for m in MEMBER_ORDER if m in calibrated])
    members = [m for m in members if m in calibrated and weights.get(m, 0.0) > 0]
    if not members:
        raise ValueError("no members left to fuse")
    w = np.array([weights[m] for m in members], dtype=np.float64)
    w = w / w.sum()
    lg = np.stack([logit(calibrated[m]) for m in members], axis=0)
    return sigmoid((w[:, None] * lg).sum(axis=0)), members


# ---------------------------------------------------------------- the bundle
def _fit_head(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(X), y)
    return sc, clf


def _head_logits(sc, clf, X):
    return logit(clf.predict_proba(sc.transform(X))[:, 1])


def build_bundle(workspace: Path = WS, features: str = "features.csv",
                 n_folds: int = N_FOLDS_DEFAULT, verbose: bool = True,
                 data_dir: str = "data", models_dir: str = "models",
                 clip_probe: str = "models/clip_probe_norm.pt") -> dict:
    """Compute the OOF score matrix for all five routes.

    Returns a JSON-serialisable dict (lists, not arrays) so it can be cached verbatim.
    """
    from sklearn.model_selection import StratifiedKFold

    ws = Path(workspace)
    md = ws / models_dir

    def say(*a):
        if verbose:
            print(*a, flush=True)

    seqs, y = load_labels(ws / data_dir)
    n = len(y)
    say(f"[oof] data={ws / data_dir} n={n} ai={int((y == 1).sum())} real={int((y == 0).sum())}")

    X1 = np.loadtxt(ws / features, delimiter=",", skiprows=1, usecols=range(2, 49))
    Xb = np.load(md / "l1b_feats_119.npy").astype(np.float64)
    Fg = np.load(md / "clip_feats_119.npy").astype(np.float64)
    Ft = np.load(md / "clip_tiles_119.npy").astype(np.float64)
    for name, a in (("l1_47", X1), ("l1b_20", Xb), ("clip_global", Fg), ("clip_tiles", Ft)):
        if a.shape[0] != n:
            raise SystemExit(f"[shape] {name}: {a.shape[0]} rows != {n}")
    XF = np.concatenate([X1, Xb], axis=1)                       # forensic
    XC = np.concatenate([X1, Xb, Fg, Ft], axis=1)               # combined
    XD = Fg                                                     # clip_domain

    probe = ws / clip_probe
    if not probe.exists():
        raise SystemExit(f"external probe not found: {probe}")
    w, b, mean, scale, probe_meta = load_external_probe(probe)
    fgn_path = md / "clip_feats_norm_119.npy"
    Fgn = np.load(fgn_path).astype(np.float64) if ("norm" in probe.name and fgn_path.exists()) else Fg
    ext = (((Fgn - mean) / scale) @ w + b)
    say(f"[oof] clip_external source={fgn_path.name if 'norm' in probe.name else 'clip_feats_119.npy'} "
        f"AUC={auc(y, ext)}")

    npr_path = md / "npr_scores_119.npy"
    if not npr_path.exists():
        raise SystemExit(f"NPR scores not found: {npr_path} (run scripts/extract_npr_scores.py)")
    npr = np.load(npr_path).astype(np.float64).reshape(-1)
    if npr.shape[0] != n:
        raise SystemExit(f"[shape] npr: {npr.shape[0]} != {n}")
    say(f"[oof] npr AUC={auc(y, npr)}")

    logits = {m: np.zeros(n) for m in MEMBER_ORDER}
    inner = {m: np.zeros(n) for m in MEMBER_ORDER}
    temperatures = {m: [] for m in MEMBER_ORDER}
    calibrated = {m: np.zeros(n) for m in MEMBER_ORDER}
    # the *deployed* operating point: per-fold member admission + per-fold thresholds,
    # reproducing scripts/evaluate_system.py fold for fold
    deployed_fused = np.zeros(n)
    deployed_pred = np.zeros(n, dtype=int)
    deployed_verdict = np.array(["review"] * n, dtype=object)
    folds = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=OUTER_SEED)
    inner_splits = max(3, n_folds - 1)
    for fold, (tr, va) in enumerate(skf.split(XF, y)):
        scf, clff = _fit_head(XF[tr], y[tr])
        scc, clfc = _fit_head(XC[tr], y[tr])
        scd, clfd = _fit_head(XD[tr], y[tr])
        logits["forensic"][va] = _head_logits(scf, clff, XF[va])
        logits["combined"][va] = _head_logits(scc, clfc, XC[va])
        logits["clip_domain"][va] = _head_logits(scd, clfd, XD[va])
        logits["clip_external"][va] = ext[va]
        logits["npr"][va] = npr[va]

        inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                    random_state=INNER_SEED_BASE + fold)
        inner["clip_external"][tr] = ext[tr]
        inner["npr"][tr] = npr[tr]
        for itr, iva in inner_skf.split(XF[tr], y[tr]):
            s1, c1 = _fit_head(XF[tr][itr], y[tr][itr])
            inner["forensic"][tr[iva]] = _head_logits(s1, c1, XF[tr][iva])
            s2, c2 = _fit_head(XC[tr][itr], y[tr][itr])
            inner["combined"][tr[iva]] = _head_logits(s2, c2, XC[tr][iva])
            s3, c3 = _fit_head(XD[tr][itr], y[tr][itr])
            inner["clip_domain"][tr[iva]] = _head_logits(s3, c3, XD[tr][iva])

        # temperature per member, fitted on the inner (unbiased-scale) training logits
        for m in MEMBER_ORDER:
            sc = TemperatureScaler().fit(inner[m][tr], y[tr])
            temperatures[m].append(round(float(sc.temperature), 4))
            calibrated[m][va] = sigmoid(logits[m][va] / temperatures[m][-1])

        # ---- deployed fold: clip_external is admitted only if its training-fold AUC
        # >= EXT_MIN_AUC (evaluate_system.py), weights renormalised over the admitted set
        ext_auc_tr = float(auc(y[tr], ext[tr]))
        members = ["combined", "forensic"]
        if ext_auc_tr >= EXT_MIN_AUC:
            members.append("clip_external")
        wts = {m: DEPLOYED_WEIGHTS[m] for m in members}

        # calibration set = training-fold REAL samples on the same unbiased score scale
        inner_cal = {m: sigmoid(inner[m][tr] / temperatures[m][-1]) for m in members}
        tr_fused, _ = fuse_probabilities(inner_cal, wts, members)
        real_scores = tr_fused[y[tr] == 0]
        thr_high = fpr_capped_threshold(real_scores, 0.03, FP_BUDGET_DEFAULT)
        thr_low = float(np.median(real_scores))
        if thr_low >= thr_high:
            thr_low = float(np.percentile(real_scores, 25))

        va_fused, _ = fuse_probabilities({m: calibrated[m][va] for m in members}, wts, members)
        deployed_fused[va] = va_fused
        deployed_pred[va] = (va_fused >= thr_high).astype(int)
        for k, i in enumerate(va):
            deployed_verdict[i] = ("suspect" if deployed_pred[i] == 1
                                   else "normal" if va_fused[k] <= thr_low else "review")
        folds.append({"fold": fold + 1, "n_train": int(len(tr)), "n_holdout": int(len(va)),
                      "holdout_seqs": [int(seqs[i]) for i in va],
                      "deployed_members": members,
                      "clip_external_train_auc": round(ext_auc_tr, 4),
                      "thr_low": round(thr_low, 4), "thr_high": round(thr_high, 4),
                      "temperatures": {m: temperatures[m][-1] for m in MEMBER_ORDER}})
        say(f"[oof] fold {fold + 1}/{n_folds} members={members} thr_high={thr_high:.4f} "
            f"temperatures={{{', '.join(f'{m}:{temperatures[m][-1]}' for m in MEMBER_ORDER)}}}")

    bundle = {
        "protocol": {
            "cv": f"stratified {n_folds}-fold outer (random_state={OUTER_SEED}); "
                  f"{inner_splits}-fold inner for calibration fitting "
                  f"(random_state={INNER_SEED_BASE}+fold)",
            "note": "identical folds/seed/features as scripts/evaluate_system.py; "
                    "no hyper-parameter or threshold selection on held-out folds",
            "calibration": "per-member temperature scaling (Guo et al. 2017); the "
                           "temperature of a fold is fitted on inner-CV logits of the "
                           "training fold, then applied to the held-out fold",
            "n_samples": n,
            "class_balance": {"ai": int((y == 1).sum()), "real": int((y == 0).sum())},
        },
        "members": {m: MEMBER_DESCRIPTION[m] for m in MEMBER_ORDER},
        "weights": {"deployed": DEPLOYED_WEIGHTS, "extended": EXTENDED_WEIGHTS,
                    "equal": EQUAL_WEIGHTS},
        "clip_external_probe": {"path": str(probe.relative_to(ws)) if probe.is_absolute() else str(probe),
                                "cv_aucs": probe_meta.get("cv_aucs"),
                                "val_auc": probe_meta.get("val_auc"),
                                "note": "cv_aucs/val_auc are on the external 18-generator "
                                        "benchmark, not on the 119 samples"},
        "seqs": [int(s) for s in seqs],
        "y": [int(v) for v in y],
        # NOTE: full float precision on purpose - the logit of a small probability is
        # very sensitive to rounding, and rounding here would make the cached matrix
        # and a recomputation disagree near the ROC operating points.
        "oof_logits": {m: [float(v) for v in logits[m]] for m in MEMBER_ORDER},
        "oof_calibrated": {m: [float(v) for v in calibrated[m]] for m in MEMBER_ORDER},
        "temperatures_per_fold": temperatures,
        "per_member_auc": {m: auc(y, logits[m]) for m in MEMBER_ORDER},
        # deployed operating point (per-fold admitted members + per-fold thresholds)
        "deployed_oof": {
            "definition": "per-fold admitted members, per-fold order-statistic threshold on "
                          "training-fold real scores (fp_budget=%d, target_fpr=0.03); "
                          "identical fold protocol to scripts/evaluate_system.py"
                          % FP_BUDGET_DEFAULT,
            "fused": [float(v) for v in deployed_fused],
            "pred": [int(v) for v in deployed_pred],
            "verdict": [str(v) for v in deployed_verdict],
            "metrics": metrics_at(y, deployed_pred),
            "fst": None,
        },
        "folds": folds,
    }
    bundle["deployed_oof"]["metrics"]["three_tier"] = {
        "suspect": int((deployed_verdict == "suspect").sum()),
        "normal": int((deployed_verdict == "normal").sum()),
        "review": int((deployed_verdict == "review").sum()),
    }
    fp_seqs = [int(seqs[i]) for i in range(n) if y[i] == 0 and deployed_pred[i] == 1]
    fn_seqs = [int(seqs[i]) for i in range(n) if y[i] == 1 and deployed_pred[i] == 0]
    bundle["deployed_oof"]["false_positives_real_seqs"] = fp_seqs
    bundle["deployed_oof"]["false_negatives_ai_seqs"] = fn_seqs
    bundle["deployed_oof"]["guardrails"] = {
        "tpr@fpr=1%": tpr_at_fpr(y, deployed_fused, 0.01),
        "fpr@tpr=95%": fpr_at_tpr(y, deployed_fused, 0.95),
    }
    return bundle


def load_deployed_config(models_dir: Path):
    """Read the frozen production fusion (weights + thresholds) from models/fusion.json."""
    p = Path(models_dir) / "fusion.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {"source": str(p), "weights": d.get("weights", {}),
                "thr_low": d.get("thr_low"), "thr_high": d.get("thr_high"),
                "disagree_review": d.get("disagree_review"),
                "temperatures": d.get("temperatures")}
    return {"source": None, "weights": {"combined": 0.60, "forensic": 0.25},
            "thr_low": None, "thr_high": None, "disagree_review": None,
            "temperatures": None}


def bundle_arrays(bundle: dict) -> dict:
    """Turn the cached JSON bundle back into numpy arrays."""
    out = {
        "y": np.array(bundle["y"], dtype=int),
        "seqs": np.array(bundle["seqs"], dtype=int),
        "oof_logits": {m: np.array(v, dtype=np.float64) for m, v in bundle["oof_logits"].items()},
        "oof_calibrated": {m: np.array(v, dtype=np.float64)
                           for m, v in bundle["oof_calibrated"].items()},
        "deployed_fused": np.array(bundle["deployed_oof"]["fused"], dtype=np.float64),
        "deployed_pred": np.array(bundle["deployed_oof"]["pred"], dtype=int),
        "deployed_verdict": np.array(bundle["deployed_oof"]["verdict"], dtype=object),
    }
    return out


def load_or_build_bundle(cache_path: Path, refresh: bool = False, **kw) -> dict:
    """Reuse the cached OOF matrix unless it is missing or `refresh` is set.

    The cache lives under models/ (git-ignored build artefact) - the evidence files in
    reports/ are always regenerated from the bundle, so no reported number comes from
    the cache alone.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        print(f"[oof] reusing cached OOF matrix: {cache_path} (pass --refresh-oof to rebuild)",
              flush=True)
        return json.loads(cache_path.read_text(encoding="utf-8"))
    bundle = build_bundle(**kw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[oof] cached OOF matrix -> {cache_path}", flush=True)
    return bundle


def write_json(path: Path, payload: dict, mirror_to_repo: bool = True):
    """Write the evidence JSON, and mirror it into the git-tracked repo/ copy.

    The working files live under the workspace (reports/*.json, next to the per-sample
    reports produced by evaluate_system.py), while the numbers that ship with the
    repository - and that the REST API / WebUI read - live in repo/reports/*.json, the
    only reports/ path that .gitignore whitelists. Writing both keeps a single source
    of truth in the script and avoids a manual copy step.
    """
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[save] {path}", flush=True)
    if not mirror_to_repo:
        return
    try:
        rel = path.resolve().relative_to(WS.resolve())
    except ValueError:
        return                                        # caller wrote outside the workspace
    mirror = ROOT / rel
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(text, encoding="utf-8")
    print(f"[mirror] {mirror}", flush=True)


def add_common_args(ap):
    ap.add_argument("--workspace", default=str(WS),
                    help="workspace root holding data/, models/, reports/, features.csv")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--cache", default="models/oof_bundle_119.json")
    ap.add_argument("--refresh-oof", action="store_true",
                    help="rebuild the OOF matrix instead of reusing the cache")
    ap.add_argument("--out", default=None, help="output JSON path (relative to the workspace)")
    return ap
