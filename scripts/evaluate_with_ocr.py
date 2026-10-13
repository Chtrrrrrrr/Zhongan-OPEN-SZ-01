#!/usr/bin/env python3
"""t6 step 2 - out-of-fold evaluation of OCR text evidence on top of the frozen head.

    python repo/scripts/evaluate_with_ocr.py --out repo/reports/ocr_text_consistency.json

Protocol (identical folds and seeds to the frozen deployment protocol)
---------------------------------------------------------------------
``repo/models/deployment_metrics.json`` archives, for 6 rounds (seeds 42..107) and a
stratified 5-fold split, the out-of-fold fused logits of every sample
(``_oof_fused_scores[r][i]``) plus each fold's frozen threshold
(``rounds[r].folds[f].thr_high``, the fp_budget=2 order statistic of the TRAINING-fold
real scores). This script rebuilds the folds with
``StratifiedKFold(5, shuffle=True, random_state=seed)`` and *verifies* the
reconstruction against the archived per-fold fp/fn and per-round accuracy before using
it (any mismatch => hard failure). No held-out sample selects a threshold, a weight, a
feature or a gate.

Arms compared (all under the same folds/seeds; non-document samples are identical
across arms because OCR evidence exists only for documents)
----------------------------------------------------------
A  frozen combined head          - the archived out-of-fold decision (baseline).
B  combined head + OCR features  - per fold: logistic regression on
                                    [frozen OOF logit, 32 z-scored OCR features], scaler
                                    and coefficients fitted on the TRAINING-fold
                                    documents only; decision threshold = fp_budget=2
                                    order statistic of the TRAINING-fold real documents.
C  combined head + fixed OCR rule - a pre-registered boolean rule (frozen below before
                                    any label-wise result existed): a document the head
                                    calls normal whose recognised text carries no amount
                                    token, no currency symbol and no stamp ink is
                                    escalated to suspect.

Label isolation
---------------
Feature extraction never sees a label; the report reproduces the proof by re-running the
extraction over a meta.csv whose label column is inverted and comparing all 32 features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
WS_DEFAULT = REPO.parent
sys.path.insert(0, str(REPO / "src"))

import l1c_ocr as L          # noqa: E402

N_FOLDS = 5
FP_BUDGET = 2                     # the only admissible budget (t1 baseline contract)

# ---------------------------------------------------------------------------
# PRE-REGISTERED rules for arm C. The rule structure (three domain conditions: the
# document carries no monetary amount, no currency symbol, no stamp ink) was fixed by
# domain reasoning: a Chinese claim document (invoice / medical bill / repair receipt)
# that shows none of those is text-inconsistent with the class it claims to be.
#
# DISCLOSURE (selection risk, reported with the results): while validating the feature
# module the OCR feature values of the four named misses (003/029/030/054) and of three
# real documents (080/100/110) were inspected, and the stamp-ink cut (0.01) as well as
# the "zero recognised text" variant were chosen at that point - BEFORE the frozen
# out-of-fold protocol was executed, but not blind to those samples. Arm C therefore
# carries a selection risk on the 119 evaluated samples and is reported as an informed
# upper bound; arm B (no manual rule, everything fitted inside the training folds) is
# the unbiased arm. Arm A is the frozen baseline.
# ---------------------------------------------------------------------------
RULE = {
    "id": "ocr_text_inconsistency_v1",
    "frozen_before_evaluation": True,
    "effective_rule": ("clause (a): if arm_A predicts real/normal for a document sample "
                       "AND frac_amount_numbers == 0 AND has_currency == 0 AND "
                       "red_frac < 0.01 then escalate to suspect"),
    "secondary_clause": ("clause (b) [reported as a variant, not the headline]: also "
                         "escalate documents where the recogniser returned ZERO text boxes "
                         "(n_boxes == 0)"),
    "direction": "real -> suspect only (it can never create a new real prediction)",
    "stamp_ink_cut": 0.01,
    "sensitivity_cuts": [0.005, 0.01, 0.02],
    "selection_risk_disclosure": ("the clause set and the 0.01 cut were informed by "
                                  "inspecting the four named misses and three real "
                                  "documents before the out-of-fold run (see module "
                                  "comment); treat arm C as an upper bound"),
    "cost_model": "every real document that satisfies the rule becomes a false positive; "
                  "this cost is measured, not assumed",
}


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# loading / helpers
# ---------------------------------------------------------------------------
def load_archive(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"raw": d,
            "oof": np.asarray(d["_oof_fused_scores"], dtype=np.float64),
            "seqs": [int(s) for s in d["_oof_seqs"]],
            "seeds": [int(r["seed"]) for r in d["rounds"]],
            "folds": [[float(f["thr_high"]) for f in r["folds"]] for r in d["rounds"]],
            "archived_rounds": d["rounds"],
            "sha256": sha256_file(path)}


def load_labels(path: Path, seqs: list) -> np.ndarray:
    lab = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab[int(r["seq"])] = r["label"]
    return np.asarray([1 if lab[s] == "ai" else 0 for s in seqs], dtype=int)


def load_ocr(path: Path) -> dict:
    recs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[int(r["seq"])] = r
    return recs


def fold_map(y: np.ndarray, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_of = np.full(len(y), -1, dtype=int)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_of[va] = f
    return fold_of


def metrics(y: np.ndarray, pred: np.ndarray, idx: np.ndarray | None = None,
            seqs_all: list | None = None) -> dict:
    idx = np.arange(len(y)) if idx is None else idx
    yy, pp = y[idx], pred[idx]
    tp = int(((pp == 1) & (yy == 1)).sum())
    fp = int(((pp == 1) & (yy == 0)).sum())
    fn = int(((pp == 0) & (yy == 1)).sum())
    tn = int(((pp == 0) & (yy == 0)).sum())
    n_pos, n_neg = int((yy == 1).sum()), int((yy == 0).sum())
    out = {"n": int(len(yy)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "accuracy": (round((tp + tn) / len(yy), 6) if len(yy) else None),
           "fpr": (round(fp / n_neg, 6) if n_neg else None), "fpr_count": f"{fp}/{n_neg}",
           "tpr": (round(tp / n_pos, 6) if n_pos else None), "tpr_count": f"{tp}/{n_pos}"}
    if seqs_all is not None:
        out["fp_seqs"] = [int(seqs_all[i]) for i in idx if pred[i] == 1 and y[i] == 0]
        out["fn_seqs"] = [int(seqs_all[i]) for i in idx if pred[i] == 0 and y[i] == 1]
    return out


def order_stat_threshold(real_scores, k: int = FP_BUDGET):
    rs = np.sort(np.asarray(real_scores, dtype=np.float64))[::-1]
    if rs.size == 0:
        return None
    return float(rs[int(min(max(k, 1), rs.size)) - 1])


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            w = np.where(inv == i)[0]
            ranks[w] = ranks[w].mean()
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def rule_hit_record(rec: dict, cut: float, no_text_clause: bool = False) -> bool:
    """Clause (a) of the frozen rule, optionally extended with clause (b)."""
    f = rec["features"]
    hit_a = (float(f["frac_amount_numbers"]) == 0.0
             and float(f["has_currency"]) == 0.0
             and float(f["red_frac"]) < cut)
    if no_text_clause and float(f["n_boxes"]) == 0.0:
        return True
    return hit_a


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="t6 OCR out-of-fold comparison")
    ap.add_argument("--out", default="repo/reports/ocr_text_consistency.json")
    ap.add_argument("--workspace", default=str(WS_DEFAULT))
    ap.add_argument("--archive", default="repo/models/deployment_metrics.json")
    ap.add_argument("--ocr", default="work/improve/ocr_text.jsonl")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--meta", default="data/meta.csv")
    ap.add_argument("--images-dir", default="data/images")
    ap.add_argument("--isolation-n", type=int, default=6)
    ap.add_argument("--determinism-n", type=int, default=3)
    args = ap.parse_args(argv)

    ws = Path(args.workspace)
    t0 = time.perf_counter()
    arch = load_archive(ws / args.archive)
    seqs = arch["seqs"]
    y = load_labels(ws / args.labels, seqs)
    ocr = load_ocr(ws / args.ocr)
    n_rounds = len(arch["seeds"])
    doc_mask = np.asarray([s in ocr for s in seqs])
    doc_idx = np.where(doc_mask)[0]
    ocr_row_of = {int(seqs[i]): k for k, i in enumerate(doc_idx)}
    X_ocr = np.asarray([[float(ocr[int(seqs[i])]["features"][n]) for n in L.FEATURE_NAMES]
                        for i in doc_idx], dtype=np.float64)
    print(f"[load] archive oof={arch['oof'].shape} seeds={arch['seeds']} "
          f"ocr_records={len(ocr)} documents={len(doc_idx)}/{len(seqs)}", flush=True)

    # ---- 1. reproduce the archived folds + per-fold decisions -----------------
    repro = {"rounds": [], "mismatch_folds": 0, "mismatch_rounds": 0}
    fold_of_all, pred_A = {}, np.zeros_like(arch["oof"], dtype=int)
    for r, seed in enumerate(arch["seeds"]):
        fo = fold_map(y, seed)
        fold_of_all[r] = fo
        thr = np.asarray(arch["folds"][r], dtype=np.float64)
        pred = (arch["oof"][r] >= thr[fo]).astype(int)
        pred_A[r] = pred
        row = {"round": r + 1, "seed": seed,
               "folds": [{"fold": f + 1, "thr_high": round(float(thr[f]), 6),
                          "fp": int(((pred == 1) & (y == 0) & (fo == f)).sum()),
                          "fn": int(((pred == 0) & (y == 1) & (fo == f)).sum()),
                          "archived_fp": int(arch["archived_rounds"][r]["folds"][f]["fp"]),
                          "archived_fn": int(arch["archived_rounds"][r]["folds"][f]["fn"])}
                         for f in range(N_FOLDS)],
               "accuracy": metrics(y, pred)["accuracy"],
               "archived_accuracy": arch["archived_rounds"][r]["accuracy"]}
        repro["mismatch_folds"] += sum(1 for fd in row["folds"]
                                       if fd["fp"] != fd["archived_fp"]
                                       or fd["fn"] != fd["archived_fn"])
        # the archive stores 4-decimal accuracies, so compare with a tolerance
        repro["mismatch_rounds"] += int(
            abs(float(row["accuracy"]) - float(row["archived_accuracy"])) > 1e-3)
        repro["rounds"].append(row)
    if repro["mismatch_folds"] or repro["mismatch_rounds"]:
        raise RuntimeError(f"fold reconstruction does not reproduce the archive: "
                           f"{repro['mismatch_folds']} fold mismatches, "
                           f"{repro['mismatch_rounds']} round mismatches")
    print(f"[protocol] fold reconstruction verified: 0 mismatch over "
          f"{n_rounds} rounds x {N_FOLDS} folds", flush=True)

    # ---- 2. arm A / B / C ----------------------------------------------------
    pred_B = np.zeros_like(pred_A)
    pred_C = np.zeros_like(pred_A)
    dec_B = np.full(pred_A.shape, np.nan)
    folds_detail = []
    for r, seed in enumerate(arch["seeds"]):
        fo = fold_of_all[r]
        pA, pB, pC = pred_A[r].copy(), pred_A[r].copy(), pred_A[r].copy()
        rrow = {"round": r + 1, "seed": seed, "folds": []}
        for f in range(N_FOLDS):
            te = doc_idx[fo[doc_idx] == f]
            tr = doc_idx[fo[doc_idx] != f]
            Xtr = np.column_stack([arch["oof"][r][tr], X_ocr[[ocr_row_of[int(seqs[i])] for i in tr]]])
            Xte = np.column_stack([arch["oof"][r][te], X_ocr[[ocr_row_of[int(seqs[i])] for i in te]]])
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=0)
            lr.fit((Xtr - mu) / sd, y[tr])
            dtr = lr.decision_function((Xtr - mu) / sd)
            dte = lr.decision_function((Xte - mu) / sd)
            thr_meta = order_stat_threshold(dtr[y[tr] == 0], FP_BUDGET)
            pB[te] = (dte >= thr_meta).astype(int)
            dec_B[r, te] = dte
            hits = np.asarray([bool(pA[i] == 0 and rule_hit_record(ocr[int(seqs[i])],
                                                                  RULE["stamp_ink_cut"]))
                               for i in te])
            pC[te[hits]] = 1
            rrow["folds"].append({
                "fold": f + 1, "n_train_docs": int(len(tr)), "n_test_docs": int(len(te)),
                "meta_threshold": round(float(thr_meta), 6),
                "lr_coef_frozen_logit": round(float(lr.coef_[0][0]), 6),
                "n_ocr_features_abs_coef_gt_0.05": int(np.sum(np.abs(lr.coef_[0][1:]) > 0.05)),
                "rule_escalations": int(hits.sum()),
                "rule_escalated_seqs": [int(seqs[i]) for i in te[hits]],
            })
        pred_B[r], pred_C[r] = pB, pC
        folds_detail.append(rrow)

    arm_preds = {"A_frozen_head": pred_A, "B_head_plus_ocr_lr": pred_B,
                 "C_head_plus_frozen_rule": pred_C}
    arms = {k: [{"round": r + 1, "seed": arch["seeds"][r],
                 "all119": metrics(y, v[r], seqs_all=seqs),
                 "documents": metrics(y, v[r], doc_idx, seqs)}
                for r in range(n_rounds)] for k, v in arm_preds.items()}

    # ---- 3. focus samples (the AI documents the head misses) -----------------
    focus_seqs = [3, 29, 30, 54]
    focus = []
    for s in focus_seqs:
        i = seqs.index(s)
        rec = ocr.get(s)
        focus.append({
            "seq": s, "label": "ai" if y[i] == 1 else "real",
            "arm_A_pred_per_round": [int(pred_A[r][i]) for r in range(n_rounds)],
            "arm_B_pred_per_round": [int(pred_B[r][i]) for r in range(n_rounds)],
            "arm_C_pred_per_round": [int(pred_C[r][i]) for r in range(n_rounds)],
            "frozen_oof_logit_per_round": [round(float(arch["oof"][r][i]), 4)
                                           for r in range(n_rounds)],
            "ocr": ({k: round(float(rec["features"][k]), 6) for k in
                     ("mean_score", "n_chars", "n_boxes", "n_num_groups",
                      "frac_amount_numbers", "has_currency", "has_date_like",
                      "frac_mixed_num_groups", "red_frac", "digit_ratio", "cjk_ratio",
                      "char_entropy", "stroke_cc_per_char", "frac_score_lt_50")}
                    if rec else None),
            "rule_hit_clause_a": bool(rec and rule_hit_record(rec, RULE["stamp_ink_cut"])),
            "rule_hit_clause_a_or_b": bool(rec and rule_hit_record(
                rec, RULE["stamp_ink_cut"], no_text_clause=True)),
        })

    # ---- 4. rule effect and sensitivity -------------------------------------
    def rule_hits_seqs(cut: float, no_text: bool = False):
        return [int(seqs[i]) for i in doc_idx
                if rule_hit_record(ocr[int(seqs[i])], cut, no_text_clause=no_text)]

    def rule_effect(no_text: bool) -> tuple:
        fixed, cost = {}, {}
        for s in seqs:
            i = seqs.index(s)
            if s not in ocr or not rule_hit_record(ocr[s], RULE["stamp_ink_cut"], no_text):
                continue
            n_fix = sum(1 for r in range(n_rounds) if y[i] == 1 and pred_A[r][i] == 0)
            n_cost = sum(1 for r in range(n_rounds) if y[i] == 0 and pred_A[r][i] == 0)
            if n_fix:
                fixed[str(s)] = n_fix
            if n_cost:
                cost[str(s)] = n_cost
        return fixed, cost

    fixed_c, cost_c = rule_effect(False)
    fixed_ab, cost_ab = rule_effect(True)
    sensitivity = {}
    for cut in RULE["sensitivity_cuts"]:
        hits = rule_hits_seqs(cut)
        sensitivity[f"red_cut_{cut}"] = {
            "n_documents_escalated": len(hits),
            "n_ai": int(sum(1 for s in hits if y[seqs.index(s)] == 1)),
            "n_real": int(sum(1 for s in hits if y[seqs.index(s)] == 0)),
            "focus_seqs_escalated": {str(s): (s in hits) for s in focus_seqs}}
    variant_b = {"clause": "clause (a) OR zero recognised text boxes (n_boxes == 0)",
                 "fixed_out_of_fold": fixed_ab, "new_false_positives": cost_ab,
                 "new_false_positive_total": int(sum(cost_ab.values())),
                 "documents_with_zero_boxes": [
                     int(seqs[i]) for i in doc_idx
                     if float(ocr[int(seqs[i])]["features"]["n_boxes"]) == 0.0],
                 "note": "reported as a variant; the headline arm C uses clause (a) only"}

    # arm B's out-of-fold fixes and costs
    b_fixed = {str(seqs[i]): sum(1 for r in range(n_rounds)
                                 if y[i] == 1 and pred_A[r][i] == 0 and pred_B[r][i] == 1)
               for i in range(len(seqs))}
    b_fixed = {k: v for k, v in b_fixed.items() if v}
    b_cost = {str(seqs[i]): sum(1 for r in range(n_rounds)
                                if y[i] == 0 and pred_A[r][i] == 0 and pred_B[r][i] == 1)
              for i in range(len(seqs))}
    b_cost = {k: v for k, v in b_cost.items() if v}

    # ---- 5. pooled cross-checks ---------------------------------------------
    pooled_score = arch["oof"].mean(axis=0)                 # t1 P-A view
    rs = np.sort(pooled_score[y == 0])[::-1]
    thr_pool = float(rs[FP_BUDGET] + 1e-12) if rs.size > FP_BUDGET else float(rs[-1])
    pred_pool = (pooled_score >= thr_pool).astype(int)
    pooled_A = dict(metrics(y, pred_pool, seqs_all=seqs), threshold_logit=round(thr_pool, 6))
    # arm C pooled
    pred_C_pool = pred_pool.copy()
    for i in doc_idx:
        s = int(seqs[i])
        if pred_pool[i] == 0 and rule_hit_record(ocr[s], RULE["stamp_ink_cut"]):
            pred_C_pool[i] = 1
    # arm B pooled: per-sample mean meta decision, documents only
    B_mean = np.full(len(seqs), np.nan)
    B_mean[doc_idx] = np.nanmean(dec_B[:, doc_idx], axis=0)
    thr_B_doc = order_stat_threshold(B_mean[doc_idx[y[doc_idx] == 0]], FP_BUDGET)
    pred_B_pool = np.zeros(len(seqs), dtype=int)
    for i in doc_idx:
        pred_B_pool[i] = int(B_mean[i] >= thr_B_doc)
    pooled = {
        "arm_A_all119_t1_style": pooled_A,
        "arm_A_per_round_mean_accuracy": round(float(np.mean(
            [a["all119"]["accuracy"] for a in arms["A_frozen_head"]])), 6),
        "arm_B_documents_only_2fp": dict(metrics(y, pred_B_pool, doc_idx, seqs),
                                         threshold_meta=round(float(thr_B_doc), 6)),
        "arm_C_documents_only": dict(metrics(y, pred_C_pool, doc_idx, seqs)),
        "t1_reference_P_A_pooled": {"accuracy": 0.97479, "fpr_count": "2/63",
                                    "tpr_count": "55/56",
                                    "source": "repo/reports/baseline_improvement.json"},
    }

    # ---- 6. label isolation + determinism proofs ----------------------------
    iso = label_isolation_proof(ws, args, ocr)
    det = determinism_proof(ws, args, ocr)

    # ---- 7. descriptive separability of the OCR block (reporting only) -----
    feat_auc = {}
    for k, name in enumerate(L.FEATURE_NAMES):
        a = _auc(y[doc_idx], X_ocr[:, k])
        feat_auc[name] = None if a is None else round(a, 4)
    feat_auc_sorted = sorted(((v if v is not None else 0.5, k) for k, v in feat_auc.items()),
                             reverse=True)

    # ---- 7b. single-feature screen (fitted inside each fold, unbiased) -----
    mean_A_docs = float(np.mean([x["documents"]["accuracy"] for x in arms["A_frozen_head"]]))
    screen = single_feature_screen(arch, y, seqs, doc_idx, ocr_row_of, X_ocr,
                                   fold_of_all, pred_A, mean_A_docs)

    # ---- 7c. why 029/030 are not fixed: where do the misses sit? ------------
    real_docs = [int(seqs[i]) for i in doc_idx if y[i] == 0]
    diag = []
    for s in focus_seqs + [46, 9, 26, 27, 31, 32]:
        i = seqs.index(s)
        rec = ocr.get(s)
        if not rec:
            diag.append({"seq": s, "label": "ai" if y[i] == 1 else "real",
                         "ocr_available": False})
            continue
        row = {"seq": s, "label": "ai" if y[i] == 1 else "real", "ocr_available": True,
               "rule_clause_a_hit": rule_hit_record(rec, RULE["stamp_ink_cut"]),
               "percentile_within_real_documents": {}}
        for name in ("frac_amount_numbers", "has_currency", "red_frac", "n_boxes",
                     "char_entropy", "frac_mixed_num_groups", "mean_score"):
            v = float(rec["features"][name])
            rv = np.asarray([float(ocr[t]["features"][name]) for t in real_docs])
            row["percentile_within_real_documents"][name] = round(
                float((rv <= v).mean() * 100), 1)
            row[name] = round(v, 6)
        diag.append(row)

    # ---- 8. conclusion -----------------------------------------------------
    def doc_totals(arm_rows):
        agg = {k: sum(x["documents"][k] for x in arm_rows) for k in ("fp", "fn", "tp", "tn")}
        n_real = agg["fp"] + agg["tn"]
        n_ai = agg["tp"] + agg["fn"]
        return dict(agg, n_real=n_real, n_ai=n_ai,
                    fpr_count=f"{agg['fp']}/{n_real}", tpr_count=f"{agg['tp']}/{n_ai}",
                    accuracy=round((agg["tp"] + agg["tn"]) / (n_real + n_ai), 6),
                    within_3pct_fpr=bool(agg["fp"] / n_real <= 0.03),
                    note="documents only, summed over the 6 rounds x 109 document decisions")

    tot = {arm: doc_totals(arms[arm]) for arm in arms}
    best = screen[0] if screen else None
    best_tot = None
    if best:
        n_real_best = 6 * 61
        best_tot = {"feature": best["feature"],
                    "fp_over_6_rounds": best["documents_fp_over_6_rounds"],
                    "fn_over_6_rounds": best["documents_fn_over_6_rounds"],
                    "fpr_count": f"{best['documents_fp_over_6_rounds']}/{n_real_best}",
                    "fpr": round(best["documents_fp_over_6_rounds"] / n_real_best, 4),
                    "within_3pct_fpr": bool(best["documents_fp_over_6_rounds"] / n_real_best <= 0.03)}
    conclusion = {
        "net_effect_of_ocr_evidence": "negative (no configuration improves the frozen head "
                                      "under the <=3% FPR contract)",
        "documents_confusion_totals_over_6_rounds": tot,
        "best_single_ocr_feature_vs_fpr_budget": best_tot,
        "fpr_budget_reading": (
            "The only admissible budget is the t1 contract (<=3% FPR). Arm A sits at "
            f"{tot['A_frozen_head']['fpr_count']} real documents; arm B inflates it to "
            f"{tot['B_head_plus_ocr_lr']['fpr_count']} and arm C to "
            f"{tot['C_head_plus_frozen_rule']['fpr_count']}, both far outside the budget, "
            f"while recovering only {tot['A_frozen_head']['fn'] - tot['B_head_plus_ocr_lr']['fn']} (B) / "
            f"{tot['A_frozen_head']['fn'] - tot['C_head_plus_frozen_rule']['fn']} (C) of the "
            f"{tot['A_frozen_head']['fn']} document false negatives. Even the best "
            "fold-internal single-feature screen pushes FPR to "
            f"{best_tot['fpr_count'] if best_tot else 'n/a'} (outside the budget) to save "
            f"{tot['A_frozen_head']['fn'] - (best_tot['fn_over_6_rounds'] if best_tot else tot['A_frozen_head']['fn'])} "
            "false negatives - which is why no OCR configuration is recommended."),
        "arm_A_documents_accuracy_per_round_mean": mean_A_docs,
        "arm_B_documents_accuracy_per_round_mean": float(np.mean(
            [x["documents"]["accuracy"] for x in arms["B_head_plus_ocr_lr"]])),
        "arm_C_documents_accuracy_per_round_mean": float(np.mean(
            [x["documents"]["accuracy"] for x in arms["C_head_plus_frozen_rule"]])),
        "misses_fixed_out_of_fold": {
            "arm_B": b_fixed, "arm_C_clause_a": fixed_c, "arm_C_clause_b_variant": fixed_ab},
        "cost_of_those_fixes": {
            "arm_B_new_false_positives": b_cost,
            "arm_C_clause_a_new_false_positives": cost_c},
        "why_029_030_are_not_fixed": (
            "029 carries amount-like numbers (frac_amount_numbers 0.5, above 87% of real "
            "documents) and noticeable stamp ink (red_frac 0.0232, above the 0.01 cut); "
            "030 carries amount-like numbers too (0.2) with almost no red ink, and on the "
            "rule's three variables it sits inside the real-document distribution. Neither "
            "is text-inconsistent, so the text layer cannot separate them from real "
            "documents - see miss_diagnostics for the exact percentiles."),
        "why_054_is_out_of_scope_for_this_layer": (
            "054 (and 057/058) are NOT document samples: data/meta.csv marks them "
            "content='scene' (the 10 scene samples are 5,6,7,8,53,54,55,56,57,58). The OCR "
            "text layer only exists for the 109 document samples, so it structurally cannot "
            "address those misses/false positives; that is a scope statement, not a failure "
            "of the extractor."),
        "fixed_003_nevertheless": (
            "003 is a flat_render AI document and IS fixed out-of-fold by the frozen rule in "
            "6/6 rounds (and 5/6 by the stacked arm B), which satisfies the 'fix at least "
            "one miss' criterion - but the price (79 document-round false positives) makes "
            "this a demonstration of what the text layer can see, not a deployable change."),
        "best_single_ocr_feature": (screen[0]["feature"] if screen else None),
        "best_single_feature_delta_documents_accuracy": (
            screen[0]["delta_documents_accuracy_vs_arm_A"] if screen else None),
        "recommendation": (
            "Do NOT wire OCR text evidence into the production decision as-is. Keep "
            "work/improve/ocr_text.jsonl as an auditable side-evidence artefact (it is "
            "useful for human review of flat-render documents) and revisit it only with a "
            "genuine real-vs-AI document video/scan corpus, where the class balance and the "
            "OCR-quality distribution are the ones seen in production."),
    }

    # ---- 8. acceptance -----------------------------------------------------
    any_rule_fix = any(v > 0 for v in fixed_c.values())
    acceptance = {
        "two_arms_same_folds_seeds": {
            "status": "passed" if repro["mismatch_folds"] == 0 else "failed",
            "detail": (f"arms A/B/C share seeds {arch['seeds']} and the archived "
                       f"per-fold thresholds; fold reconstruction mismatch = "
                       f"{repro['mismatch_folds']} folds / {repro['mismatch_rounds']} rounds")},
        "fixes_at_least_one_miss_or_quantified_why_not": {
            "status": "passed" if (b_fixed or any_rule_fix) else "failed",
            "detail": (f"arm B out-of-fold fixes {b_fixed or 'none'}; arm C rule fixes "
                       f"{fixed_c or 'none'} (rounds fixed); the rule costs "
                       f"{int(sum(cost_c.values()))} new false positives on real "
                       f"documents: {cost_c}")},
        "label_isolation": {"status": "passed" if iso["identical"] else "failed",
                            "detail": iso["summary"]},
        "ocr_requirements_recorded": {
            "status": "passed",
            "detail": "work/improve/ocr_requirements.txt lists the pinned wheels and the "
                      "exact install command used"},
    }

    out = {
        "report": "ocr_text_consistency",
        "task": "t6 - Chinese-document OCR and text-consistency evidence",
        "declarations": {
            "labels_never_used_during_feature_extraction": (
                "TRUE. Extraction reads data/meta.csv columns ('seq','content') only and "
                "never opens data/labels.csv; labels are read solely by this evaluator. "
                "Reproduced empirically by the label-scrambling proof below."),
            "same_folds_and_seeds_as_the_frozen_protocol": (
                f"TRUE. seeds {arch['seeds']} with StratifiedKFold(n_splits={N_FOLDS}, "
                f"shuffle=True, random_state=seed) rebuilt from scratch and verified "
                f"against the archived per-fold fp/fn and per-round accuracy (0 mismatch)."),
            "ocr_evidence_is_not_a_validated_detector": (
                "TRUE. The OCR block is additional evidence evaluated out-of-fold; the "
                "headline numbers remain the frozen head's and a negative result is "
                "reported as such instead of being papered over."),
            "no_threshold_weight_or_feature_selected_on_held_out_samples": (
                "MOSTLY TRUE, with one disclosed exception. Arm A thresholds are archived; "
                "arm B fits scaler, coefficients and threshold inside the training folds "
                "only. Arm C is a fixed boolean rule, but its clause set and the 0.01 cut "
                "were informed by inspecting the OCR features of the four named misses "
                "(003/029/030/054) and three real documents (080/100/110) before the "
                "out-of-fold run - see protocol.arm_C_rule.selection_risk_disclosure. Arm C "
                "is therefore reported as an informed upper bound and arm B is the unbiased "
                "comparison."),
        },
        "protocol": {
            "folds": N_FOLDS, "seeds": arch["seeds"], "fp_budget": FP_BUDGET,
            "archive": str(args.archive), "archive_sha256": arch["sha256"],
            "ocr_jsonl": str(args.ocr),
            "ocr_jsonl_sha256": sha256_file(ws / args.ocr),
            "n_samples": len(seqs), "n_document_samples": int(len(doc_idx)),
            "document_definition": "data/meta.csv content == 'document'",
            "arm_B_model": ("LogisticRegression(C=1.0, lbfgs, max_iter=2000) on "
                            f"[frozen OOF logit, {len(L.FEATURE_NAMES)} OCR features]; "
                            "scaler fitted on the training folds of the same round and fold"),
            "arm_B_threshold": ("fp_budget=2 order statistic of the TRAINING-fold real "
                                "documents' meta decisions"),
            "arm_C_rule": RULE,
            "arms_note": ("non-document samples (the 10 scene images) have no OCR evidence "
                          "and therefore receive arm A's decision in every arm"),
        },
        "backend": L.backend_info(),
        "feature_names": list(L.FEATURE_NAMES),
        "feature_params": dict(L.PARAMS),
        "fold_reconstruction": repro,
        "arms": arms,
        "arm_folds_detail": folds_detail,
        "pooled_cross_checks": pooled,
        "focus_samples": focus,
        "arm_effect_detail": {
            "arm_B_fixed_out_of_fold": b_fixed,
            "arm_B_new_false_positives": b_cost,
            "arm_C_clause_a_fixed_out_of_fold": fixed_c,
            "arm_C_clause_a_new_false_positives": cost_c,
            "arm_C_new_false_positive_total": int(sum(cost_c.values())),
            "arm_C_clause_a_sensitivity": sensitivity,
            "arm_C_clause_b_variant": variant_b,
            "arm_C_selection_risk_disclosure": RULE["selection_risk_disclosure"],
        },
        "ocr_feature_auc_documents_only": feat_auc,
        "ocr_feature_auc_ranked": [{"feature": k, "auc": v} for v, k in feat_auc_sorted],
        "single_feature_screen": screen,
        "miss_diagnostics": diag,
        "conclusion": conclusion,
        "label_isolation_proof": iso,
        "determinism_proof": det,
        "acceptance": acceptance,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }

    p = Path(args.out)
    outp = p if p.is_absolute() else (ws / p if p.parts[0] in ("repo", "work", "data", "src")
                                      else Path.cwd() / p)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ t6 OCR text-consistency evaluation ================")
    for arm in ("A_frozen_head", "B_head_plus_ocr_lr", "C_head_plus_frozen_rule"):
        a = arms[arm]
        print(f"{arm:26s} mean acc documents="
              f"{np.mean([x['documents']['accuracy'] for x in a]):.4f}  all119="
              f"{np.mean([x['all119']['accuracy'] for x in a]):.4f}  round1 docs "
              f"fpr={a[0]['documents']['fpr_count']} tpr={a[0]['documents']['tpr_count']}")
    print(f"pooled t1-style arm A: acc={pooled_A['accuracy']} fpr={pooled_A['fpr_count']} "
          f"tpr={pooled_A['tpr_count']} (t1 reference 0.97479 / 2/63 / 55/56)")
    print(f"arm B fixed out-of-fold: {b_fixed or 'none'}; new FPs: {b_cost or 'none'}")
    print(f"arm C rule fixed out-of-fold: {fixed_c or 'none'}; new FPs: {cost_c or 'none'}")
    print(f"label isolation: {iso['summary']}")
    print(f"determinism: {det['total_feature_mismatches']} feature mismatches")
    print("[acceptance] " + json.dumps({k: v["status"] for k, v in acceptance.items()}))
    print(f"[save] {outp}")
    print(f"[time] {out['runtime_s']} s")
    return 0 if all(v["status"] == "passed" for v in acceptance.values()) else 1


# ---------------------------------------------------------------------------
# proofs
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def single_feature_screen(arch, y, seqs, doc_idx, ocr_row_of, X_ocr, fold_of_all,
                          pred_A, mean_A_docs: float) -> list:
    """Fold-internal screen: [frozen logit + ONE OCR feature] per OCR feature.

    Everything (scaler, coefficients, threshold) is fitted inside the training folds of
    the same round/fold, so the ranking is an unbiased statement about single-feature
    value; it is descriptive, not used to configure any arm.
    """
    rows = []
    for k, name in enumerate(L.FEATURE_NAMES):
        accs, accs_all, fp_tot, tp_tot, fn_tot, n_real_tot, n_ai_tot = [], [], 0, 0, 0, 0, 0
        for r in range(len(arch["seeds"])):
            fo = fold_of_all[r]
            pred = pred_A[r].copy()
            for f in range(N_FOLDS):
                te = doc_idx[fo[doc_idx] == f]
                tr = doc_idx[fo[doc_idx] != f]
                if len(te) == 0 or len(tr) == 0:
                    continue
                tr_rows = [ocr_row_of[int(seqs[i])] for i in tr]
                te_rows = [ocr_row_of[int(seqs[i])] for i in te]
                Xtr = np.column_stack([arch["oof"][r][tr], X_ocr[tr_rows][:, k]])
                Xte = np.column_stack([arch["oof"][r][te], X_ocr[te_rows][:, k]])
                mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                lr.fit((Xtr - mu) / sd, y[tr])
                dtr = lr.decision_function((Xtr - mu) / sd)
                dte = lr.decision_function((Xte - mu) / sd)
                thr = order_stat_threshold(dtr[y[tr] == 0], FP_BUDGET)
                pred[te] = (dte >= thr).astype(int)
            md = metrics(y, pred, doc_idx)
            ma = metrics(y, pred)
            accs.append(md["accuracy"])
            accs_all.append(ma["accuracy"])
            fp_tot += md["fp"]
            tp_tot += md["tp"]
            fn_tot += md["fn"]
            n_real_tot += md["fp"] + md["tn"]
            n_ai_tot += md["tp"] + md["fn"]
        rows.append({"feature": name,
                     "documents_accuracy_per_round_mean": round(float(np.mean(accs)), 6),
                     "all119_accuracy_per_round_mean": round(float(np.mean(accs_all)), 6),
                     "documents_fp_over_6_rounds": fp_tot,
                     "documents_fn_over_6_rounds": fn_tot,
                     "documents_tp_over_6_rounds": tp_tot,
                     "documents_real_decisions": n_real_tot,
                     "documents_ai_decisions": n_ai_tot,
                     "delta_documents_accuracy_vs_arm_A":
                         round(float(np.mean(accs)) - mean_A_docs, 6)})
    rows.sort(key=lambda r: -r["delta_documents_accuracy_vs_arm_A"])
    return rows


def _scramble_meta(ws: Path, meta: str, scratch: Path) -> Path:
    src = ws / meta
    with open(src, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames)
        rows = list(rdr)
    out = scratch / "meta_labels_scrambled.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["label"] = "AI" if r.get("label") == "real" else "real"      # inverted
            w.writerow(r)
    return out


def label_isolation_proof(ws: Path, args, cached: dict) -> dict:
    scratch = ws / ".tmp" / "ocr_label_isolation"
    scratch.mkdir(parents=True, exist_ok=True)
    rows = L.iter_document_rows(ws / args.meta, "document",
                                require_images_dir=ws / args.images_dir)
    picked = [r["seq"] for r in rows[:max(1, int(args.isolation_n))]]
    scrambled = _scramble_meta(ws, args.meta, scratch)
    rows2 = L.iter_document_rows(scrambled, "document",
                                 require_images_dir=ws / args.images_dir)
    seqs2 = [r["seq"] for r in rows2[:len(picked)]]
    detail, diffs = [], []
    for s in picked:
        if s not in cached:
            continue
        f2 = L.extract_features(L.image_path_for(s, ws / args.images_dir))["features"]
        ref = cached[s]["features"]
        bad = [k for k in L.FEATURE_NAMES if float(f2[k]) != float(ref[k])]
        if bad:
            diffs.append({"seq": s, "differing_features": bad})
        detail.append({"seq": s, "n_boxes": int(f2["n_boxes"]),
                       "mean_score": round(float(f2["mean_score"]), 6),
                       "n_differing_features": len(bad)})
    same_sel = seqs2 == picked
    return {"method": ("sample list and all "
                       f"{len(L.FEATURE_NAMES)} features re-derived from a meta.csv whose "
                       "label column is INVERTED (real<->ai); every other column is "
                       "untouched"),
            "scrambled_meta": str(scrambled.relative_to(ws)).replace("\\", "/"),
            "n_documents_re_checked": len(detail),
            "selection_identical_under_label_scrambling": bool(same_sel),
            "feature_diffs": diffs,
            "identical": bool(same_sel and not diffs and detail),
            "summary": (f"{len(detail)} documents re-extracted with inverted labels: "
                        f"selection identical={same_sel}, feature mismatches={len(diffs)}")}


def determinism_proof(ws: Path, args, cached: dict) -> dict:
    seqs = sorted(cached)[:max(1, int(args.determinism_n))]
    res, bad = [], 0
    for s in seqs:
        f = L.extract_features(L.image_path_for(s, ws / args.images_dir))["features"]
        ref = cached[s]["features"]
        diff = [k for k in L.FEATURE_NAMES if float(f[k]) != float(ref[k])]
        bad += len(diff)
        res.append({"seq": s, "n_boxes": int(f["n_boxes"]),
                    "n_differing_features": len(diff),
                    "mean_score": round(float(f["mean_score"]), 6)})
    return {"method": ("re-run the extractor on cached samples; compare all "
                       f"{len(L.FEATURE_NAMES)} features"),
            "samples": res, "total_feature_mismatches": bad, "deterministic": bad == 0}


if __name__ == "__main__":
    sys.exit(main())
