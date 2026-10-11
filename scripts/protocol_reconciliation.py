#!/usr/bin/env python3
"""Reconcile the frozen data split with the evaluation protocol actually used.

Audit context (repo/docs/compliance-audit.md F3c / A6.5 / B3.8 / B11.3): the frozen
split in data/splits_frozen.json is a single 0.6/0.2/0.2 partition, while the headline
metrics and this metrics pack use a stratified 5-fold out-of-fold (OOF) protocol over
all 119 samples. The two are different protocols and the audit rated the item "partially
compliant". This script produces repo/reports/protocol_reconciliation.json, which

  1. states each protocol's purpose and the authority relationship between them;
  2. quantifies why the headline metrics are computed out-of-fold instead of on the
     frozen 20% test split (a 24-sample test set cannot separate 95% from 83%);
  3. publishes a reproducible signature of the evaluation protocol - fold count, seeds,
     stratification key, per-sample fold assignment hash, feature/artifact SHA-256
     hashes, member list and threshold source - so B11.3 ("the OOF fold split has no
     independent signature file") is closed;
  4. records the verification that the active (non-frozen) split file data/splits.json
     still reproduces data/splits_frozen.json's per-split signatures;
  5. answers whether the CV protocol should also be frozen, with the minimal change.

Nothing here modifies data/, split_dataset.py, docs/ or src/ - it only reads them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metrics_common as mc

FROZEN_SPLIT = "data/splits_frozen.json"
ACTIVE_SPLIT = "data/splits.json"
EVAL_SIG_SIMULATION_DRAWS = 2000


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_signature(members) -> str:
    """Same construction as scripts/split_dataset.py::split_signature."""
    return hashlib.sha256("\n".join(sorted(m["sha256"] for m in members)).encode()).hexdigest()


def fold_assignment(y: np.ndarray, n_splits: int = mc.N_FOLDS_DEFAULT, seed: int = mc.OUTER_SEED):
    """Reproduce the exact fold assignment used by evaluate_system.py / metrics_common.py."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.zeros(len(y), dtype=int)
    holdout = {}
    for k, (_, va) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
        fold[va] = k
        holdout[k] = [int(i) + 1 for i in va]
    return fold, holdout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mc.add_common_args(ap)
    ap.add_argument("--frozen-split", default=FROZEN_SPLIT)
    ap.add_argument("--active-split", default=ACTIVE_SPLIT)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out_path = ws / (args.out or "reports/protocol_reconciliation.json")
    bundle = mc.load_or_build_bundle(ws / args.cache, refresh=args.refresh_oof,
                                     workspace=ws, features=args.features)
    a = mc.bundle_arrays(bundle)
    y, seqs = a["y"], a["seqs"]
    fused_cfg = mc.load_deployed_config(ws / "models")

    # ---- 1. the two protocols -----------------------------------------------------
    frozen = json.loads((ws / args.frozen_split).read_text(encoding="utf-8"))
    active = json.loads((ws / args.active_split).read_text(encoding="utf-8"))
    active_sigs = {part: split_signature(active["splits"][part])
                   for part in ("train", "val", "test")}
    sig_match = {part: active_sigs[part] == frozen["signatures"][part]
                 for part in ("train", "val", "test")}
    part_stats = {}
    for part in ("train", "val", "test"):
        rows = active["splits"][part]
        part_stats[part] = {"n": len(rows),
                            "ai": sum(1 for r in rows if r["label"] == "ai"),
                            "real": sum(1 for r in rows if r["label"] == "real")}

    fold, holdout_seq_by_fold = fold_assignment(y)
    bundle_fold_of_seq = {}
    for k, f in enumerate(bundle["folds"]):
        for s in f["holdout_seqs"]:
            bundle_fold_of_seq[int(s)] = k
    recomputed_fold_of_seq = {int(seqs[i]): int(fold[i]) for i in range(len(y))}
    fold_matches = recomputed_fold_of_seq == bundle_fold_of_seq
    fold_vec_sha = hashlib.sha256(
        ",".join(str(int(fold[i])) for i in range(len(y))).encode()).hexdigest()

    # ---- 2. why the headline metrics are out-of-fold -------------------------------
    score, members = mc.fuse_probabilities(a["oof_calibrated"], fused_cfg["weights"])
    thr = float(fused_cfg["thr_high"])
    pred = (score >= thr).astype(int)
    full = mc.metrics_at(y, pred)
    seq_to_idx = {int(s): i for i, s in enumerate(seqs)}
    frozen_split_metrics = {}
    for part in ("train", "val", "test"):
        ii = np.array([seq_to_idx[int(r["seq"])] for r in active["splits"][part]])
        frozen_split_metrics[part] = {
            "n": int(len(ii)), **mc.metrics_at(y[ii], pred[ii]),
            "auc": (mc.auc(y[ii], score[ii]) if len(np.unique(y[ii])) > 1 else None),
            "false_positives_seq": sorted(int(seqs[i]) for i in ii if y[i] == 0 and pred[i] == 1),
            "false_negatives_seq": sorted(int(seqs[i]) for i in ii if y[i] == 1 and pred[i] == 0),
        }
    # variance of a single 24-sample test set drawn with the frozen split's class ratio
    rng = np.random.RandomState(2024)
    n_real_test = part_stats["test"]["real"]
    n_ai_test = part_stats["test"]["ai"]
    real_idx, ai_idx = np.where(y == 0)[0], np.where(y == 1)[0]
    accs, fprs, tprs = [], [], []
    for _ in range(EVAL_SIG_SIMULATION_DRAWS):
        ii = np.concatenate([rng.choice(real_idx, n_real_test, replace=False),
                             rng.choice(ai_idx, n_ai_test, replace=False)])
        m = mc.metrics_at(y[ii], pred[ii])
        accs.append(m["accuracy"]), fprs.append(m["fpr"]), tprs.append(m["tpr"])
    accs, fprs, tprs = np.array(accs), np.array(fprs), np.array(tprs)
    single_split_variance = {
        "rule": f"{EVAL_SIG_SIMULATION_DRAWS} draws of a {n_real_test + n_ai_test}-sample test "
                f"set with the frozen split's class ratio ({n_real_test} real / {n_ai_test} ai), "
                f"scored with the frozen weights and the frozen threshold",
        "accuracy": {"mean": round(float(accs.mean()), 4), "std": round(float(accs.std(ddof=1)), 4),
                     "ci95": [round(float(np.percentile(accs, 2.5)), 4),
                              round(float(np.percentile(accs, 97.5)), 4)],
                     "min": round(float(accs.min()), 4), "max": round(float(accs.max()), 4)},
        "fpr": {"mean": round(float(fprs.mean()), 4), "std": round(float(fprs.std(ddof=1)), 4),
                "ci95": [round(float(np.percentile(fprs, 2.5)), 4),
                         round(float(np.percentile(fprs, 97.5)), 4)]},
        "tpr": {"mean": round(float(tprs.mean()), 4), "std": round(float(tprs.std(ddof=1)), 4),
                "ci95": [round(float(np.percentile(tprs, 2.5)), 4),
                         round(float(np.percentile(tprs, 97.5)), 4)]},
        "share_of_draws_reaching_95pct_accuracy": round(float((accs >= 0.95).mean()), 4),
        "reading": "a single 20% test set of 24 samples produces accuracy readings from "
                   f"{accs.min():.4f} to {accs.max():.4f} (95% interval "
                   f"[{np.percentile(accs, 2.5):.4f}, {np.percentile(accs, 97.5):.4f}]) on the "
                   "same frozen model: it cannot separate the 95% acceptance level from 83%. "
                   "The 5-fold out-of-fold protocol scores every one of the 119 samples "
                   "exactly once and is therefore used for the reported metrics.",
    }

    # ---- 3. reproducible evaluation-protocol signature -----------------------------
    artifacts = {}
    for rel in ("features.csv", "data/labels.csv", "models/l1b_feats_119.npy",
                "models/clip_feats_119.npy", "models/clip_feats_norm_119.npy",
                "models/clip_tiles_119.npy", "models/npr_scores_119.npy",
                "models/clip_probe_norm.pt", "models/npr.pt", "models/fusion.json",
                "repo/scripts/evaluate_system.py", "repo/scripts/metrics_common.py",
                "repo/scripts/calibration_analysis.py", "repo/scripts/member_analysis.py",
                "repo/scripts/threshold_stability.py", "repo/scripts/resource_profile.py",
                "repo/scripts/stratified_analysis.py",
                FROZEN_SPLIT, ACTIVE_SPLIT):
        artifacts[rel] = sha256_file(ws / rel)
    signature_core = {
        "cv": {"n_folds": mc.N_FOLDS_DEFAULT, "shuffle": True, "random_state": mc.OUTER_SEED,
               "stratification_key": "label (ai/real)",
               "implementation": "sklearn.model_selection.StratifiedKFold over all 119 samples",
               "inner": {"n_folds": 4,
                         "random_state": "100 + outer fold",
                         "purpose": "unbiased in-training score scale for the per-member "
                                    "temperature scalers and the fold thresholds"}},
        "fold_assignment_sha256": fold_vec_sha,
        "fold_assignment_by_seq": {int(seqs[i]): int(fold[i]) for i in range(len(y))},
        "holdout_seqs_by_fold": {str(k): v for k, v in holdout_seq_by_fold.items()},
        "members": sorted(fused_cfg["weights"].keys()),
        "member_weights": fused_cfg["weights"],
        "threshold_source": {"rule": "order statistic on the training-fold REAL scores "
                                     "(fpr_capped_threshold)",
                             "target_fpr": 0.03, "absolute_fp_budget": mc.FP_BUDGET_DEFAULT,
                             "frozen_thresholds": {"thr_low": fused_cfg["thr_low"],
                                                   "thr_high": fused_cfg["thr_high"]},
                             "source": fused_cfg["source"]},
        "external_member_admission": {"rule": "clip_external joins a fold only if its "
                                               "training-fold AUC >= 0.65",
                                      "threshold": mc.EXT_MIN_AUC},
        "artifacts_sha256": artifacts,
        "oof_cache": {"path": args.cache, "sha256": sha256_file(ws / args.cache)},
        "headline_metrics_on_this_signature": full,
    }
    protocol_sha = hashlib.sha256(
        json.dumps(signature_core, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    payload = {
        "report": "protocol_reconciliation",
        "manual_sections": ["10.1–10.3 数据划分与泄漏防护", "D1–D8（data/splits.json.protocol）",
                            "13.1 阈值标定（val 只用于标定）",
                            "15.2 验收标准的验证方式（多次随机划分的稳定性验证）",
                            "17.3 数据划分冻结"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "repo/scripts/protocol_reconciliation.py",
        "audit_reference": "docs/compliance-audit.md F3c / F3 / A6.5 / B3.8 / B11.3",
        "question": "冻结划分（0.6/0.2/0.2 单次切分）与最终评测协议（119 张分层 5 折折外）"
                    "不是同一套协议，审计判为「部分符合」。本文件给出两者的关系、评测协议的"
                    "可复现签名，以及建议的最小改动。",
        "two_protocols": {
            "frozen_split": {
                "file": args.frozen_split, "status": "frozen",
                "created": frozen.get("created"), "seed": frozen.get("seed"),
                "ratios": frozen.get("ratios"),
                "signatures": frozen.get("signatures"),
                "signature_construction": "sha256 of the newline-joined, sorted member "
                                          "sha256 values (scripts/split_dataset.py::split_signature)",
                "verification": {
                    "method": "recomputed from data/splits.json's member sha256 lists using the "
                              "same construction",
                    "per_part_match": sig_match,
                    "all_match": all(sig_match.values()),
                },
                "part_sizes": part_stats,
                "governance": active.get("protocol"),
                "leakage_protection": {"near_duplicate_groups": active.get("groups"),
                                       "phash": active.get("phash")},
                "purpose": "项目期数据划分规范与 D1–D8 纪律（去重并组、同源组不跨划分、"
                           "val 只用于阈值/权重/模型选择、test 禁用于搜索）；"
                           "用于演示与审计，不作为最终指标的计算协议",
                "not_used_for": ["最终指标（reports/metrics.json 的 oof_overall）",
                                 "本次指标包的任何数字（calibration/member/threshold/"
                                 "resource/stratified）"],
            },
            "out_of_fold_cv": {
                "files": ["repo/scripts/evaluate_system.py",
                          "repo/scripts/metrics_common.py",
                          "repo/scripts/calibration_analysis.py",
                          "repo/scripts/member_analysis.py",
                          "repo/scripts/threshold_stability.py"],
                "used_for": "最终指标与全部诊断指标（本指标包）",
                "headline_metrics": full,
                "no_selection_on_held_out_folds": True,
                "protocol_sha256": protocol_sha,
            },
        },
        "authority_relationship": {
            "both_anchor_to": {"sample_set": "data/labels.csv（119 张，data/images/*.png）",
                               "seed": 42,
                               "contents_hash": "data/manifest.json 的 sha256 清单"},
            "difference": "划分单元不同：冻结划分是「一次三方切分」（71/24/24）；"
                          "评测协议是「5 折轮转折外」（每张样本恰好被 hold out 一次，n=119）",
            "how_D8_is_satisfied": "D8 的实质是「最终报告只使用冻结的模型与阈值，且不得在 "
                                   "test 上搜索」。模型/权重/阈值冻结于 models/fusion.json"
                                   "（本次修补未改动）；折外协议比冻结划分满足更强的纪律："
                                   "任何样本都不参与自己的打分，也不参与自己所在折的阈值选择，"
                                   "因而不存在 test 泄漏的可能",
            "residual_gap_closed_by_this_file": "此前折切分没有独立签名文件（审计 B11.3）；"
                                                "本文件给出 fold_assignment_sha256 + 特征/产物 "
                                                "SHA-256 + protocol_sha256",
            "must_not_be_confused": "冻结划分上的读数（下方 frozen_split_metrics）仅作参考，"
                                    "不是验收指标；它们由同一批折外分数与冻结阈值计算，"
                                    "样本量小（train 71 / val 24 / test 24）",
        },
        "why_out_of_fold_for_the_headline": {
            "reason": "119 张样本下 20% 测试集只有 24 张，指标方差过大；手册 15.2 也要求"
                      "「多次随机划分的稳定性验证」，折外协议即其实现",
            "measured_single_split_variance": single_split_variance,
            "frozen_split_metrics_reference_only": frozen_split_metrics,
            "full_set_out_of_fold": full,
        },
        "evaluation_protocol_signature": {"protocol_sha256": protocol_sha, **signature_core},
        "reproducibility_checks": {
            "fold_assignment_recomputed_matches_bundle": bool(fold_matches),
            "fold_assignment_sha256": fold_vec_sha,
            "frozen_split_signatures_reproduce": bool(all(sig_match.values())),
            "notes": ["折切分只依赖 (样本顺序, 标签, n_folds=5, seed=42)，因此可在任意机器上"
                      "逐位复现；上面的 sha256 是逐样本折号向量的指纹",
                      "artifacts_sha256 覆盖特征文件、探针/网络权重、融合配置与全部评测脚本，"
                      "可用于确认「同一签名 = 同一批输入」"],
        },
        "recommendation": {
            "should_the_cv_protocol_also_be_frozen": True,
            "conclusion": "建议把评测协议一并冻结；但不要改动 data/splits_frozen.json 的既有"
                          "字段——它的 train/val/test 签名是 split_dataset.py --verify-frozen "
                          "的校验对象，改动该文件会使既有冻结签名与文件内容不一致。",
            "minimal_change": {
                "option_a_recommended": {
                    "action": "新增 data/eval_protocol_frozen.json（不修改既有冻结文件）",
                    "content_source": "本文件 evaluation_protocol_signature 块可直接作为初版内容"
                                      "（去掉 artifacts_sha256 中与产出无关的项即可）",
                    "why": "零侵入：既满足 D8/17.3「划分与协议冻结」，又保持 splits_frozen.json "
                           "的签名校验有效",
                    "verify_command": "python -c \"import json,hashlib,sys;d=json.load(open('data/"
                                      "eval_protocol_frozen.json',encoding='utf-8'));"
                                      "core={k:v for k,v in d.items() if k!='protocol_sha256'};"
                                      "print(hashlib.sha256(json.dumps(core,sort_keys=True,"
                                      "ensure_ascii=False).encode()).hexdigest())\" "
                                      "  # 与文件内 protocol_sha256 比对",
                    "owner": "scripts/split_dataset.py 的负责人（data/ 与划分脚本不在本指标包"
                             "的改动范围内）",
                },
                "option_b_not_recommended": {
                    "action": "在 data/splits_frozen.json 顶层增加 evaluation_protocol 键",
                    "why_not": "会改变此前被引用/记录的冻结文件内容与其 sha256；需要同时更新"
                               "所有引用该哈希的位置，收益与风险不成比例",
                },
            },
            "immutable_parts_of_the_protocol": [
                "119 张样本清单与标签", "n_folds=5、shuffle、random_state=42、分层键=label",
                "逐折 clip_external 准入规则（train-fold AUC ≥ 0.65）",
                "阈值来源（训练折真实分数的顺序统计量，目标 FPR 3%、绝对预算 2 个误报）",
                "内层 4 折与 random_state=100+fold（温度标定的无偏分数尺度）",
            ],
        },
        "handoff": [
            "docs/results.md §3 需说明「最终验收采用折外协议而非冻结划分」的理由与二者关系"
            "（该文件属 docs-verifier，本指标包只提供证据）",
            "若采纳 option_a，需由 data/ 负责人落盘 data/eval_protocol_frozen.json 并在 "
            "docs/reproduce.md 增加校验命令",
        ],
        "limitations": [
            "frozen_split_metrics_reference_only 中的三段读数用「单一冻结阈值」计算以保持子集"
            "可比；部署路径实际使用逐折阈值，因此这些数字与 reports/metrics.json 的 "
            "oof_overall 不可直接比较。",
            "本文件只调和协议关系，不改变任何已发布的指标：metrics.json 与指标包全部沿用"
            "分层 5 折折外协议。",
            "冻结划分签名只证明「活跃划分文件与冻结文件的成员集合一致」，不证明划分算法本身"
            "无泄漏；泄漏防护由 D4/D5 的并组断言与 groups.multi_member=0 提供。",
        ],
    }
    mc.write_json(out_path, payload)
    print(f"[protocol] fold assignment matches bundle: {fold_matches}; frozen signatures "
          f"reproduce: {all(sig_match.values())}; protocol_sha256={protocol_sha[:16]}...")
    print(f"[protocol] single-24-sample test set accuracy std="
          f"{single_split_variance['accuracy']['std']} "
          f"ci95={single_split_variance['accuracy']['ci95']}")


if __name__ == "__main__":
    main()
