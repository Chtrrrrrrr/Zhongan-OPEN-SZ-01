#!/usr/bin/env python3
"""批处理一键评测（手册 16.1「批处理脚本用于 100 套样本一键评测」）。

对任意目录/清单批量跑检测流水线，输出:
  * 指标表：Accuracy / FPR / TPR / Precision / 混淆矩阵（另附 decided-only 与三档分布）
  * 失败案例清单：FP / FN / review(弃权) / 解码失败，含首因说明
  * 逐样本报告落盘：<out>/samples/<name>.json（可选 .html）
  * 机器可读汇总：<out>/metrics.json + <out>/metrics.md + <out>/failures.csv

用法:
  python scripts/batch_evaluate.py --dir data/images --labels data/labels.csv \\
      --out reports/batch/run1
  python scripts/batch_evaluate.py --dir data/images --labels data/labels.csv --limit 20
  python scripts/batch_evaluate.py --dir demo_assets --ext mp4 --video
  python scripts/batch_evaluate.py --selftest          # 合成样本自测（不需要模型/数据）
  python scripts/batch_evaluate.py --dir data/images --no-model --limit 4   # 占位模式冒烟

标签来源（按优先级）:
  1) --labels <csv/txt>：列 seq,filename,label 或 filename,label（自动识别表头）
  2) --label-dirs：目录名即标签（如 --label-dirs ai=ai,real=real；默认识别 ai/real 同义词）
  3) 无标签：只输出结论分布与逐样本报告，不计算监督指标
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # <repo>
sys.path.insert(0, str(ROOT / "src"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

POS_WORDS = {"ai", "aigc", "fake", "generated", "gen", "synthetic", "synth", "1", "true",
             "yes", "y", "t", "positive", "pos", "是", "假", "生成"}
NEG_WORDS = {"real", "genuine", "natural", "camera", "photo", "authentic", "0", "false",
             "no", "n", "f", "negative", "neg", "否", "真", "真实"}
FAILURE_ORDER = ["false_positive", "false_negative", "review", "decode_error", "error"]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_label(raw: str) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if s in POS_WORDS:
        return "ai"
    if s in NEG_WORDS:
        return "real"
    # 目录名/文件名的模糊匹配（ai_generated / real_photo / fake-2 …）
    for w in ("generated", "synthetic", "fake", "ai"):
        if w in s:
            return "ai"
    for w in ("real", "genuine", "camera", "natural"):
        if w in s:
            return "real"
    return None


def read_metric_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _has_artifacts(d: Path) -> bool:
    if not d.is_dir():
        return False
    for name in ("fusion.json", "clip_probe_norm.pt", "clip_probe.pt", "forensic_lr.joblib",
                 "combined_lr.joblib", "clip_domain_lr.joblib", "l1_lr.joblib", "npr.pt"):
        if (d / name).exists():
            return True
    if (d / "hf_cache").is_dir():
        return True
    return any(p.suffix in (".pt", ".joblib", ".pth") for p in d.glob("*") if p.is_file())


def find_models_dir(explicit: str | None) -> Path | None:
    cands = [Path(explicit)] if explicit else []
    cands += [ROOT / "models", ROOT.parent / "models", Path.cwd() / "models"]
    uniq, seen = [], set()
    for c in cands:
        try:
            key = str(c.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    for c in uniq:
        if _has_artifacts(c):
            return c
    for c in uniq:
        if c.is_dir():
            return c
    return None


def resolve_clip_dir(base: Path | None) -> Path | None:
    if base is None or not base.is_dir():
        return None
    if (base / "config.json").exists():
        return base
    subs = [d for d in base.iterdir() if d.is_dir()]
    for d in subs:
        if (d / "config.json").exists():
            return d
    return subs[0] if subs else None


# --------------------------------------------------------------------------- #
# 标签解析
# --------------------------------------------------------------------------- #
def load_labels_file(path: Path) -> dict:
    """返回 {'by_name': {...}, 'by_stem': {...}, 'by_seq': {...}}。"""
    out = {"by_name": {}, "by_stem": {}, "by_seq": {}, "source": str(path), "rows": 0}
    if not path.is_file():
        raise SystemExit(f"[error] 标签文件不存在：{path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return out
    delim = "," if ("," in lines[0]) else None
    rows = [ln.split(delim) if delim else ln.split() for ln in lines]
    rows = [[c.strip().strip('"') for c in r] for r in rows]
    header = [c.lower() for c in rows[0]]
    has_header = any(c in ("label", "filename", "file", "path", "seq", "id") for c in header)
    idx = {}
    if has_header:
        for i, c in enumerate(header):
            if c in ("seq", "id", "index"):
                idx["seq"] = i
            elif c in ("filename", "file", "path", "name"):
                idx["name"] = i
            elif c in ("label", "target", "y", "class"):
                idx["label"] = i
        rows = rows[1:]
    else:
        if len(rows[0]) >= 3:
            idx = {"seq": 0, "name": 1, "label": 2}
        elif len(rows[0]) == 2:
            idx = {"name": 0, "label": 1}
        else:
            raise SystemExit(f"[error] 无法识别标签列：{lines[0]!r}")
    if "label" not in idx:
        raise SystemExit(f"[error] 标签文件缺少 label 列：{lines[0]!r}")
    for r in rows:
        if len(r) <= idx["label"]:
            continue
        lab = normalize_label(r[idx["label"]])
        if lab is None:
            continue
        if "name" in idx and idx["name"] < len(r):
            raw = r[idx["name"]].replace("\\", "/")
            out["by_name"][Path(raw).name.lower()] = lab
            out["by_stem"][Path(raw).stem.lower()] = lab
        if "seq" in idx and idx["seq"] < len(r):
            try:
                out["by_seq"][int(float(r[idx["seq"]]))] = lab
            except ValueError:
                pass
        out["rows"] += 1
    return out


def labels_from_dirs(files: list[Path], root: Path, mapping: dict[str, str]) -> dict:
    out = {"by_name": {}, "by_stem": {}, "by_seq": {}, "source": "directory-names", "rows": 0}
    for p in files:
        rel = None
        for cand in (root, root.parent):
            try:
                rel = p.resolve().relative_to(cand.resolve())
                break
            except ValueError:
                continue
        if rel is None:
            continue
        parts = [x.lower() for x in rel.parts[:-1]]
        # 优先用自定义映射，其次用同义词识别
        lab = None
        for part in parts:
            for sub, mapped in mapping.items():
                if sub in part:
                    lab = mapped
                    break
            if lab:
                break
            lab = normalize_label(part)
            if lab and part not in ("data", "images", "img", "samples", "test", "val", "train"):
                break
            lab = None
        if lab:
            out["by_name"][p.name.lower()] = lab
            out["by_stem"][p.stem.lower()] = lab
            out["rows"] += 1
    return out


def resolve_label(entry: dict, labels: list[dict]) -> str | None:
    for src in labels:
        for key, val in (("by_name", entry["name"].lower()), ("by_stem", entry["stem"].lower()),
                         ("by_seq", entry["seq"])):
            if val in src.get(key, {}):
                return src[key][val]
    return None


def discover_files(roots: list[Path], exts: set[str], recursive: bool, limit: int | None) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        if r.is_file():
            files.append(r)
            continue
        if not r.is_dir():
            print(f"[warn] 跳过不存在的输入：{r}", file=sys.stderr)
            continue
        it = r.rglob("*") if recursive else r.glob("*")
        files += [p for p in sorted(it) if p.is_file() and p.suffix.lower() in exts]
    uniq: list[Path] = []
    seen = set()
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if limit:
        uniq = uniq[:limit]
    return uniq


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def confusion(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """混淆矩阵与派生指标。分母为 0 时返回 None（不可评估），而不是伪造成 0。"""
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n = int(len(y_true))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": n, "n_pos_ai": n_pos, "n_neg_real": n_neg,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "fpr": round(fp / n_neg, 4) if n_neg else None,
        "tpr": round(tp / n_pos, 4) if n_pos else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / n_pos, 4) if n_pos else None,
        "specificity": round(tn / n_neg, 4) if n_neg else None,
        "fnr": round(fn / n_pos, 4) if n_pos else None,
        "note": ("样本只有单一类别：FPR/特异性或 TPR/AUC 不可评估，请同时提供 ai 与 real 标签"
                 if not (n_pos and n_neg) else None),
    }


def bootstrap_ci(y_true: np.ndarray, pred: np.ndarray, draws: int = 2000, seed: int = 7) -> dict:
    if len(y_true) == 0:
        return {"accuracy_ci95": None, "fpr_ci95": None}
    rng = np.random.default_rng(seed)
    accs, fprs = [], []
    n = len(y_true)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        yt, pd = y_true[idx], pred[idx]
        accs.append(float((yt == pd).mean()))
        neg = int((yt == 0).sum())
        if neg:
            fprs.append(float((pd[yt == 0] == 1).mean()))
    out = {
        "accuracy_ci95": [round(float(np.percentile(accs, 2.5)), 4),
                          round(float(np.percentile(accs, 97.5)), 4)],
        "fpr_ci95": ([round(float(np.percentile(fprs, 2.5)), 4),
                      round(float(np.percentile(fprs, 97.5)), 4)] if fprs else None),
        "draws": draws, "seed": seed,
    }
    return out


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score      # noqa: PLC0415
        return round(float(roc_auc_score(y_true, score)), 4)
    except Exception:  # noqa: BLE001 - 无 sklearn 时用秩统计
        order = np.argsort(score)
        ranks = np.empty(len(score), dtype=np.float64)
        ranks[order] = np.arange(1, len(score) + 1)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if not n_pos or not n_neg:
            return None
        return round(float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 4)


def percentile(values: list[float], q: float) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None
    return round(float(np.percentile(vals, q)), 1)


# --------------------------------------------------------------------------- #
# 检测器
# --------------------------------------------------------------------------- #
def build_detector(args, models_dir: Path | None):
    """返回 (detector_or_none, pipeline_modules_or_none, note)。"""
    if args.no_model:
        return None, None, "--no-model：占位模式（不出结论，只验证批处理链路）"
    try:
        from detector import AIGCDetector, PipelineConfig        # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return None, None, f"无法导入 src/detector.py：{type(e).__name__}: {e}"

    def opt(*rel):
        if models_dir is None:
            return None
        for r in rel:
            p = models_dir / r
            if p.exists():
                return p
        return None

    clip = resolve_clip_dir(opt("hf_cache/models--openai--clip-vit-large-patch14/snapshots",
                                "clip-vit-large-patch14"))
    probe = opt("clip_probe_norm.pt", "clip_probe.pt")
    forensic = opt("forensic_lr.joblib")
    fusion = opt("fusion.json")
    if not any([clip, probe, forensic, fusion]):
        return None, None, f"未找到模型文件（models_dir={models_dir}），占位模式"
    cfg = PipelineConfig(
        clip_dir=str(clip) if clip else None,
        clip_probe=str(probe) if probe else None,
        l1_lr_model=str(opt("l1_lr.joblib")) if opt("l1_lr.joblib") else None,
        forensic_lr=str(forensic) if forensic else None,
        clip_domain_lr=str(opt("clip_domain_lr.joblib")) if opt("clip_domain_lr.joblib") else None,
        combined_lr=str(opt("combined_lr.joblib")) if opt("combined_lr.joblib") else None,
        npr_weights=str(opt("npr.pt")) if (args.enable_npr and opt("npr.pt")) else None,
        fusion_json=str(fusion) if fusion else None,
        enable_aeroblade=not args.disable_aeroblade,
        time_budget_s=args.time_budget,
        device=args.device,
    )
    note = f"device={cfg.device or 'auto'} clip={'yes' if clip else 'no'} " \
           f"forensic={'yes' if forensic else 'no'} fusion={'yes' if fusion else 'no'}"
    try:
        det = AIGCDetector(cfg)
        return det, {"PipelineConfig": PipelineConfig}, note
    except Exception as e:  # noqa: BLE001
        return None, None, f"检测器构建失败：{type(e).__name__}: {e}"


def warm_up(det, sample: Path | None) -> dict:
    if det is None or sample is None:
        return {"done": False, "ms": None, "sample": str(sample) if sample else None}
    t0 = time.perf_counter()
    try:
        det.detect_image(str(sample))
        return {"done": True, "ms": round((time.perf_counter() - t0) * 1000, 1), "sample": str(sample)}
    except Exception as e:  # noqa: BLE001
        return {"done": False, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "sample": str(sample), "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_batch(args) -> dict:
    t_start = time.perf_counter()
    out_dir = Path(args.out)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    exts = set()
    for token in (args.ext or "").split(","):
        token = token.strip().lower()
        if token:
            exts.add(token if token.startswith(".") else "." + token)
    if not exts:
        exts = set(VIDEO_EXTS) if args.video else set(IMAGE_EXTS)
    if args.video:
        exts |= VIDEO_EXTS

    roots = [Path(p) for p in (args.dir or [])] + [Path(p) for p in (args.file or [])]
    if args.list:
        lp = Path(args.list)
        if not lp.is_file():
            raise SystemExit(f"[error] 清单文件不存在：{lp}")
        roots += [Path(ln.strip()) for ln in lp.read_text(encoding="utf-8-sig").splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    if not roots:
        raise SystemExit("[error] 请用 --dir / --file / --list 指定待评测输入")

    files = discover_files(roots, exts, args.recursive, args.limit)
    if not files:
        raise SystemExit(f"[error] 未发现可评测文件（扩展名={sorted(exts)}）")

    label_sources = []
    if args.labels:
        label_sources.append(load_labels_file(Path(args.labels)))
    mapping = {}
    for item in (args.label_dirs or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            sub, lab = item.split("=", 1)
            mapped = normalize_label(lab)
            if mapped:
                mapping[sub.strip().lower()] = mapped
    if args.label_dirs or args.infer_labels:
        label_sources.append(labels_from_dirs(files, roots[0], mapping))

    entries = []
    for i, p in enumerate(files, start=1):
        e = {"seq": i, "path": p, "name": p.name, "stem": p.stem,
             "kind": "video" if p.suffix.lower() in VIDEO_EXTS else "image"}
        e["label"] = resolve_label(e, label_sources) if label_sources else None
        entries.append(e)
    labelled = [e for e in entries if e["label"]]
    print(f"[batch] 输入 {len(entries)} 个文件，其中带标签 {len(labelled)} 个；"
          f"输出目录 {out_dir}", flush=True)

    det, _mods, note = build_detector(args, find_models_dir(args.models_dir))
    print(f"[batch] 检测器：{note}", flush=True)
    warm_sample = next((e["path"] for e in entries if e["kind"] == "image"), None)
    warm = warm_up(det, warm_sample)
    if warm.get("done"):
        print(f"[batch] 预热 {warm['ms']} ms（{Path(warm['sample']).name}）", flush=True)

    records = []
    for e in entries:
        rec = {"seq": e["seq"], "file": str(e["path"]), "name": e["name"], "kind": e["kind"],
               "label": e["label"], "verdict": None, "verdict_cn": None, "confidence": None,
               "fused_score": None, "disagreement": None, "model_scores": {},
               "latency_ms": {}, "uncertainty": [], "status": "ok", "error": None,
               "report": None, "failure_type": None, "reason": None}
        t0 = time.perf_counter()
        if det is None:
            rec["verdict"] = "review"
            rec["verdict_cn"] = "待复核"
            rec["status"] = "stub"
            rec["uncertainty"] = [note]
        else:
            try:
                if e["kind"] == "video":
                    rep = det.detect_video(str(e["path"]))
                else:
                    rep, _ = det.detect_image(str(e["path"]))
                rec.update({
                    "verdict": rep.verdict, "verdict_cn": rep.verdict_cn,
                    "confidence": rep.confidence, "fused_score": rep.fused_score,
                    "disagreement": rep.disagreement, "model_scores": rep.model_scores,
                    "latency_ms": dict(rep.latency_ms or {}),
                    "uncertainty": list(rep.uncertainty or []),
                    "input_hash": rep.input_hash, "resolution": rep.resolution,
                })
                rec["report"] = rep
            except Exception as ex:  # noqa: BLE001 - 单样本失败不中断批量
                rec["status"] = "error"
                rec["verdict"] = "review"
                rec["verdict_cn"] = "待复核"
                rec["error"] = f"{type(ex).__name__}: {ex}"
        rec["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["latency_ms"] = {**(rec["latency_ms"] or {}), "batch_wall": rec["wall_ms"]}

        if rec["verdict"] not in ("normal", "review", "suspect"):
            rec["status"] = "error"
            rec["verdict"] = "review"

        # 失败分类（有标签时按混淆矩阵语义，无标签时只标 review/error）
        if rec["status"] == "error":
            rec["failure_type"] = "decode_error" if rec["error"] else "error"
        elif rec["label"] == "real" and rec["verdict"] == "suspect":
            rec["failure_type"] = "false_positive"
        elif rec["label"] == "ai" and rec["verdict"] == "normal":
            rec["failure_type"] = "false_negative"
        elif rec["verdict"] == "review":
            rec["failure_type"] = "review"

        # 逐样本报告落盘
        try:
            if rec["report"] is not None:
                rec["report"].save_json(str(samples_dir / f"{e['path'].stem}.json"))
                if args.html and rec["failure_type"] in ("false_positive", "false_negative"):
                    from report import save_html                      # noqa: PLC0415
                    save_html(rec["report"], str(samples_dir / f"{e['path'].stem}.html"))
            else:
                payload = {k: v for k, v in rec.items() if k != "report"}
                payload["report_schema"] = "batch-stub/1.0"
                (samples_dir / f"{e['path'].stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as ex:  # noqa: BLE001
            print(f"[warn] 报告落盘失败 {e['name']}：{ex}", file=sys.stderr)
        rec["report"] = None
        records.append(rec)
        if args.progress_every and len(records) % args.progress_every == 0:
            print(f"[batch] {len(records)}/{len(entries)} …", flush=True)

    metrics = aggregate_metrics(records, labelled)
    failures = build_failures(records)
    metrics.update({
        "generated_at": now_iso(),
        "inputs": [str(p) for p in roots],
        "n_files": len(entries),
        "n_labelled": len(labelled),
        "detector": note,
        "warmup": warm,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "protocol": {
            "positive_class": "ai（AI 生成/篡改）",
            "negative_class": "real（真实拍摄）",
            "three_tier": "normal→real，suspect→ai，review→弃权（另计覆盖率与保守下界）",
            "note": "本脚本仅做评测聚合，不训练/不选阈值；阈值来自模型侧冻结配置",
        },
        "out_dir": str(out_dir),
        "models_dir": str(find_models_dir(args.models_dir)),
    })

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    write_failures_csv(out_dir / "failures.csv", failures)
    (out_dir / "per_sample.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "metrics.md").write_text(render_markdown(metrics, failures, records),
                                        encoding="utf-8")
    print_table(metrics, failures)
    print(f"[batch] 完成：{out_dir}/metrics.json, metrics.md, failures.csv, "
          f"per_sample.json, samples/*.json（{len(records)} 份）", flush=True)
    return metrics


def aggregate_metrics(records: list[dict], labelled: list[dict]) -> dict:
    verdict_counts = {"normal": 0, "review": 0, "suspect": 0}
    for r in records:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1
    lat_total = [r["latency_ms"].get("total") for r in records if r["latency_ms"]]
    out: dict = {
        "n_records": len(records),
        "n_labelled": len(labelled),
        "statuses": {
            "ok": sum(1 for r in records if r["status"] == "ok"),
            "stub": sum(1 for r in records if r["status"] == "stub"),
            "error": sum(1 for r in records if r["status"] == "error"),
        },
        "three_tier": verdict_counts,
        "latency_ms": {
            "p50": percentile(lat_total, 50), "p95": percentile(lat_total, 95),
            "max": percentile(lat_total, 100),
            "mean_wall": round(statistics.fmean([r["wall_ms"] for r in records]), 1) if records else None,
        },
    }
    per_model: dict[str, list[float]] = {}
    for r in records:
        for k, v in (r.get("model_scores") or {}).items():
            per_model.setdefault(k, []).append(float(v))
    out["per_channel_mean_score"] = {k: round(float(np.mean(v)), 4)
                                    for k, v in sorted(per_model.items()) if v}
    out["per_channel_n"] = {k: len(v) for k, v in sorted(per_model.items())}
    causes = {}
    for r in records:
        if r["verdict"] != "review":
            continue
        c = review_cause(r.get("uncertainty"))
        causes[c] = causes.get(c, 0) + 1
    out["review_causes"] = causes

    if not labelled or not any(r["label"] for r in records):
        out["supervised"] = None
        out["note"] = "未提供标签：仅输出结论分布/耗时/逐样本报告，不计算 Accuracy/FPR/TPR"
        return out

    y = np.array([1 if r["label"] == "ai" else 0 for r in records])
    pred_bin = np.array([1 if r["verdict"] == "suspect" else 0 for r in records])   # review→real
    decided_mask = np.array([r["verdict"] != "review" for r in records])
    score = np.array([float(r["confidence"] or 0.0) for r in records])

    sup = {
        "binary_review_as_negative": confusion(y, pred_bin),
        "binary_review_as_negative_ci95": bootstrap_ci(y, pred_bin),
        "auc_confidence": roc_auc(y, score),
        "coverage": round(float(decided_mask.mean()), 4),
        "coverage_n": int(decided_mask.sum()),
        "abstain_review_n": int((~decided_mask).sum()),
    }
    if decided_mask.sum() >= 2 and len(set(y[decided_mask].tolist())) == 2:
        sup["decided_only"] = confusion(y[decided_mask], pred_bin[decided_mask])
    else:
        sup["decided_only"] = None
    # 保守下界：把弃权样本按最不利方向计（real 弃权算误报、ai 弃权算漏报）
    worst = []
    for r in records:
        if r["verdict"] == "suspect":
            worst.append(1)
        elif r["verdict"] == "normal":
            worst.append(0)
        else:
            worst.append(1 if r["label"] == "real" else 0)
    sup["accuracy_conservative_review_as_error"] = round(
        float((np.array(worst) == y).mean()), 4)
    sup["review_breakdown"] = {
        "real_abstain": sum(1 for r in records if r["verdict"] == "review" and r["label"] == "real"),
        "ai_abstain": sum(1 for r in records if r["verdict"] == "review" and r["label"] == "ai"),
    }
    if "oof_overall" in (read_metric_json(ROOT / "reports" / "metrics.json") or {}):
        ref = read_metric_json(ROOT / "reports" / "metrics.json")["oof_overall"]
        sup["reference_offline_oof"] = {"accuracy": ref.get("accuracy"), "fpr": ref.get("fpr"),
                                        "tpr": ref.get("tpr"),
                                        "source": "reports/metrics.json#oof_overall"}
    out["supervised"] = sup
    return out


REASON_KEYS = ("fused score", "model disagreement", "超时", "无可用", "无法解码", "失败",
               "不可用")   # 判定/降级类说明优先于解释分支的提示语


def pick_reason(uncertainty: list | None) -> str:
    """从 uncertainty 中挑出「决策/降级」首因，而不是解释分支的提示语。"""
    items = [str(u) for u in (uncertainty or []) if u]
    for key in REASON_KEYS[:2]:
        for u in items:
            if key in u:
                return u
    for key in REASON_KEYS[2:]:
        for u in items:
            if key in u:
                return u
    return items[0] if items else ""


def review_cause(uncertainty: list | None) -> str:
    """review（弃权）的成因归类：不确定带 / 模型分歧 / 其它。"""
    text = " ".join(str(u) for u in (uncertainty or []))
    if "disagreement" in text or "分歧" in text:
        return "model_disagreement"
    if "uncertainty band" in text or "不确定带" in text:
        return "uncertainty_band"
    return "other"


def build_failures(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        if not r["failure_type"]:
            continue
        reason = r["error"] or pick_reason(r.get("uncertainty"))
        out.append({
            "seq": r["seq"], "file": r["file"], "name": r["name"], "kind": r["kind"],
            "label": r["label"], "verdict": r["verdict"], "confidence": r["confidence"],
            "fused_score": r["fused_score"], "failure_type": r["failure_type"],
            "reason": reason,
        })
    out.sort(key=lambda x: (FAILURE_ORDER.index(x["failure_type"])
                            if x["failure_type"] in FAILURE_ORDER else 99, x["seq"]))
    return out


def write_failures_csv(path: Path, failures: list[dict]):
    fields = ["seq", "file", "name", "kind", "label", "verdict", "confidence",
              "fused_score", "failure_type", "reason"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in failures:
            w.writerow(row)


def render_markdown(metrics: dict, failures: list[dict], records: list[dict]) -> str:
    def f(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "不可评估（缺样本/缺标签）"

    sup = metrics.get("supervised")
    lines = ["# 批量评测报告", "",
             f"- 生成时间：{metrics['generated_at']}",
             f"- 样本数：{metrics['n_files']}（带标签 {metrics['n_labelled']}）",
             f"- 检测器：{metrics['detector']}",
             f"- 预热：{metrics['warmup']}",
             f"- 总耗时：{metrics['elapsed_s']} s",
             f"- 输入：{', '.join(metrics['inputs'])}", ""]
    lines += ["## 指标表", ""]
    if sup:
        b = sup["binary_review_as_negative"]
        lines += ["| 指标 | 值 |", "| --- | --- |",
                  f"| Accuracy（review→real，与 reports/metrics.json 同口径） | {f(b['accuracy'])} |",
                  f"| Accuracy 95% CI | {sup['binary_review_as_negative_ci95']['accuracy_ci95']} |",
                  f"| FPR | {f(b['fpr'])} |",
                  f"| FPR 95% CI | {sup['binary_review_as_negative_ci95']['fpr_ci95']} |",
                  f"| TPR（Recall） | {f(b['tpr'])} |",
                  f"| Precision | {f(b['precision'])} |",
                  f"| AUC（连续置信度） | {f(sup['auc_confidence'])} |",
                  f"| 覆盖率（非弃权占比） | {f(sup['coverage'])}（{sup['coverage_n']}/{metrics['n_records']}） |",
                  f"| 保守下界（弃权视为错误） | {f(sup['accuracy_conservative_review_as_error'])} |",
                  f"| 混淆矩阵 | TP={b['tp']} FP={b['fp']} FN={b['fn']} TN={b['tn']}"
                  f"（AI={b['n_pos_ai']} / real={b['n_neg_real']}） |", ""]
        if b.get("note"):
            lines += [f"> 注意：{b['note']}", ""]
        if sup.get("decided_only"):
            d = sup["decided_only"]
            lines += [f"仅在已决样本上（{metrics['n_records'] - sup['abstain_review_n']} 条）："
                      f"Accuracy={f(d['accuracy'])} / FPR={f(d['fpr'])} / TPR={f(d['tpr'])} / "
                      f"TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']}", ""]
    else:
        lines += ["> 未提供标签：不计算监督指标（Accuracy/FPR/TPR）。", ""]
    tt = metrics["three_tier"]
    lines += ["## 三档结论分布", "",
              f"- suspect（疑似生成/篡改）：{tt['suspect']}",
              f"- normal（正常）：{tt['normal']}",
              f"- review（待复核）：{tt['review']}"
              f"（成因：{metrics.get('review_causes') or '—'}）", ""]
    lines += ["## 失败案例清单", "",
              f"共 {len(failures)} 条。", ""]
    if failures:
        lines += ["| # | 文件 | 标签 | 结论 | 置信度 | 类型 | 首因 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in failures[:100]:
            lines.append(f"| {row['seq']} | {row['name']} | {row['label']} | {row['verdict']} | "
                         f"{row['confidence']} | {row['failure_type']} | "
                         f"{(row['reason'] or '')[:80]} |")
    lines += ["", "## 逐样本报告", "",
              f"逐样本 JSON：`samples/<name>.json`（{len(records)} 份）；"
              f"逐样本明细：`per_sample.json`。", ""]
    return "\n".join(lines)


def print_table(metrics: dict, failures: list[dict]):
    def f(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
    tt = metrics["three_tier"]
    print("\n" + "=" * 72)
    print(f"批量评测：{metrics['n_files']} 样本（带标签 {metrics['n_labelled']}）  "
          f"耗时 {metrics['elapsed_s']} s")
    print("-" * 72)
    sup = metrics.get("supervised")
    if sup:
        b = sup["binary_review_as_negative"]
        print(f"{'指标':<34}{'值':>14}")
        print(f"{'Accuracy (review→real)':<34}{f(b['accuracy']):>14}")
        print(f"{'Accuracy 95% CI':<34}{str(sup['binary_review_as_negative_ci95']['accuracy_ci95']):>14}")
        print(f"{'FPR (误报率)':<34}{f(b['fpr']):>14}")
        print(f"{'TPR / Recall':<34}{f(b['tpr']):>14}")
        print(f"{'Precision':<34}{f(b['precision']):>14}")
        print(f"{'AUC (连续置信度)':<34}{f(sup['auc_confidence']):>14}")
        print(f"{'覆盖率 (非弃权)':<34}{f(sup['coverage']):>14}")
        print(f"{'保守下界 (弃权=错误)':<34}{f(sup['accuracy_conservative_review_as_error']):>14}")
        print(f"混淆矩阵  TP={b['tp']:<4} FP={b['fp']:<4} FN={b['fn']:<4} TN={b['tn']:<4}"
              f"  (AI={b['n_pos_ai']} / real={b['n_neg_real']})")
        if b.get("note"):
            print(f"[注意] {b['note']}")
        if sup.get("decided_only"):
            d = sup["decided_only"]
            print(f"已决样本  Acc={f(d['accuracy'])} FPR={f(d['fpr'])} TPR={f(d['tpr'])} "
                  f"(n={d['n']})")
    else:
        print("未提供标签：跳过 Accuracy/FPR/TPR（仅结论分布与耗时）")
    print(f"三档分布  suspect={tt['suspect']}  normal={tt['normal']}  review={tt['review']}"
          f"  成因={metrics.get('review_causes') or '—'}")
    lat = metrics["latency_ms"]
    print(f"耗时      p50={lat['p50']} ms  p95={lat['p95']} ms  max={lat['max']} ms  "
          f"mean_wall={lat['mean_wall']} ms")
    by_type = {}
    for f in failures:
        by_type[f["failure_type"]] = by_type.get(f["failure_type"], 0) + 1
    print(f"失败案例  {by_type if by_type else '无'}（共 {len(failures)} 条，详见 failures.csv）")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _synth_png(path: Path, seed: int, size: int = 64):
    import struct
    import zlib
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            raw += bytes(((x * 7 + seed) % 256, (y * 5 + seed * 3) % 256, ((x + y) * 3) % 256))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    data = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    path.write_bytes(data)


def selftest(args) -> int:
    """合成 6 张图 + 标签文件，跑一遍批处理并校验产物齐全。"""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="opensz01_batch_selftest_"))
    imgs = tmp / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    for i in range(1, 7):
        _synth_png(imgs / f"{i:03d}.png", seed=i)
    labels = tmp / "labels.csv"
    labels.write_text("seq,filename,label\n" + "".join(
        f"{i},images/{i:03d}.png,{'ai' if i % 2 else 'real'}\n" for i in range(1, 7)),
        encoding="utf-8")
    ns = argparse.Namespace(**vars(args))
    ns.dir = [str(imgs)]
    ns.file = []
    ns.list = None
    ns.labels = str(labels)
    ns.out = str(tmp / "out")
    ns.limit = None
    ns.recursive = False
    ns.video = False
    ns.ext = "png"
    ns.no_model = True            # 自测不依赖模型（只验证链路与指标计算）
    ns.progress_every = 0
    metrics = run_batch(ns)
    out = Path(ns.out)
    checks = {
        "metrics.json": (out / "metrics.json").is_file(),
        "metrics.md": (out / "metrics.md").is_file(),
        "failures.csv": (out / "failures.csv").is_file(),
        "per_sample.json": (out / "per_sample.json").is_file(),
        "samples>=6": len(list((out / "samples").glob("*.json"))) == 6,
        "n_files==6": metrics["n_files"] == 6,
        "labels==6": metrics["n_labelled"] == 6,
    }
    # 指标计算单测（不依赖模型）
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 0, 1, 1, 0])
    c = confusion(y, p)
    r4 = lambda x: round(x, 4)          # 指标统一四舍五入，比较时同口径
    checks["confusion_math"] = (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2) \
        and abs(c["accuracy"] - r4(4 / 6)) < 1e-9 and abs(c["fpr"] - r4(1 / 3)) < 1e-9 \
        and abs(c["tpr"] - r4(2 / 3)) < 1e-9 and abs(c["precision"] - r4(2 / 3)) < 1e-9
    checks["bootstrap"] = bootstrap_ci(y, p, draws=200)["accuracy_ci95"] is not None
    checks["auc"] = roc_auc(y, np.array([0.9, 0.3, 0.2, 0.6, 0.8, 0.1])) is not None
    ok = all(checks.values())
    print("\n[selftest] " + json.dumps(checks, ensure_ascii=False))
    print(f"[selftest] {'通过' if ok else '失败'}；产物目录 {out}")
    return 0 if ok else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OPEN-SZ-01 批量评测（一键跑 100 套样本）")
    ap.add_argument("--dir", action="append", default=[], help="待评测目录（可重复）")
    ap.add_argument("--file", action="append", default=[], help="单个文件（可重复）")
    ap.add_argument("--list", default=None, help="文件清单（每行一个路径）")
    ap.add_argument("--out", default=None, help="输出目录（默认 reports/batch/<时间戳>）")
    ap.add_argument("--labels", default=None, help="标签文件（seq,filename,label 或 filename,label）")
    ap.add_argument("--label-dirs", default=None,
                    help="按目录名取标签，如 'ai=ai_generated,real=real_photo'")
    ap.add_argument("--infer-labels", action="store_true", help="自动按目录名/文件名推断标签")
    ap.add_argument("--ext", default=None, help="扩展名过滤，如 png,jpg")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--video", action="store_true", help="评测视频（抽帧聚合）")
    ap.add_argument("--html", action="store_true", help="对 FP/FN 额外生成单文件 HTML 报告")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--no-model", action="store_true", help="占位模式（不加载模型，验证链路）")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--time-budget", type=float, default=5.0)
    ap.add_argument("--enable-npr", action="store_true")
    ap.add_argument("--disable-aeroblade", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="合成样本自测后退出")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.selftest:
        return selftest(args)
    if not args.out:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        reports = ROOT / "reports" if (ROOT / "reports").is_dir() else ROOT.parent / "reports"
        args.out = str(reports / "batch" / stamp)
    run_batch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
