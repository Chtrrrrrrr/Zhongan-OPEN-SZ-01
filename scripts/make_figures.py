#!/usr/bin/env python3
"""Generate the figures used by the PPT / report / demo.

Writes into assets/:
  architecture.png     layered pipeline diagram
  ablation_auc.png     feature-set ablation (OOF AUC)
  results_frontier.png accuracy vs measured FPR operating frontier
  latency.png          cold / warm / video latency against the 5s budget

Pure matplotlib (no graphviz needed).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# CJK-capable font (Windows ships Microsoft YaHei / SimHei); fall back gracefully
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent


def fig_architecture(out: Path):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    layers = [
        ("L0 输入解析\n格式/解码/规范化\nsha256+感知哈希", "#E3F2FD", 0.6),
        ("L1 取证线索\n容器/元数据/压缩/频域\n+ L1b 拍摄态vs渲染态", "#E8F5E9", 3.1),
        ("L2 深度取证\n冻结 CLIP-L/14\n全局 + 3×3 原生分辨率分块", "#FFF3E0", 5.6),
        ("L3 异常定位\nELA + 噪声不一致图\n（TruFor 为可选升级位）", "#F3E5F5", 8.1),
        ("决策层\n校准 → 融合 → 三档决策\n结构化报告", "#FFEBEE", 10.6),
    ]
    for title, color, x in layers:
        ax.add_patch(FancyBboxPatch((x, 4.4), 2.3, 2.0, boxstyle="round,pad=0.12",
                                    linewidth=1.4, edgecolor="#37474F", facecolor=color))
        ax.text(x + 1.15, 5.4, title, ha="center", va="center", fontsize=9.5)
    for i in range(len(layers) - 1):
        x0 = layers[i][2] + 2.3
        ax.add_patch(FancyArrowPatch((x0, 5.4), (layers[i + 1][2], 5.4),
                                     arrowstyle="-|>", mutation_scale=16, color="#37474F"))

    ax.text(0.6, 6.9, "OPEN-SZ-01 系统架构（图片主链路）", fontsize=13, weight="bold")

    # evidence producers + video channel
    ax.add_patch(FancyBboxPatch((0.6, 1.4), 8.0, 2.3, boxstyle="round,pad=0.15",
                                linewidth=1.2, edgecolor="#90A4AE", facecolor="#FAFAFA"))
    ax.text(0.85, 3.35, "证据生产者（任一路超时/异常可跳过，其余证据继续形成结论）",
            fontsize=9.5, style="italic", color="#455A64")
    ax.text(1.0, 2.8, "forensic  =  L1(47) ⊕ L1b(20)        OOF AUC 0.908", fontsize=9.5)
    ax.text(1.0, 2.3, "combined =  forensic ⊕ CLIP 全局(1024) ⊕ 分块(3×1024)   OOF AUC 0.966",
            fontsize=9.5, weight="bold")
    ax.text(1.0, 1.8, "NPR：已实现但被消融剔除（本任务 AUC 0.569）｜外部冻结探针：AUC 0.648，未过准入线",
            fontsize=8.5, color="#B71C1C")

    ax.add_patch(FancyBboxPatch((9.0, 1.4), 4.4, 2.3, boxstyle="round,pad=0.15",
                                linewidth=1.2, edgecolor="#90A4AE", facecolor="#FAFAFA"))
    ax.text(9.25, 3.35, "视频通道（Stage-1）", fontsize=10, weight="bold")
    ax.text(9.25, 2.85, "10 帧均匀抽样 → 批量全局 CLIP", fontsize=9)
    ax.text(9.25, 2.45, "+ L1/L1b 取证 → mean/P90 聚合", fontsize=9)
    ax.text(9.25, 2.05, "实测 3.5 s（预算 5 s）", fontsize=9)
    ax.text(9.25, 1.65, "命题样本无视频 → 通道验证，非基准", fontsize=8.5, color="#B71C1C")

    ax.text(0.6, 0.6, "三档输出：正常 / 待复核 / 疑似生成或篡改   ｜   对外表述："
                      "「疑似生成合成内容」（与《人工智能生成合成内容标识办法》一致）",
            fontsize=9.5, color="#37474F")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_ablation(out: Path, reports: Path):
    fp = reports / "feature_ablation.json"
    if not fp.exists():
        print(f"[skip] {fp} not found")
        return
    data = json.loads(fp.read_text(encoding="utf-8"))["results"]
    names = {"full": "全部特征", "no_container": "去容器指纹", "pixel_only": "仅像素取证",
             "container_plus_l1b": "仅容器+L1b"}
    labels = [names.get(k, k) for k in data]
    aucs = [data[k]["oof_auc"] for k in data]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, aucs, color=["#1E88E5", "#43A047", "#FB8C00", "#8E24AA"])
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("折外 AUC")
    ax.set_title("特征集合消融（119 张命题样本，分层 5 折折外）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


def fig_frontier(out: Path, reports: Path):
    fp = reports / "error_analysis.csv"
    metrics = reports / "metrics.json"
    if not (fp.exists() and metrics.exists()):
        print("[skip] error_analysis.csv / metrics.json not found")
        return
    m = json.loads(metrics.read_text(encoding="utf-8"))
    op = m.get("operating_points_best_accuracy", {})
    if not op:
        return
    labels, accs, fprs = [], [], []
    for k, v in op.items():
        if "official_ratio" in k:
            lab = "2/70=2.86%\n(命题7:3口径)"
            fpr = 2 / 70
        else:
            lab = k.replace("fpr_le_", "").replace("_", " ")
            try:
                fpr = float(k.split("fpr_le_")[1].split("pct")[0].split("_")[0]) / 100
            except Exception:
                fpr = 0.0
        labels.append(lab)
        accs.append(v)
        fprs.append(fpr)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, accs, width=0.4, label="可达最高准确率", color="#1E88E5")
    ax.bar(xs + 0.2, fprs, width=0.4, label="对应实测 FPR", color="#E53935")
    ax.axhline(0.95, ls="--", color="#2E7D32", lw=1.2, label="准确率目标 95%")
    ax.axhline(0.03, ls=":", color="#6A1B9A", lw=1.2, label="FPR 目标 3%")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title("准确率-误报前沿：为什么 63 个真实样本无法表达 3%")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


def fig_latency(out: Path, reports: Path):
    fp = reports / "smoke_test.json"
    if not fp.exists():
        print(f"[skip] {fp} not found")
        return
    d = json.loads(fp.read_text(encoding="utf-8"))
    warm = d.get("warm_image_ms", [])
    items = [("冷启动", d.get("cold_start_ms", 0), "#8E24AA"),
             ("图片热启动(均值)", float(np.mean(warm)) if warm else 0, "#1E88E5"),
             ("视频", d.get("video", {}).get("ms", 0), "#43A047")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(items))
    ax.bar(xs, [i[1] for i in items], color=[i[2] for i in items], width=0.5)
    for x, (lab, v, _) in zip(xs, items):
        ax.text(x, v + 200, f"{v:.0f} ms", ha="center", fontsize=10)
    ax.axhline(5000, ls="--", color="#E53935", lw=1.4, label="命题预算 5000 ms")
    ax.set_xticks(xs)
    ax.set_xticklabels([i[0] for i in items])
    ax.set_ylabel("端到端耗时 (ms)")
    ax.set_title("端到端耗时实测（冷启动为一次性模型加载，服务启动时预热）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    assets = ROOT / args.assets
    assets.mkdir(parents=True, exist_ok=True)
    reports = WS / args.reports
    fig_architecture(assets / "architecture.png")
    fig_ablation(assets / "ablation_auc.png", reports)
    fig_frontier(assets / "results_frontier.png", reports)
    fig_latency(assets / "latency.png", reports)


if __name__ == "__main__":
    main()
