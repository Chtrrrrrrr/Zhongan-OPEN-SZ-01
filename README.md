# Zhongan-OPEN-SZ-01 · AI 生成材料反欺诈风控检测

> 众安 1024 · AI Coding 挑战赛　命题编号：**OPEN-SZ-01**
> ZongAn 1024 · AI Coding Challenge — Problem ID: **OPEN-SZ-01**

---

## 中文

### 项目简介

针对理赔等场景中 AI 生成的虚假图片/视频材料，构建一套可降级、可解释的反欺诈检测系统。系统采用 **L0 输入解析 → L1 低成本取证线索 → L2 AIGC 主检测 → L3 篡改检测与定位 → 决策层（多证据融合 → 置信度校准 → 三档决策 → 结构化报告）** 的分层流水线，目标是在命题混合样本上达到识别准确率 ≥ 95%、真实材料误报率 ≤ 3%、单份材料端到端 ≤ 5 秒（以上为【命题约束/验收标准】，最终以实测为准）。

- 分层架构详情见 [docs/architecture.md](docs/architecture.md)
- 项目规划与技术验证依据：《附1 OPEN-SZ-01 项目规划与技术验证手册》

### 架构图（占位）

```
输入(图片/视频)
   │
   ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ L0 输入解析  │ → │ L1 取证线索  │ → │ L2 AIGC主检测│ → │ L3 篡改定位  │
│  <0.1s      │   │  <0.3s      │   │  <1.5s      │   │  <1.0s      │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
              │             │             │                 │
              ▼             ▼             ▼                 ▼
       ┌──────────────────────────────────────────────────────┐
       │ 决策层：融合 → 校准 → 三档决策 → 结构化报告            │
       │ （低风险 / 待复核 / 高风险）                           │
       └──────────────────────────────────────────────────────┘
```

### 目录说明

| 目录 | 用途 |
| --- | --- |
| `src/` | 核心源代码（L0–L3 各层与决策层实现） |
| `scripts/` | 数据提取、评测、批处理等脚本 |
| `data/` | 数据目录（**大文件不入库**，见 .gitignore） |
| `docs/` | 架构、运维与协作文档 |
| `demo/` | 演示材料与 Demo 脚本 |
| `models/` | 模型权重（**不入库**，经 Git LFS 或外部分发） |
| `reports/` | 结构化报告与评测结果输出 |
| `tests/` | 单元测试与验证用例 |
| `assets/` | README/文档引用的图片等静态资源 |

### 实测结果（摘要）

在命题给定 119 张样本上。**口径必须分清**（详见 [docs/results.md](docs/results.md) §0）。

> ⚠️ **本节已于 2026-10-13 按独立审计（`work/audit/audit-1.md`）重写。**
> 旧版把「95.94% / 2.12% / 场景照 0%(0/41)」列为达标，但这三个数字
> **（a）无法由仓库内现役产物复现**，**（b）与现役冻结阈值的实测不符**。
> 下表改为只列可复算的数字，并明确区分「现役冻结阈值口径」与「折外口径」。

**现役冻结阈值口径**（`repo/models/fusion.json` 的 `thr_high=0.826623`，
即服务实际会输出的判定；逐轮折外分数上作用该阈值）：

| 指标 | 实测（per-round，6 轮均值） | 标准 | 判定 |
| --- | --- | --- | --- |
| 综合准确率 | **92.44%**（逐轮 91.60 / 94.12 / 91.60 / 90.76 / 92.44 / 94.12） | ≥95% | ❌ 未达标 |
| 误报率 FPR | **3.17%（2/63）** | ≤3% | ❌ 未达标（2/63=3.17%） |
| 召回 TPR | 87.50% | — | — |

**同一个 `fp_budget=2` 规则真正蕴含的工作点**（阈值需**严格小于**池化纯真实分数第 3 大值，
故取 `-0.4243068333323333` = 第 3 大 + 1e-12；**不要四舍五入成 -0.424307**，
那会比第 3 大值大而放进 3 个误报）：

| 指标 | 汇总池化 | per-round（6 轮均值） | 标准 |
| --- | --- | --- | --- |
| 综合准确率 | **97.48%（TP55/TN61/FP2/FN1）** | **96.08%** | ≥95% ✅ |
| 误报率 FPR | **3.17%（2/63）** | 5.03%（逐轮 FP 2–5） | ≤3% ❌ |
| 召回 TPR | 98.21% | 97.32% | — |

**关键结论（必须与结论一起说）**：现役冻结阈值 **不是**它自己声明的规则所蕴含的工作点，
两者都可在同一份归档折外分数上复算（`reports/operating_point_analysis.json`）。
按命题要求的 7:3 比例重采样（70 真 / 30 假，2000 次）：现役阈值 P(准确率≥95%)=**0.63**，
规则蕴含点=**0.94**。
但**任何可复算的工作点都无法同时满足 ≥95% 与 ≤3%**：在本集合上 ≤3% 只允许 ≤1 个误报
（1/63=1.59%），而把阈值提到 0 误报会摧毁全部召回（准确率跌到约 53%）。

| 指标 | 实测值 | 证据 |
| --- | --- | --- |
| 真实理赔场景照误报（41 张网络真实照） | **92.68%（38/41）** ❌ | `reports/real41_domain_gap.json`（实测，`source=repo`、thr_high=0.826623） |
| 折外融合 AUC | **0.9750**（`combined` 单路 0.9753） | `reports/metrics.json` |
| 端到端 P50 / P95（全量 119 张） | **1.17 s / 1.59 s**（归档快照） | `reports/smoke_test.json` |
| 峰值显存（分配） | **1765.3 MiB** / 设备 **8150.6 MiB** | `reports/resource_profile.json` → `peak_allocated_mib_end_to_end`（⚠️ 该文件的 `total_memory_mib=12199.4` 与本机实际 8150.6 MiB 不符，疑来自另一台机器，已于 2026-10-13 标注） |
| AIGC 工具覆盖 | 训练 **18 类**生成器 + 外部 13 种 | `reports/cross_generator.json` |
| 视频通道（受控编码短片） | 通道级 arm B AUC 0.99；**臂 A（网络真实照）AUC 0.3225** | `reports/video_temporal_validation.json` |

> 数字一律取自归档证据文件，不手抄单次运行值。
> 视频结论务必读 `reports/video_temporal_validation.json` 的 `declarations`：
> 命题样本 **0 条视频**，真假两类均由静帧编码而成，因此**只是通道级验证，不是业务能力**。

> 数字一律取自归档证据文件（`reports/*.json`、`models/deployment_metrics.json`、
> `reports/fix_metrics_restored.json` —— 原 `models/fix_metrics.json` 缺失，该文件是它的可入库替代物，
> 见其 `restoration_note`），不手抄单次运行值。
>
> **两轮迭代**（过程、被否决方案与负面结果都在文档里，不在代码注释里）：
> 1. [docs/iteration-2026-10-11.md](docs/iteration-2026-10-11.md) —— 分块规格搜索、patch-token 线索族、
>    阈值单位缺陷修复、注意力解释去重复计算；
> 2. [docs/known-issue-scene-photo-fpr.md](docs/known-issue-scene-photo-fpr.md) —— 真实场景照被系统性
>    误判的发现、归因与可分性分析。**注意**：该文档所述"场景照误报 100% → 0%"是在
>    **把外部语料（含这 41 张）并入训练负类**的条件下测得的；用**现役冻结产物**重测同一批
>    41 张网络真实照，误报仍为 **92.68%（38/41）**（`reports/real41_domain_gap.json`）。
>    即该修复**未固化进任何已冻结、可复现的产物**。
>
> **第三轮（2026-10-13，独立审计后）**：`work/audit/audit-1.md` 对本案做了对抗式审计，
> 上表即其结论的落地。相关证据：`reports/operating_point_analysis.json`、
> `reports/baseline_improvement.json`、`reports/decision_gate_negative_result.json`。

### 快速开始

```bash
# 1. 依赖（torch 需按 CUDA 版本单独安装，见 requirements.txt）
pip install -r requirements.txt

# 2. 一键复现全部结果（特征 → 分块规格搜索 → CLIP 探针训练 → 评测 → 报告 → 多轮验证）
python scripts/run_all.py --steps 1,2,2b,4,11

# 3. 单张图片 / 单条视频检测
python scripts/predict.py --image ../data/images/001.png --out ../reports

# 4. 启动演示服务（含启动预热）
python demo/app.py --port 7860     # 打开 http://127.0.0.1:7860

# 5. 演示前自检（冷/热启动、全量热启动 P50/P95、图片与视频预算）
python scripts/smoke_test.py
```

> **工作区布局说明**：本仓库位于工作区的 `repo/` 子目录；命题原始数据、提取出的图片、
> 模型权重与运行产物位于工作区根（`../data/`、`../models/`、`../reports/`），均按
> `.gitignore` 排除在版本控制之外（权重与数据体积过大）。评测证据（`reports/*.json`）
> 已同步归档到本仓库 `reports/` 以便复核。
>
> **权重加载路径**：工作区里有两份内容不同的 `fusion.json`（现役冻结 vs 过期工作副本）。
> 所有部署权重的加载统一走 `scripts/models_path.py`：**默认加载仓库内冻结的 `models/fusion.json`
> （`thr_low/thr_high = 0.0266 / 0.826623`）**，逐件回退与两份文件的 SHA-256 对照表见
> [docs/reproduce.md §1.1](docs/reproduce.md)。

### 团队成员

褚泰然 · 王潘榞 · 高诚

---

## English

### Overview

A fraud-detection system for AI-generated claim materials (images/videos). The pipeline is a degradable, explainable stack: **L0 input parsing → L1 low-cost forensic cues → L2 AIGC main detection → L3 tamper detection & localization → Decision layer (multi-evidence fusion → confidence calibration → three-tier decision → structured report)**. Targets set by the problem statement: ≥95% accuracy, ≤3% false-positive rate on genuine materials, ≤5s end-to-end per item (subject to final measurement).

See [docs/architecture.md](docs/architecture.md) for the layered architecture.

### Quick Start (placeholder)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # install torch separately per your CUDA build
python scripts/predict.py --image <path-to-image>    # single image/video CLI
# NOTE: `scripts/run_detect.py` does not exist; the CLI entry point is scripts/predict.py.
# Also see docs/reproduce.md §1.1: the competition images and the model weights are NOT
# tracked in this repository, so a fresh clone has no runnable input (audit-3 M1).
```

### Team

Tairan Chu · Panyuan Wang · Cheng Gao

---

## License

MIT — see [LICENSE](LICENSE).
