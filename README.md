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

在命题给定 119 张样本上。**两套口径必须分清**（详见 [docs/results.md](docs/results.md) §0）：

| 指标 | 折外（OOF，防泄漏） | 冻结阈值实测（现役服务） | 标准 |
| --- | --- | --- | --- |
| 综合准确率 | **93.14%**（6 个 CV 种子，范围 91.6–95.8%） | **100%**（头见过这 119 张，仅用于演示/回归） | ≥95% |
| 误报率 FPR | **3.17%**（每个种子都是 2/63） | 0% | ≤3% |
| 融合 OOF AUC | **0.9750**（`combined` 单路 0.9753） | — | — |
| 2-FP 预算下可达上限 | **98.32%** | — | — |
| 端到端 P50 / P95（全量 119 张，含热力图与注意力解释） | **1.10 s / 1.53 s** | 同左 | ≤5 s ✅ |
| 4928 px 大图端到端 | **1.50 s** | — | ≤5 s ✅ |
| 峰值显存 | **1416 MiB** / 8150 MiB | — | — |
| AIGC 工具覆盖 | 训练覆盖 **18 类**生成器 | — | ≥3 种 ✅ |

> 数字一律取自归档证据文件（`reports/*.json`、`models/deployment_metrics.json`），不手抄单次运行值。
> 本轮迭代（分块规格搜索、patch-token 线索族、阈值单位缺陷修复、注意力解释去重复计算）
> 的过程、被否决方案与负面结果见 **[docs/iteration-2026-10-11.md](docs/iteration-2026-10-11.md)**。
>
> ⚠️ **已知缺陷（未修复）**：上述指标只对**与命题样本同分布的材料**成立。用冻结阈值推理
> 41 张网络采集的真实事故/理赔照片时 **41/41 = 100% 被判为疑似生成**（归档配置同样 100%，
> 非本轮引入的回归）——命题 119 张的"真实"类几乎全是手机拍摄的纸质件，模型学到的是
> "拍摄的纸=真"这一捷径。事实、归因与下一步验证顺序见
> **[docs/known-issue-scene-photo-fpr.md](docs/known-issue-scene-photo-fpr.md)**。

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
# python scripts/run_detect.py --input <image-or-video>
```

### Team

Tairan Chu · Panyuan Wang · Cheng Gao

---

## License

MIT — see [LICENSE](LICENSE).
