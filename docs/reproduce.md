# OPEN-SZ-01 可复现资料包说明

> 目的：支持他人按照本资料重新还原/运行出演示中展示的结果（对应提交要求"可复现演示结果的资料包"）。
> 范围：**全链路已交付**——数据提取与盘查、D1–D8 划分协议、L0/L1/L1b 取证、L2 深度取证（分块 CLIP）、L3 定位、融合校准与三档决策、结构化报告、视频通道、评测与消融实验。
> 原则：训练与评测流程**可离线复现**；预训练权重一次性下载后本地缓存；版本、阈值、评测证据随包提供。
> 一键入口：`python scripts/run_all.py`（分步说明见 §5）。

---

## 1. 资料包构成

```
repo/                          # 代码仓库（https://github.com/Chtrrrrrrr/Zhongan-OPEN-SZ-01）
├── src/
│   ├── l0_input.py            # L0 解析与规范化（PNG 块走查、sha256+aHash、最长边512缩放；支持内存解析）
│   ├── feature_spec.py        # ★ 特征块布局的单一真源（分块尺度/聚合/patch 开关），离线与在线共用
│   ├── l1_forensics.py        # L1 47 维取证特征
│   ├── l1b_domain.py          # L1b 20 维"拍摄态 vs 渲染态"取证（噪声均匀性/光照梯度/锐度一致性/频谱）
│   ├── l2_clip.py             # 冻结 CLIP-L/14 特征提取 + 线性头
│   ├── l2_clip_tiles.py       # 原生分辨率分块 CLIP 特征（关键增益来源）
│   ├── l2_npr.py              # NPR 分支（已实现，经消融剔除）
│   ├── l2_aeroblade.py        # AEROBLADE 接入位（可选，未启用）
│   ├── l3_localization.py     # L3 ELA + 噪声不一致异常图、区域框、块级归因、patch-token 统计
│   ├── fusion.py              # 温度标定 + 加权 logit 融合 + 三档决策（阈值统一为概率空间）
│   ├── detector.py            # 统一图片/视频服务（逐成员超时与降级）
│   ├── video.py               # 视频通道：均匀抽帧 → 逐帧证据 → 聚合 → 可疑片段
│   └── report.py              # 结构化 JSON + 自包含 HTML 报告
├── scripts/
│   ├── extract_features.py    # 批量 L1 特征 → features.csv
│   ├── extract_l1b_features.py# 批量 L1b 特征 → models/l1b_feats.npy
│   ├── extract_clip_features.py # 批量 CLIP 全局 + 分块特征
│   ├── extract_clip_patch_features.py # ★ 冻结 ViT 的 patch-token 统计（训练-free 线索族）
│   ├── tile_clip_sweep.py     # ★ 分块规格搜索：网格/重叠/预缩放 × 聚合 × 多尺度 × 多种子
│   ├── build_feature_spec.py  # ★ 把选中的规格冻结成 models/feature_spec.json
│   ├── calibrate_deployment.py# ★ 多轮嵌套 CV → 冻结头/阈值 + 出部署口径指标
│   ├── verify_matrix.py       # ★ 验证矩阵（单元 × 聚合 × 成员组合 × 多种子）
│   ├── verify_rounds.py       # ★ 逐轮折外证据、成员 LOO、判定规则对照、错误归因
│   ├── external_holdout.py    # ★ 外部素材集的冻结阈值检查（误报频率/泛化排序）
│   ├── calibrate_with_external_reals.py # ★ 扩充纯真实标定池（手册 R2）及可分性检查
│   ├── extract_external_corpus.py       # ★ 外部带标签语料的特征块提取（当前规格）
│   ├── train_with_external_reals.py     # ★ 外部语料只进训练折：缺陷修复前后对照
│   ├── tune_fusion_for_reals.py         # ★ 融合权重 × 命题权重联合选择（场景照安全）
│   ├── freeze_external_fix.py           # ★ 冻结修复后的头与阈值（原 `models/fix_metrics.json` 已缺失，
│   │                                     #   入库替代物见 reports/fix_metrics_restored.json）
│   ├── train_clip_probe.py    # 在 18 生成器基准上训练冻结 CLIP 线性探针
│   ├── train_npr.py           # 训练 NPR 分支（消融用）
│   ├── extract_npr_scores.py  # NPR 打分与消融判定
│   ├── evaluate_system.py     # 嵌套 CV 评测 + 延迟基准 + 报告生成（核心）
│   ├── experiment_features.py # 特征集消融（含剔除容器指纹的对照）
│   ├── experiment_fusion.py   # 融合规则消融（combined/forensic/consensus/avg）
│   ├── diagnose_errors.py     # 错误归因（最困难真实/生成样本画像）
│   ├── robustness_matrix.py   # 鲁棒性矩阵（重压缩/缩放/截图/噪声/模糊）
│   ├── make_figures.py        # 生成架构图、消融图、前沿图、耗时图
│   ├── smoke_test.py          # 冷启动 + 全量热启动 P50/P95 + 视频预算自检
│   ├── test_demo_api.py       # 演示服务 HTTP 接口测试
│   ├── make_test_video.py     # 合成视频烟测片
│   ├── predict.py             # 单图/单视频 CLI
│   ├── baseline_l1.py         # L1 先导基线（分层 5 折 × 5 种子）
│   └── run_all.py             # 一键复现（含 2b 规格搜索与 11 多轮验证）
├── demo/app.py                # 演示服务（零外部 Web 依赖，含启动预热）
├── api/server.py              # REST 服务（含启动预热与报告落盘）
├── tests/test_l1.py           # L0/L1 自测
├── assets/                    # 架构图/结果图/报告模板
├── reports/                   # 评测证据（metrics / 各类消融 / 错误分析 / 冒烟测试）
└── docs/                      # 方案说明、实测结果、演示脚本、PPT 大纲、本说明、迭代记录
```

数据侧（随资料包分发的数据目录，大文件按 `.gitignore` 不入 git）：

```
data/
├── images/                    # 119 张命题样本 PNG（001.png–119.png）
├── labels.csv                 # seq, filename, label（ai/real）
├── manifest.json              # 每图 sha256 / 宽高 / 字节数 / 源 xlsx media 成员 / 标签
├── splits.json                # 示例划分（seed=42, 0.6/0.2/0.2）
├── splits_frozen.json         # 冻结划分签名（--verify-frozen 校验对象）
├── phash_cache.json           # aHash64 感知哈希缓存
└── external/                  # AIGC-Detection-Benchmark 分片（训练用，约 1.4 GB）

models/                        # 训练产物（不入 git，需按 §5 复现生成）
├── hf_cache/                  # CLIP-L/14 本地权重快照（约 1.6 GB）
├── feature_spec.json          # ★ 冻结的特征块布局（分块尺度/聚合/patch 开关），离线与在线共用
├── clip_tiles_spec.npy        # ★ 按 feature_spec 聚合好的分块特征（119 张）
├── clip_patch.npy             # ★ patch-token 统计（119 张）
├── clip_probe_norm.pt         # 统一重编码训练的冻结 CLIP 线性探针
├── forensic_lr.joblib         # [L1 ⊕ L1b ⊕ patch] 头（含外部语料训练，见 reports/fix_metrics_restored.json）
├── combined_lr.joblib         # [forensic ⊕ CLIP全局 ⊕ 分块] 头（同上）
├── clip_domain_lr.joblib      # CLIP 全局头（视频通道用）
├── fusion.json                # 冻结的融合权重 + 温度 + 三档阈值（阈值空间=概率）
├── fix_metrics.json           # ✗ 已缺失（.gitignore 的 `models/*`；生成后未归档）
│                              #   → 替代物：reports/fix_metrics_restored.json（可入库、含来源哈希）
├── deployment_metrics.json    # ★ 多轮嵌套 CV 的部署口径指标与阈值标定过程
└── external_corpus/           # ✗ 已缺失（1561 张外部语料特征块）
                               #   → 来源登记：reports/external_corpus_manifest.json（含本地可核实的 41 张 sha256）

工作区 scripts/（盘查侧，随包附）
├── extract_xlsx_samples.py    # 从命题 xlsx 流式提取 119 张样本与标签
├── split_dataset.py           # D1–D8 划分协议脚本（纯标准库）
└── build_audit.py             # 数据盘查报告生成 + 独立重推校验（9 项）
```

> ⚠️ **2026-10-13 补充（audit-3 M1）：上表这 3 个脚本与 `docs/data-audit.md` 位于
> 「**工作区根**」，不在本 git 仓库内**；仓库里只有 `repo/scripts/` 那一套。
> 同样**不在仓库内**的还有：
> - **命题原始输入**：`AI生成材料反欺诈风控检测.xlsx`（约 235 MB，工作区根）
> - **提取后的 119 张图**：工作区根 `data/images/001.png … 119.png`
> - **模型权重**：`repo/models/` 只有 `.gitkeep` 被跟踪；`fusion.json`、两个 `*_lr.joblib`、
>   `deployment_metrics.json` 等**被 `.gitignore:57 models/*` 排除**
> - **外部语料**：`models/external_corpus/`（1561 张特征块）缺失
>
> **后果**：`git clone` 本仓库后**没有任何可直接运行的输入**，`scripts/run_all.py` 的
> 第 1 步（读 `data/images/NNN.png`）即失败。**"可复现资料包"在第一步就断链。**
> 提交时必须以**附带资料包**（zip/网盘）形式一并提供下列内容，或在仓库内补白名单：
> 119 张命题图 + `data/labels.csv` + `data/manifest.json` + `models/fusion.json` +
> `models/*.joblib` + `models/feature_spec.json` + `models/deployment_metrics.json`。
> 这正是 `models/fix_metrics.json` 当初消失的同一机制。
>
> 另：`scripts/run_detect.py` 在 `README.md` 英文段的 Quick Start 里被引用，但**该文件不存在**，
> 应为 `scripts/predict.py`（已注明）。

> 注：命题原始文件 `AI生成材料反欺诈风控检测.xlsx`（约 235 MB）为 ZIP 容器，119 个 PNG 位于 `xl/media/`，D 列给出标签；提取后可凭 `data/manifest.json` 中的 sha256 逐字节校验。

## 1.1 权重与阈值的加载路径（单一入口，含两份 `fusion.json` 的 SHA-256 对照）

工作区里存在**两份同名但不同内容**的 `fusion.json`，历史上"相对当前工作目录解析 `models/`"的脚本会
静默加载过期那份，从而打印与文档不一致的阈值。现在所有需要部署权重的入口统一走
`scripts/models_path.py`，不再各自拼路径。

**解析顺序（`resolve_models_dir()`，先命中先用）**

1. 显式 `--models-dir` 参数；
2. `MODEL_DIR` 环境变量；
3. `<repo>/models` —— **现役冻结产物，默认加载的就是这一份**；
4. `<workspace>/models` —— 旧工作副本，仅在 3 不存在时兜底，并在日志里标明 `source=workspace-fallback`。

**默认加载哪一份：`repo/models/fusion.json`（下表左列，现役冻结值）。**

**SHA-256 对照表**

| 文件 | SHA-256 | `thr_low` | `thr_high` | 角色 |
|---|---|---|---|---|
| `repo/models/fusion.json` | `ef5b111910da02e32cd1e70a9e9a32c1ce953a70df1ed964ea0666ac9d448328` | **0.0266** | **0.826623** | **现役冻结产物，默认加载** |
| `<工作区根>/models/fusion.json` | `d7062fe5198eb38c58211d76ef178b6a65196c055b7f426277f249586190a011` | 0.036440409414102856 | 0.6742875975794557 | 过期工作副本，仅在显式指定时才用 |

> 两份文件的字节不同（2215 B vs 246 B，且后者缺 `calibration`/`feature_spec` 等字段），
> 因此哈希必然不同——这本身就是"加载了过期副本"的探针：任何脚本打印的
> `thr_low/thr_high` 只要不是 `0.0266 / 0.826623`，就说明它读到的是右列那份。

**按件回退（`resolve_artifact()`）**：冻结的仓库副本是**子集**——它带 `fusion.json`、两个部署头、
`feature_spec.json`、`deployment_metrics.json` 与 119 张特征块，但**不带**大体积/可选工作件
（`clip_probe_norm.pt`、`clip_domain_lr.joblib`、`l1_lr.joblib`、`hf_cache/`、`npr.pt`）。
这些文件按件回退到 `<工作区根>/models/`，并逐条打印 `source=workspace-fallback`，
所以"用了哪一份 fusion / 哪一个探针"在日志里始终显式可见。

**已接入该入口的脚本**：`scripts/smoke_test.py`（默认 `--out repo/reports/smoke_test.json`）、
`scripts/predict.py`、`scripts/external_holdout.py`、`scripts/run_all.py`（把 `--models-dir`
透传给只读校验步骤）。`scripts/run_all.py` 的**写入**步骤仍写工作区 `models/`（它是重建流程，
不覆盖现役冻结产物）。

**核对命令（评审可直接运行，无需 GPU；任一不一致会打印 WARNING）**

```bash
cd repo
python -c "import sys,pathlib as p; sys.path.insert(0,'scripts'); import models_path as m; \
d,s=m.resolve_models_dir(None,workspace=p.Path('..')); print(m.describe(d,s)); print(m.assert_expected_fusion(d))"
# 期望：[models] <repo>/models  (source=repo)  +  fusion thresholds OK (frozen): low=0.0266 high=0.826623
python -c "import hashlib,pathlib as p; print(hashlib.sha256(p.Path('models/fusion.json').read_bytes()).hexdigest()); print(hashlib.sha256(p.Path('../models/fusion.json').read_bytes()).hexdigest())"
# 期望：ef5b1119...0328（冻结，默认加载） / d7062fe5...a011（过期副本）
```

---

## 1.2 外部语料（`models/external_corpus/`）：构成、来源与本地缺失（可追溯性登记）

> ⛔ **2026-10-13 更正（audit-1 F1/D3、audit-2 N3）**：下文所述的"头号指标
> 95.94% / FPR 2.12% / 场景照 0-of-41"**已被独立审计否证**：
> 它**无法由现役产物 `repo/models/` 复现**（依赖本就不存在的外部语料 + best-of-9 权重选择），
> 且"场景照 0-of-41"是训练集内读数——用现役冻结产物重测同一批 41 张为 **92.68%（38/41）**
> （`reports/real41_domain_gap.json`）。
> **可复算的现役口径见 `reports/operating_point_analysis.json`**：
> 现役阈值逐轮 92.44% / FPR 3.17%（2/63）；规则蕴含点 96.08% 逐轮 / 97.48% 池化。
> 下文保留，作为"为什么缺这份语料会导致结论不可复现"的记录。

头号指标 **95.94% / FPR 2.12% / 场景照误报 0-of-41**（⛔ 已作废，见上）的训练侧来自一份 1561 张的外部带标签语料。
**该语料（特征块与源图）在本仓库与当前工作区都不存在**，因此本文如实登记其构成与来源，
而不是提供一个无法核实的数字。

**构成**（`reports/external_reals_fix.json → corpus`：`real=1040`、`fake=521`；目录 5 个）

| 目录 | 标签 | 张数/角色 | 来源 |
|---|---|---|---|
| `real` | 真实 | 41 张网络采集的真实事故/理赔照（车祸/水浸/火灾/财产损坏/食品变质/空白单据） | `docs/known-issue-scene-photo-fpr.md` §1；逐张 `image_url`/`source_page` 见 `repo/testset/real/_real_claim_index.json` |
| `real_synthbuster` | 真实 | Synthbuster 论文级真假配对基准的真实侧（与上面 41 张合计 1040） | Synthbuster（Bstyvul）；`docs/known-issue-scene-photo-fpr.md` §1 |
| `fake_ai` | 生成 | 公开 AI 生成电商损坏商品图（electronics / product / screen 各 55 张） | `docs/results.md` §6 |
| `fake_dataset` | 生成 | 文生图票据（生成单据）50 张 | `docs/results.md` §6 |
| `fake_tamper` | 生成 | 传统篡改（拼接/复制粘贴）12 张 | `docs/results.md` §6 |

覆盖生成器与提取命令见 `docs/known-issue-scene-photo-fpr.md` §4 与 §7；
语料**不等价于真实理赔分布**（它是基准配对 + 公开生成数据集，见该文 §6.2）。

**本地缺失与哈希可计算性（如实声明）**

- `repo/models/external_corpus/`、`<工作区根>/models/external_corpus/`、`work/**/external_corpus/`
  均已检索，**目录不存在**（预期产物 `l1.npy/l1b.npy/clip.npy/patch.npy/tiles.npy/meta.json` 均缺失）；
  源树 `testset/dsh_testset/{real,real_synthbuster,fake_ai,fake_dataset,fake_tamper}` 同样不存在。
- 因此 **1561 张里只有 41 张（= 2.63%）能给出逐文件 sha256**——即本地留存的
  `repo/testset/real/*.jpg`（真实侧的一部分，逐张哈希已登记在 `reports/external_corpus_manifest.json → sha256.hashed_locally`）。
- 其余 **1520 张**（999 张 Synthbuster 真实侧 + 521 张生成/篡改侧）的源图与特征块都不在本地，
  **逐文件 sha256 无法计算**，本仓库**不提供估计值或占位哈希**；相应地，
  **95.94% 无法从零重建**——它只能在语料重新落地后按 §4.5/§6 的流程重跑复现。
- 完整性索引交叉核对：`sha256.index_cross_check` 记录本地 41 张与源索引的一致性。
  注意 `repo/reports/testset_real_41_integrity.json` 同时记录了 **41/41** 个文件的
  `pixels_sha256_mismatch`（源索引按**解码像素**记录的 sha256 与本地的对应值不一致）；
  该差异**原样留存、不作抹平**，本轮也未进一步归因（41 张本地文件字节的 sha256 已另行登记）。

**补救步骤**（与 `reports/external_corpus_manifest.json → remediation` 一致）

1. 重新取得源语料（Synthbuster 配对 + 公开 AI 生成电商/票据 + 传统篡改 + 41 张理赔照），
   按 5 个目录放回 `testset/dsh_testset/`；
2. 用 `declared.extraction_command` 重跑 `scripts/extract_external_corpus.py`，生成 `models/external_corpus/*.npy + meta.json`；
3. 语料落地后，把本登记从"声明 + 41 张哈希"升级为"**1561 张逐文件哈希**"（对源树逐文件 sha256，
   与 `meta.json` 的路径对齐即可——提取脚本解码时已计算每张图哈希，只需写入产物）；
4. 之后再按 §4.5/§6 的命令重跑训练与冻结，使 95.94% 可由仓库内文件独立重建。

---

## 2. 环境依赖

| 项目 | 要求 | 说明 |
|---|---|---|
| Python | ≥ 3.10 | 盘查与划分脚本仅用到标准库 |
| 基础依赖 | `pip install -r requirements.txt` | numpy / scipy / scikit-learn / pillow / opencv-python-headless / imageio / imagehash / openpyxl / matplotlib / requests / tqdm |
| 深度学习 | torch 按本机 CUDA 版本单独安装（⏳ P1 起需要） | 例：CUDA 12.1 `pip install torch --index-url https://download.pytorch.org/whl/cu121`；CPU `.../whl/cpu` |
| 硬件 | 消费级独显笔记本即可；盘查/划分/基线阶段纯 CPU 可跑 | GPU 仅 L2/L3 阶段需要 |
| 网络 | 全流程离线（模型权重需一次性预先下载 ⏳） | 演示与复现不依赖外网 |

## 3. 安装步骤

```bash
# 1. 获取代码
git clone <仓库地址> && cd repo

# 2. 创建虚拟环境（建议 Python 3.10+）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. 安装依赖（torch 如需请先按上表单独安装）
pip install -r requirements.txt

# 4. 放置数据：将数据目录按 §1 结构放到约定位置
#    - 若从命题 xlsx 自行提取，见 §4.1；直接使用随包 data/ 则跳过
```

## 4. 运行命令与参数配置

### 4.1 命题样本提取（从 xlsx 还原 data/）

```bash
python scripts/extract_xlsx_samples.py --xlsx "AI生成材料反欺诈风控检测.xlsx" --out data
```

- 输出：`data/images/001.png–119.png`、`data/labels.csv`、`data/manifest.json`
- 预期：119 个 PNG、标签 56 ai / 63 real；`manifest.json` 含逐文件 sha256。

### 4.2 数据盘查与独立重推校验（【已验证】结果复现）

```bash
python scripts/build_audit.py
```

- 预期输出：`docs/data-audit.md` 重新生成；校验 **9/9 PASS**：
  样本总数=119、AI=56、真实=63、seq 1–56 全 AI、57–119 全真实、跳变=1、全部有效 PNG 头、无完全重复（sha256 0 对）、labels/manifest 与源 xlsx 一致（zip member 与磁盘文件逐字节 sha256 比对）。

### 4.3 数据划分与泄漏防护（D1–D8）

```bash
# 默认划分：0.6/0.2/0.2，seed=42 → train 71 / val 24 / test 24（分层比例吻合）
python scripts/split_dataset.py

# 自定义比例与种子
python scripts/split_dataset.py --ratios 0.7,0.15,0.15 --seed 7

# 生成器/采集方式联合分层（D2/D3，需提供 meta CSV：seq,generator,acquisition）
python scripts/split_dataset.py --meta data/meta.csv

# 感知哈希近重复检测（D4；纯 stdlib PNG 解码 + 8×8 aHash，实测 0.23–0.55s/张，有缓存）
python scripts/split_dataset.py --phash

# 冻结与校验闭环（D8；exit=0 表示当前划分与冻结签名一致）
python scripts/split_dataset.py --freeze data/splits_frozen.json
python scripts/split_dataset.py --verify-frozen data/splits_frozen.json
```

- 预期：`data/splits.json` 含 protocol 声明块（阈值标定仅允许 val、test frozen=true）；`--phash` 模式下 aHash≤6 的 77 对近重复、4 个混标签组被原子分配到同一划分，断言无 group 跨界。

### 4.4 L0/L1 特征提取与先导基线（【已验证】结果复现）

```bash
cd repo

# 1) 批量提取 47 维 L1 特征（读取 data/labels.csv 指向的样本）
python scripts/extract_features.py --data-dir <数据目录> --out features.csv

# 2) 复现先导基线：L2 正则逻辑回归，分层 5 折 × 5 随机种子
python scripts/baseline_l1.py --features features.csv
```

- 预期输出（对照手册 11.1）：
  - `samples=119  ai=56  real=63  dims=47`
  - 平均 AUC ≈ **0.916**，最高交叉验证准确率 ≈ **85.7%**（各折 80.0/87.5/87.5/91.3/91.3% 量级）
  - 单特征 AUC 排序前列：nidat ≈0.719、hasSrgb ≈0.685、blockiness ≈0.678
- 说明：该基线不使用任何预训练权重，CPU 离线可复现；其意义是验证链路与量化差距，不代表最终系统性能。

### 4.5 L2/L3 与端到端检测（2026-10-13 补齐）

> 原为 `⏳ 待补` 占位符，且恰好位于"如何复现头条指标"的位置（audit-1、audit-3 均点名）。
> 下面给出**当前实际可跑**的入口与**已知不可复现项**。

#### 4.5.1 权重来源与哈希

| 产物 | 位置 | 是否入库 | 说明 |
|---|---|---|---|
| `fusion.json` | `repo/models/` | ❌ `.gitignore:57` | 现役阈值 `thr_low=0.0266 / thr_high=0.826623`；两份副本的 SHA-256 对照见 §1.1 |
| `combined_lr.joblib` / `forensic_lr.joblib` | `repo/models/` | ❌ | 与工作区根同名文件的**系数完全相同**（仅内嵌 `features` 标签串不同），见 `reports/operating_point_analysis.json` 旁注 |
| `feature_spec.json` / `clip_tiles_spec.npy` / `clip_patch.npy` | `repo/models/` | ❌ | 冻结特征块布局 |
| `deployment_metrics.json` | `repo/models/` | ❌ | **本文件是大部分指标的复算来源**，必须随包分发 |
| `clip_probe_norm.pt` / `clip_domain_lr.joblib` | 工作区根 `models/` | ❌ | `scripts/models_path.py` 会按件回退到工作区根 |

**哈希核对命令**（无需 GPU）：

```bash
python -c "import hashlib,pathlib; \
  [print(p, hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]) \
   for p in ['repo/models/fusion.json','models/fusion.json','repo/models/combined_lr.joblib']]"
```

#### 4.5.2 端到端评测（需要 GPU + 119 张图）

```bash
# 全量 119 张：冷启动 + 热启动 P50/P95 + 冻结阈值判定 + 视频自检
python repo/scripts/smoke_test.py                 # → repo/reports/smoke_test.json（本次运行覆盖）
# 默认从 <repo>/models 加载冻结产物；会打印 source=repo 与阈值自检行

# 单图/单视频
python repo/scripts/predict.py --image <path> --out repo/reports

# 复算"现役阈值 vs 规则蕴含点"的对照（纯 CPU，无 GPU）
python repo/scripts/make_operating_point.py       # → repo/reports/operating_point_analysis.json
```

#### 4.5.3 已知不可复现项（必须如实告知）

1. **头条 95.94% / FPR 2.12% / 场景照 0-of-41 无法复现**：依赖缺失的 `models/external_corpus/`
   （1561 张）与 best-of-9 权重选择。可复算的现役数字见 `reports/operating_point_analysis.json`。
2. **`models/external_corpus/` 缺失**：见 §1.2 的来源登记与补哈希步骤。
3. **119 张命题图与原始 xlsx 不在仓库内**：见 §1 的 ⚠️ 说明；`run_all.py` 第 1 步即失败。
4. **TruFor 定位未接入**：`src/l3_localization.py` 现为自适应 ELA+噪声启发式（零权重，不参与判定），
   接入训练型篡改定位仍是下一步（`reports/localization_regions.json`）。
5. **视频**：命题不含视频，`repo/reports/video_temporal_validation.json` 的两类均由静帧编码，
   属**通道级**验证；臂 A（网络真实照）AUC 0.3225，**不构成业务能力**。

## 5. 预期输出对照表

| 步骤 | 产物 | 关键预期值 |
|---|---|---|
| 4.1 提取 | `data/images/`、`labels.csv`、`manifest.json` | 119 PNG；ai=56 / real=63 |
| 4.2 盘查 | 工作区根 `docs/data-audit.md`（**不在本仓库**） | 独立重推校验 9/9 PASS |
| 4.3 划分 | `data/splits.json`、`splits_frozen.json` | seed=42 → 71/24/24；verify-frozen exit=0 |
| 4.4 基线 | `features.csv` + 终端指标 | AUC ≈0.916；最高 acc ≈85.7% |
| 4.5 端到端 | `reports/smoke_test.json`、`reports/operating_point_analysis.json` | 现役阈值逐轮 **92.44%** / FPR **3.17%（2/63）**；规则蕴含点 **96.08% 逐轮 / 97.48% 池化**（同一误报数） |
| 4.5 视频 | `reports/video_temporal_validation.json` | 臂 B AUC 0.99（通道级）；**臂 A 0.3225** |

## 6. 复现纪律与常见问题

1. **不重新调参**：复现目标是还原演示结果；阈值、融合权重、划分签名均以随包冻结文件为准（D6–D8）。任何"在测试集上重搜阈值"的做法均属评测泄漏，不在本资料包支持范围。
   ⚠️ **但须注意**：特征规格（C8+meanmax）与融合权重**本身是在 119 张上做的设计选择**
   （`reports/tile_clip_sweep.json` 的 24 组搜索、`reports/fusion_tuning_reals.json` 的 best-of-9），
   因此严格说不是"零选择"；该披露见 `docs/results.md` §3 第 5 条。
2. **哈希校验**：复现前可用 `manifest.json` 校验数据完整性；`splits_frozen.json` 校验划分一致性；模型权重哈希见 §4.5.1。
3. **结果波动说明**：基线交叉验证的折级准确率随种子有小幅波动属正常（协议为 5 折 × 5 种子取均值）；划分脚本在相同 seed 下结果确定。
4. **已知边界**：命题样本 119 张不含视频；视频复现需公开视频数据集（⏳ P3 给出清单与来源）。类别比例 47:53 与评测集 7:3 不同，阈值不可按本集比例外推。
5. **常见问题**：
   - `extract_features.py` 报 `[WARN] ... not a PNG`：检查数据目录与 `labels.csv` 的 `filename` 是否一致；
   - `--verify-frozen` 非零退出：划分参数（seed/ratios/meta/phash 开关）与冻结时不一致，按 §4.3 原参数重跑；
   - OpenCV 缺库：确认安装的是 `opencv-python-headless` 且虚拟环境已激活；
   - Windows 路径：命令中路径含中文或空格时加引号。
