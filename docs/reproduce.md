# OPEN-SZ-01 可复现资料包说明

> 目的：支持他人按照本资料重新还原/运行出演示中展示的结果（对应提交要求"可复现演示结果的资料包"）。
> 范围：数据提取与盘查、D1–D8 划分协议、L0/L1 取证管线与先导基线。L2/L3 深度模型与融合决策的复现说明将在 P1–P3 阶段随代码补入（对应章节以 ⏳ 标注）。
> 原则：全部流程**离线可运行**；数据盘查与划分部分仅依赖 Python 标准库；版本、哈希、冻结签名随包提供。

---

## 1. 资料包构成

```
repo/                          # 代码仓库（GitHub 骨架）
├── src/
│   ├── l0_input.py            # L0 输入解析与规范化（PNG 块走查、sha256+aHash、最长边512缩放）
│   ├── l1_forensics.py        # L1 47 维取证特征（色彩/纹理/噪声/自相关/块效应/通道相关/多尺度残差/容器）
│   ├── l2_npr.py              # L2 NPR 接入（⏳ P1）
│   ├── l2_clip.py             # L2 CLIP/AIDE 接入（⏳ P1）
│   └── l2_aeroblade.py        # L2 AEROBLADE 接入（⏳ P3，可选）
├── scripts/
│   ├── extract_features.py    # 批量 L1 特征提取 → features.csv
│   └── baseline_l1.py         # 先导基线：L2 逻辑回归，分层 5 折 × 5 种子
├── assets/report-template.json# 结构化报告模板（手册 9.2）
├── docs/                      # 方案说明 / 演示脚本 / PPT 大纲 / 本说明
├── requirements.txt           # Python 依赖（torch 单独安装）
└── README.md

数据侧（随资料包分发的数据目录，大文件不入 git）：
data/
├── images/                    # 119 张命题样本 PNG（001.png–119.png）
├── labels.csv                 # seq, filename, label（ai/real）
├── manifest.json              # 每图 sha256 / 宽高 / 字节数 / 源 xlsx media 成员 / 标签
├── splits.json                # 示例划分（seed=42, 0.6/0.2/0.2）
├── splits_frozen.json         # 冻结划分签名（--verify-frozen 校验对象）
└── phash_cache.json           # aHash64 感知哈希缓存（sha256 索引）

盘查侧脚本（工作区 scripts/，随包附）：
scripts/
├── extract_xlsx_samples.py    # 从命题 xlsx 流式提取 119 张样本与标签
├── split_dataset.py           # D1–D8 划分协议脚本（纯标准库）
└── build_audit.py             # 数据盘查报告生成 + 独立重推校验（9 项）
```

> 注：命题原始文件 `AI生成材料反欺诈风控检测.xlsx`（约 235 MB）为 ZIP 容器，119 个 PNG 位于 `xl/media/`，D 列给出标签；提取后可凭 `data/manifest.json` 中的 sha256 逐字节校验。

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

### 4.5 L2/L3 与端到端检测（⏳ P1–P3 补入）

⏳ 待补：模型权重获取方式与哈希、NPR/CLIP-AIDE 推理命令、融合与阈值标定命令（`--calibrate` 参数：校准集路径 / 目标 FPR / bootstrap 次数）、TruFor 定位命令、视频通道命令（`--frames 16` 等间隔抽帧）、批量评测命令与报告输出路径。补充时将同步更新 `assets/report-template.json` 的版本字段。

## 5. 预期输出对照表

| 步骤 | 产物 | 关键预期值 |
|---|---|---|
| 4.1 提取 | `data/images/`、`labels.csv`、`manifest.json` | 119 PNG；ai=56 / real=63 |
| 4.2 盘查 | `docs/data-audit.md` | 独立重推校验 9/9 PASS |
| 4.3 划分 | `data/splits.json`、`splits_frozen.json` | seed=42 → 71/24/24；verify-frozen exit=0 |
| 4.4 基线 | `features.csv` + 终端指标 | AUC ≈0.916；最高 acc ≈85.7% |
| 4.5 端到端 ⏳ | `reports/*.json`（按 `assets/report-template.json`） | 冻结评测指标（P4 填入） |

## 6. 复现纪律与常见问题

1. **不重新调参**：复现目标是还原演示结果；阈值、融合权重、划分签名均以随包冻结文件为准（D6–D8）。任何"在测试集上重搜阈值"的做法均属评测泄漏，不在本资料包支持范围。
2. **哈希校验**：复现前可用 `manifest.json` 校验数据完整性；`splits_frozen.json` 校验划分一致性；模型权重哈希 ⏳ 随 P1 补入。
3. **结果波动说明**：基线交叉验证的折级准确率随种子有小幅波动属正常（协议为 5 折 × 5 种子取均值）；划分脚本在相同 seed 下结果确定。
4. **已知边界**：命题样本 119 张不含视频；视频复现需公开视频数据集（⏳ P3 给出清单与来源）。类别比例 47:53 与评测集 7:3 不同，阈值不可按本集比例外推。
5. **常见问题**：
   - `extract_features.py` 报 `[WARN] ... not a PNG`：检查数据目录与 `labels.csv` 的 `filename` 是否一致；
   - `--verify-frozen` 非零退出：划分参数（seed/ratios/meta/phash 开关）与冻结时不一致，按 §4.3 原参数重跑；
   - OpenCV 缺库：确认安装的是 `opencv-python-headless` 且虚拟环境已激活；
   - Windows 路径：命令中路径含中文或空格时加引号。
