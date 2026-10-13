# testset/real（41 张真实图片）评测报告

- 评测对象：`repo/testset/real/`（41 张真实理赔场景照 + `_real_claim_index.json`，索引声明 8 类、全部 real）
- 被测系统：当前冻结交付版（`models/fusion.json`，`thr_low=0.0364` / `thr_high=0.6743`，成员 `combined` 0.6 + `forensic` 0.25）
- 环境：python 3.14.7 / torch 2.11.0+cu128 / CUDA（RTX 5070 Laptop）/ CLIP ViT-L/14 本地快照
- 运行时间：2026-10-11，单次运行

---

## 0. 一句话结论

**41 张真实照片里 38 张被判"疑似生成/篡改"、3 张转人工、0 张判为"正常"，误报率 FPR = 92.68%（要求 ≤3%）。**
根因不是格式/压缩，而是 **CLIP 证据通道（`combined` 头里的 CLIP global + CLIP tiles）对"本域之外的真实照片"整体判错**：测试集的连续分值整体落在域内 **AI 簇**的分值带里。时延、报告 schema、稳定性等非精度指标不受影响。

---

## 1. 复现命令

```powershell
cd repo
$env:PYTHONIOENCODING="utf-8"
# 标签文件由索引生成：work/testset_real_labels.csv（filename,label，41 行全 real）
python scripts/batch_evaluate.py `
  --dir "C:\Files\Deepseek Harness\Zhongan-OPEN-SZ-01\repo\testset\real" `
  --labels "C:\Files\Deepseek Harness\Zhongan-OPEN-SZ-01\work\testset_real_labels.csv" `
  --html --out "C:\Files\Deepseek Harness\Zhongan-OPEN-SZ-01\repo\reports\batch\real41"
```

> 注：`--infer-labels` 只按**目录名**取标签，不按文件名推断，因此本目录（根目录即 `real/`）会得到 0 个标签、进而跳过 FPR；本次用显式 `--labels` 补齐。这是脚本的口径限制，不是数据问题。

---

## 2. 结果

| 指标 | 值 |
| --- | --- |
| 样本数 | 41（全部 real；AI=0） |
| 三档分布 | **suspect 38 / review 3 / normal 0** |
| FPR（误报率，suspect 记误报） | **0.9268**（95% CI [0.8293, 1.0]） |
| Accuracy（review→real，与 `reports/metrics.json` 同口径） | 0.0732 |
| 保守下界（弃权也记错） | 0.0000 |
| Precision | 0.0000（TP=0, FP=38） |
| 覆盖率（非弃权） | 0.9268 |
| fused score 分布 | min 0.444 / p25 0.879 / **median 0.967** / p75 0.985 / max 0.995（阈值 0.674） |
| review 成因 | 模型分歧 2、不确定带 1 |
| 耗时 | p50 **794 ms** / p95 853 ms / max 878 ms（预热 12.3 s，总 47.7 s）→ 单材料 ≤5 s 达标 |
| 归档质检 | schema 1.1.0，4 类标记 41/41 齐全，`all_pass=true`；无解码失败、无降级运行 |

### 逐类 FPR

| 类别 | 误报/总数 | FPR |
| --- | --- | --- |
| fire_damage | 2/2 | 1.00 |
| flood_damage | 5/5 | 1.00 |
| food_issue | 8/8 | 1.00 |
| injury | 1/1 | 1.00 |
| property_damage | 4/4 | 1.00 |
| vehicle_collision | 8/8 | 1.00 |
| vehicle_damage | 4/4 | 1.00 |
| medical_bill_blank | 6/9 | 0.67（另 3 张 review） |

八个类别里七个 100% 误报；连最接近域内"单据/截图"风格的电子医疗票据也只有 6/9 直接过阈值，其余 3 张掉进 review 带。

---

## 3. 根因归因（全部有实验证据）

### 3.1 归因方法可信度

部署头是**标准化输入上的线性逻辑回归**，因此单张图的 logit 可精确分解为 `intercept + Σ coef_i·(x_i-mean_i)/scale_i`，可按证据块（L1 / L1b / CLIP global / CLIP tiles）加总；用同一套权重复算 41 张的 `forensic`/`combined` 概率与批量脚本自身输出**最大偏差 4.99e-05**，确认复现的是同一条管线（`work/diagnose_fpr.py`）。

### 3.2 `combined` 头的分组贡献（平均 logit 贡献）

| 证据块 | 域内真实（63 张 PNG） | 域内 AI（56 张 PNG） | **本测试集（41 张 JPEG）** |
| --- | --- | --- | --- |
| L1（47 维手工特征） | -0.053 | +0.059 | -0.090 |
| L1b（20 维相机/渲染域） | -0.019 | +0.021 | +0.004 |
| CLIP global（1024 维） | -0.940 | +1.217 | **+1.751** |
| CLIP tiles（3072 维） | -6.108 | +6.871 | **+5.284** |
| **头 logit 合计** | **-7.392** | **+7.896** | **+6.676** |

→ 误报几乎全部来自 CLIP 两块（tiles 为主）；域内手工特征（L1/L1b）对 `combined` 几乎不贡献。

### 3.3 反事实实验：换掉哪一块能救回来

把某一证据块替换为"域内 63 张真实图的中位数"后重算融合判决（其余块保持本图自身证据）：

| 反事实 | suspect | review | normal | FPR |
| --- | --- | --- | --- | --- |
| 基线 | 38 | 3 | 0 | **0.927** |
| 恢复 L1 容器特征（9 维） | 38 | 3 | 0 | 0.927 |
| 恢复整个 L1 块 | 38 | 3 | 0 | 0.927 |
| 恢复 L1+L1b | 38 | 3 | 0 | 0.927 |
| **恢复 CLIP global+tiles** | 0 | 30 | 11 | **0.000** |

→ 保留图片自身的 L1/L1b 证据、仅把 CLIP 证据换回域内分布，误报直接归零。**因果落点确定在 CLIP 证据块**。

### 3.4 域偏移量级：测试集整体落进"AI 簇"

| 头 | 域内真实 | 域内 AI | 本测试集真实 |
| --- | --- | --- | --- |
| combined logit（均值 / 极差） | **-7.39** / [-9.90, -5.55] | **+7.90** / [+4.56, +13.16] | **+6.68** / [+0.13, +10.57] |
| clip_domain logit（均值） | -4.69 | +5.34 | **+5.71** |
| forensic logit（均值） | -3.31 | +3.61 | +1.00 |

域内真实类的 logit 极差只有 4.35（63 张挤在极窄一簇），而测试集的整段分值区间 [+0.13, +10.57] 完全落在域内 AI 簇的分值带内。`combined`/`clip_domain`/`forensic` 在 119 套上 AUC 均为 1.0000 —— 高 AUC 在这里意味着决策边界实际上学成了**"是不是这 63 张真实图所在的 CLIP 小簇"**，而不是"是不是 AI 生成"。

### 3.5 对照实验：排除"缩放/JPEG 压缩"解释

把域内 63 张真实图按网页 CDN 的处理方式重做（长边→1067 px 再 JPEG q80，尺寸分布与本测试集一致），用同一套头与阈值重打分：

| 变体 | suspect | review | normal | FPR | combined 中位 logit | CLIP tiles 平均贡献 |
| --- | --- | --- | --- | --- | --- | --- |
| clean（原 PNG） | 0 | 12 | 51 | 0.000 | -7.30 | -6.11 |
| 仅缩放（PNG） | 0 | 9 | 54 | 0.000 | -7.17 | -5.86 |
| 仅 JPEG q80 | 0 | 18 | 45 | 0.000 | -6.29 | -5.08 |
| **缩放+JPEG q80（web_like）** | 0 | **48** | 15 | **0.000** | **-4.83** | -3.53 |

→ 网页式重编码**不会**造成"判成 AI"（FPR 仍为 0），但会显著侵蚀置信度（review 从 12 张涨到 48 张）。因此本次 92.7% 误报**不是压缩/格式伪影**，而是**内容与外观域的偏移**（真实事故照/食品照/票据截图这批网络图片，落到了域内真实簇之外）。

### 3.6 顺带纠正一个"看起来很像"的假设

L1 的 9 个容器特征在 JPEG 上确实塌陷（`nidat` 81→0、`hasSrgb` 0.38→0、`hasExif`/`hasItxt` 0.22→0、`bytes` 1.41 MB→0.22 MB），但它们对 `combined` 的贡献只有 ±0.03 logit 量级（容器组平均贡献位移 +0.0007），**不是本次误报的原因**。仓库自带的 `reports/robustness_matrix.json` 也显示 JPEG q50–q95 重编码后 FPR 仍为 0.0317，与此一致。不过它们仍是**泛化隐患**（见 6.4）。

---

## 4. 测试集可信度核查

- **文件与标签**：磁盘 41 张，索引 41 个唯一文件名，一一对应，无缺失、无多余；41 张标签唯一且全为 `real`（8 个文件名在索引里出现 2 次、来源与尺寸不同，磁盘只保留 1 份，41/41 尺寸至少与其中一条索引记录一致）。
- **无泄漏**：与 119 张训练样本做 256-bit phash 比对，最近一对相差 18 bit（≈93% 相似），无 ≤16 bit 的近重复 → 独立测试集成立。
- **内容抽查**：抽查 4 张（高速追尾现场、发霉面包、浙江省电子医疗票据等）均为真实拍摄照片或真实电子票据截图，其中追尾图带百度图片水印，未见生成痕迹；全部无 EXIF（CDN 重编码所致，符合预期）。
- **完整性缺口（需修）**：索引中的 `sha256_pixels` **无法用任何标准约定从磁盘文件复现**（RGB/RGBA/BGR 像素字节、灰度、原文件字节、PNG 重编码，41/41 全部对不上）。该字段因此不能用于逐像素溯源校验；标签可信度目前只依赖索引声明的来源 URL。建议补一个可复现的哈希脚本并重算该字段。

---

## 5. 影响评估

- 交付文档里"真实误报率 ≤3%"的结论目前**只在同域成立**：119 套 OOF FPR 3.17%、归档 `verify-24` FPR 0.0000。换成本次这种"网络来源真实照片"，FPR = **92.7%**。
- 训练/标定语料（命题 xlsx 的 63 张真实 PNG）与真实业务输入（网页/理赔 App 上传的 JPEG）**不是同一个分布**，当前阈值与 CLIP 头把同域窄簇之外的输入一律推向 AI 侧。
- 这类失效在比赛中是**验收级风险**：真实误报率是最容易在隐藏测试集上被击穿的指标，且样本越真实、越"网络化"越容易触发。

---

## 6. 建议（按优先级，本次未实施）

1. **把这类测试集纳入标定与验收口径**：阈值目前只在同域 63 张真实图上按目标 FPR 选取，等于同分布内自证。至少要用"网络来源真实照片"做真实侧标定/验收。
2. **重训或重标定 CLIP 相关头**（`combined`、`clip_domain`）：加入真实网页图增强（CDN 缩放、JPEG 重编码、水印、无 EXIF），或在检测到 OOD 迹象时降权 CLIP 证据。
3. **增加 OOD/弃权机制**：对"离真实簇过远的输入"输出 review 而不是 suspect。现状 review 带（0.0364–0.6743）在同域有效，在 OOD 上几乎被穿透（41 张里 38 张直接越过高阈值）。
4. **显式标注容器指纹特征**：`nidat/hasSrgb/hasExif/hasItxt/hasGama/hasChrm/bytes` 度量的是封装格式而非"AI 性"，在非 PNG 输入上应禁用或单独建模，避免 `forensic` 头（权重 0.25）把"换了封装"当作证据。
5. **修复后回归 119 套**，确认同域指标不回退，并冻结新的阈值产物（`models/fusion.json`）与证据文件。

---

## 7. 证据文件

| 文件 | 内容 |
| --- | --- |
| `repo/reports/batch/real41/metrics.json` / `metrics.md` | 本次评测的指标、协议说明、归档质检、argv 与时间 |
| `repo/reports/batch/real41/per_sample.json` | 41 张逐样本：判决、置信度、分模型分、逐段耗时 |
| `repo/reports/batch/real41/failures.csv` | 失败清单（38 false_positive + 3 review） |
| `repo/reports/batch/real41/samples/*.json` / `*.html` | 41 份完整报告（schema 1.1.0）+ 38 份 FP 单文件 HTML |
| `repo/reports/testset_real_41_diagnosis.json` | 归因：分组贡献、反事实、L1 特征级位移、域内/域外对比 |
| `repo/reports/testset_real_41_reprocessing.json` | 63 张域内真实图 web 式重编码对照实验 |
| `work/check_testset_real.py` | 索引/标签/尺寸/phash 一致性核查、生成标签文件 |
| `work/diagnose_fpr.py` | 归因与反事实脚本 |
| `work/perturb_indomain.py` | 重编码对照实验脚本 |
