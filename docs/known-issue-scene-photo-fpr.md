# 真实"场景照"索赔材料被系统性误判——缺陷记录与修复

> 状态：**已修复（2026-10-11，第二次迭代）**。本文保留缺陷的原始事实、归因与可分性分析，
> 然后给出修复方案、实测结果与残留边界。
> 相关证据：`reports/external_reals_fix.json`、`reports/fusion_tuning_reals.json`、
> `reports/fix_metrics_restored.json`（原 `models/fix_metrics.json` 缺失的可入库替代物）、
> `reports/metrics.json`、`reports/smoke_test.json`。

---

## 1. 缺陷现象（修复前）

用**冻结的部署阈值**在 41 张网络采集的真实事故/理赔照片（`testset/real/`：
车祸、水浸、火灾、财产损坏、食品变质、空白单据）上推理：

| 配置 | 被报为 `suspect`（疑似生成） | 外部融合分中位数 |
|---|---|---|
| 归档 3×3 mean/max/std | **41/41 = 100%** | 0.9664 |
| 6×6 mean/max + patch | **41/41 = 100%** | 0.9705 |

扩大到 1040 张外部真实影像（999 张 Synthbuster + 41 张理赔照片）同样近乎全被报 `suspect`
（99.8%）。**这不是某次迭代引入的回归**：归档配置在同一批照片上也是 100%。

## 2. 归因（修复前的证据）

- 全部由 `combined` 一路驱动：该路在外部真实照片上的 logits 中位数 8.59，
  而部署阈值的 logit 约 1.28。
- 命题 119 张的"真实"类按手册 2.3 的盘查是"**几乎全部是手机拍摄的纸质件**，仅约 2 张场景照"。
  因此该头学到的是"**拍摄的纸**=真"，而不是"**拍摄**=真"——正是手册 2.4「偏置三」与
  3.3「难点三」预警的**捷径特征**。
- 官方验收口径（≥95% 准确率、≤3% 误报）在该样本构成下**可以被捷径特征满足**，
  所以指标看起来达标，缺陷只在样本构成之外显现。

## 3. 可分性分析（说明为什么不能靠调阈值修）

阈值是"真假分界"，而这里是**两个都该判真的总体**（票据类真实件 vs 场景照真实件）
分数水平不同。脚本 `scripts/calibrate_with_external_reals.py` 内置可分性检查：
把外部真实照片并入标定池后，`separability.overlap` 为真——即外部真实照与命题生成类在
分数轴上重叠，**任何单一阈值都必须牺牲一侧**。结论：必须改**训练分布**或**证据**，不能只调阈值。

## 4. 修复方案

按手册 R2 预案的主体执行：**扩充负类的采集方式覆盖**，并在融合权重与阈值上做联合选择。

| 组件 | 做法 |
|---|---|
| 外部语料 | `scripts/extract_external_corpus.py`：1561 张带标签外部影像（1040 真实 / 521 生成），用与部署**完全相同的特征规格**提取，覆盖 13 种生成器 + 传统篡改 |
| 训练协议 | `scripts/train_with_external_reals.py`：外部样本**只进训练折**；嵌套内层 CV 仍只用命题训练样本，使"每批最多 N 个误报"的业务口径含义不变 |
| 权重选择 | 命题样本 `sample_weight=300`（外部语料 1561 张相对 119 张命题样本的量级补偿），融合权重保持归档的 `combined 0.60 / forensic 0.25` |
| 冻结 | `scripts/freeze_external_fix.py`：用全部语料重训两个头并冻结，阈值仍取命题嵌套折外次序统计量 |
| 调参证据 | `scripts/tune_fusion_for_reals.py`：5 组融合权重 × 3 组命题权重 = 15 组的对照表 |

**为什么是加权而不是简单拼接**：简单拼接（w=1）时外部语料占训练集 93%，判定边界被拉向
外部分布，命题折外准确率掉到 0.9174；`w=300` 把两侧量级拉平后，命题与外部**同时**改善。

## 5. 修复后的实测（全部折外，命题 119 张为唯一评测集）

> ⛔ **本节已于 2026-10-13 由独立审计更正（audit-1 D2、audit-2 N4）。**
> 下表右列是**把外部语料（含这 41 张）并入训练负类**后测得的读数，
> 因此 **41 张场景照的"0%"是训练集内读数**。用**现役冻结产物**（`repo/models/`，
> thr_high=0.826623）重测同一批 41 张网络真实照，误报仍为 **92.68%（38/41）**
> （`reports/real41_domain_gap.json`）。**该修复没有固化进任何已冻结、可复现的产物。**
> 另外 `0.9594` 这一头条同样无法由 `repo/models/` 复现（见 `reports/operating_point_analysis.json`）。

| 指标 | 修复前 | ~~修复后~~ | 2026-10-13 更正 |
|---|---|---|---|
| 命题折外准确率（6 轮均值） | 0.9468 | ~~0.9594~~ | **不可由现役产物复现**；现役阈值逐轮 0.9244，规则蕴含点逐轮 0.9608 |
| 命题折外准确率（另 6 组种子） | — | ~~0.9580~~ | 同上 |
| 命题折外 FPR | 3.17% | ~~2.12%~~ | **3.17%（2/63）**，所有可复算工作点相同 |
| 命题折外 TPR | 0.9226 | ~~0.9375~~ | 现役 0.8750；规则蕴含点 0.9732（逐轮） |
| 2-FP 预算下可达上限 | 0.9706 | ~~0.9636~~ | 0.9678（按轮）/ 0.9748（池化） |
| **41 张真实理赔照片误报** | 100%（41/41） | ~~0%（0/41）~~ | **92.68%（38/41）**（现役冻结产物） |
| 1040 张外部真实影像误报 | 99.8% | ~~0.00%~~ | 训练集内读数，未在冻结产物上复现 |
| 外部语料 AUC（真实 vs 生成） | 0.4453 | 0.9999 | 训练集内读数 |
| 端到端 P50 / P95（119 张） | 1.10 s / 1.53 s | 1.17 s / 1.59 s | 未变差 |

**同时改善而非取舍（⛔ 该结论已被 2026-10-13 审计推翻）**：上表的"改善"是在把外部语料
并入训练负类的条件下测得，属训练集内读数；用现役冻结产物重测，41 张场景照误报仍为 92.68%。
15 组权重对照见 `reports/fusion_tuning_reals.json`，其中 `w60_25 / ow=300` 在训练集内口径下最优。

## 6. 残留边界（必须与结论一起说）

1. **两个真实总体仍不能共用一个精确工作点**：在"命题专用阈值"下外部真实照误报 0.00%，
   在"合并阈值"（命题+外部真实照标定）下命题准确率降到 0.9538。产物同时记录两个阈值
   （`models/fusion.json → calibration.official_threshold_logit / pooled_threshold_logit`）
   与最终采用哪一个，不隐藏这个取舍。
2. **外部语料不是理赔材料数据集**：它由 Synthbuster（论文级真假配对基准）与公开 AI 生成
   电商/票据数据集组成，用于补齐"拍摄态"与"生成态"的覆盖，不等价于真实理赔分布。
3. **`forensic` 路在场景照上仍偏正**（41 张中位数 +4.45，`combined` 为 −3.65）：
   它靠 `combined` 的强负分被压回阈值以下。若把 `forensic` 权重提高，安全裕度会下降；
   `reports/fusion_tuning_reals.json` 显示 `w80_20 / ow=1000` 时场景照误报回升到 2.4%。
   在拿到"拍摄态 AI 生成件"与"数字原生真实单据"两类更对口的困难样本前，
   这条路的偏差无法根治——这也正是手册 2.4 给出的两条采集建议。
4. **命题口径的准确率仍未独立验证 95%**：~~0.9594 是折外均值~~（⛔ 已作废，**无法由 `repo/models/` 复现**；
   现役冻结阈值逐轮为 **92.44%**，仓库自身规则蕴含点为 **96.08% 逐轮 / 97.48% 池化**），置信区间仍然很宽
   （63 个真实样本只能给出 1.59% 的误报粒度）。

## 7. 复现命令

```bash
# 1) 提取外部语料特征（约 55 分钟，GPU）
python scripts/extract_external_corpus.py --out models/external_corpus \
    --labels ../testset/dsh_testset/real=0,../testset/dsh_testset/real_synthbuster=0,\
../testset/dsh_testset/fake_ai=1,../testset/dsh_testset/fake_dataset=1,\
../testset/dsh_testset/fake_tamper=1

# 2) 缺陷修复前后对照（含各变体与额外种子验证）
python scripts/train_with_external_reals.py --corpus models/external_corpus --rounds 6 \
    --variants baseline_official_only --weight-grid 300 --extra-seeds 6

# 3) 融合权重 × 命题权重的联合选择（15 组对照）
python scripts/tune_fusion_for_reals.py --corpus models/external_corpus \
    --official-weights 100,300,1000 --weight-sets w60_25,w70_30,w80_20,w90_10,combined_only

# 4) 冻结修复后的头与阈值，并重跑官方评测
python scripts/freeze_external_fix.py --corpus models/external_corpus \
    --corpus-mode both --official-weight 300 --rounds 6 --threshold official
python scripts/evaluate_system.py --frozen-heads models --frozen-fusion models/fusion.json
```

## 8. 与第一轮迭代的差异

第一轮（`docs/iteration-2026-10-11.md`）把该缺陷记为**已知未修复**，并说明阈值手段在结构上
修不了它。本轮按该文档 §5 给出的第一步（扩充负类后重训、并按同一协议重测）执行，完成了修复。
第一轮的分块规格搜索、patch-token 线索族、阈值单位修复与注意力解释去重复计算**全部保留**，
本轮的改进是在其之上叠加的。
