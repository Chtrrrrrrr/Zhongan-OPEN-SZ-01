# 批量评测报告

- 生成时间：2026-10-11T03:41:57Z
- 样本数：24（带标签 24）
- 检测器：device=auto clip=yes forensic=yes fusion=yes
- 预热：{'done': True, 'ms': 13658.5, 'sample': 'data\\images\\001.png'}
- 总耗时：46.44 s
- 输入：data\images

## 指标表

| 指标 | 值 |
| --- | --- |
| Accuracy（review→real，与 reports/metrics.json 同口径） | 1.0000 |
| Accuracy 95% CI | [1.0, 1.0] |
| FPR | 0.0000 |
| FPR 95% CI | [0.0, 0.0] |
| TPR（Recall） | 1.0000 |
| Precision | 1.0000 |
| AUC（连续置信度） | 1.0000 |
| 覆盖率（非弃权占比） | 0.6667（16/24） |
| 保守下界（弃权视为错误） | 0.6667 |
| 混淆矩阵 | TP=11 FP=0 FN=0 TN=13（AI=11 / real=13） |

仅在已决样本上（16 条）：Accuracy=1.0000 / FPR=0.0000 / TPR=1.0000 / TP=11 FP=0 FN=0 TN=5

## 三档结论分布

- suspect（疑似生成/篡改）：11
- normal（正常）：5
- review（待复核）：8（成因：{'model_disagreement': 7, 'uncertainty_band': 1}）

## 失败案例清单

共 8 条。

| # | 文件 | 标签 | 结论 | 置信度 | 类型 | 首因 |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 057.png | real | review | 0.0637 | review | fused score 0.064 in uncertainty band (0.036, 0.674) |
| 13 | 063.png | real | review | 0.0265 | review | fused score 0.027 <= low threshold 0.036 |
| 14 | 068.png | real | review | 0.0467 | review | fused score 0.047 in uncertainty band (0.036, 0.674) |
| 17 | 083.png | real | review | 0.0508 | review | fused score 0.051 in uncertainty band (0.036, 0.674) |
| 18 | 088.png | real | review | 0.0366 | review | fused score 0.037 in uncertainty band (0.036, 0.674) |
| 20 | 098.png | real | review | 0.016 | review | fused score 0.016 <= low threshold 0.036 |
| 21 | 104.png | real | review | 0.0138 | review | fused score 0.014 <= low threshold 0.036 |
| 23 | 114.png | real | review | 0.0114 | review | fused score 0.011 <= low threshold 0.036 |

## 归档质检（报告 schema）

- report schema 版本：`1.1.0`
- 必备标记：explanation, latency_ms.L1b, localization.available, versions.artifact_sha256
- 标记齐全：{'explanation': 24, 'localization.available': 24, 'versions.artifact_sha256': 24, 'latency_ms.L1b': 24}
- 结论：全部样本通过

## 逐样本报告

逐样本 JSON：`samples/<name>.json`（24 份）；逐样本明细：`per_sample.json`。
