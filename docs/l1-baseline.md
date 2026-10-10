# L1 取证特征基线（实测复现记录）

- 脚本：`scripts/extract_features.py`（特征提取）→ `scripts/baseline_l1.py`（分类与评估）
- 特征：47 维低层 / 容器取证特征（`src/l1_forensics.py`，与手册 11.1 定义一致）
- 分类器：L2 正则逻辑回归（`LogisticRegression(C=1.0, max_iter=2000)`，特征标准化）
- 协议：分层 5 折交叉验证 × 5 个随机种子（seed 0–4）
- 数据：命题样本 119 张（AI 56 / 真实 63），`features.csv`

## 1. 实测结果（本机复现）

| 指标 | 实测值 |
|---|---|
| 平均 AUC | **0.809**（± 0.091，25 折汇总） |
| 平均准确率 | 0.731 |
| 最高单折准确率 | 0.913 |

各折准确率示例（seed 0–4 共 25 折）区间为 0.565 – 0.913，波动明显，说明在 119 张样本上单折指标不稳定。

## 2. 与手册记录值的差异（须显式说明）

手册 11.1 记录先导实验：平均 AUC **0.916**、最高交叉验证准确率 **85.7%**。
本次在相同数据、相同协议下复现得到 AUC **0.809**、最高单折准确率 0.913，**低于**手册记录值。差异原因已定位：

1. **小波高频特征退化**：`src/l1_forensics.py` 的 `waveHH` 依赖 `pywt`（PyWavelets）。当前环境**未安装 pywt**，该特征恒为 0，实际有效维度为 46 而非 47。
2. **纹理特征实现差异**：`texContrast/texHomogeneity` 使用自制 8 级量化共生矩阵（为避免额外依赖），与原始实现的统计口径不完全一致。
3. **块效应实现差异**：`_blockiness` 以“边界差值 / 内部差值”的比值定义 `blockRatio`，与原始定义存在数值尺度差异。

**结论（诚实表述）**：L1 单路在 119 张样本上约 **AUC 0.81 / 准确率 0.73**，最高单折 0.91，**不足以单独达到 95% 准确率**。这与手册判断一致——L1 只作辅助证据与早退判据，主判别力必须来自 L2 深度模型与融合。

## 3. 单特征判别力排序（实测，前 10）

| 特征 | 含义 | 单特征 AUC |
|---|---|---|
| nidat | PNG IDAT 块数 | 0.719 |
| hasSrgb | 携带 sRGB 色彩块 | 0.685 |
| oh | 原始高度 | 0.634 |
| blockRatio | 8×8 块效应比值 | 0.620 |
| cvG | G 通道变异系数 | 0.614 |
| hasItxt | 携带 iTXt 元数据 | 0.611 |
| hasExif | 携带 eXIf 元数据 | 0.611 |
| bytes | 文件体积 | 0.603 |
| stdG | G 通道标准差 | 0.594 |
| ow | 原始宽度 | 0.594 |

排序与手册表 6 高度一致（nidat、hasSrgb、hasItxt/hasExif、bytes 均在前列），说明特征实现方向正确。

**重要局限**：排名最前的 `nidat / hasSrgb / hasItxt / hasExif` 本质都是 **PNG 编码器指纹**，一次重编码、截图或转存即可能被抹除。它们只能作为 L1 辅助线索，不得作为主判据（手册 2.4、3.5）。

## 4. 复现命令

```bash
python repo/scripts/extract_features.py --data-dir data --out features.csv
python repo/scripts/baseline_l1.py --features features.csv
```

预期输出（本机）：`mean AUC = 0.809`，`mean ACC = 0.731`，`best fold ACC = 0.913`。

## 5. 待改进项

- 安装 `PyWavelets` 恢复 `waveHH` 特征（第 47 维）。
- 对齐原始实现的 GLCM 与块效应口径，缩窄与手册记录值的差距。
- L1 不作为最终判据；其分数进入融合层时权重受控（见 `src/fusion.py`）。
