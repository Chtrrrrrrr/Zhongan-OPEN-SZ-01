# 批量评测归档（batch archive）

本目录存放**可入库的批处理评测归档**。合规审计 F9 要求"归档的逐样本报告必须是当前
report schema、且不能新旧两套混放"，因此这里只保留一份**小规模但完整**的归档：

| 归档 | 样本数 | report schema | 入库 | 说明 |
| --- | --- | --- | --- | --- |
| `verify-24/` | 24（11 ai + 13 real） | **1.1.0** | ✅ | 本目录唯一入库归档，含逐样本报告与全部聚合产物 |
| `verify-119/` | 119（56 ai + 63 real） | **1.1.0** | ❌（`.gitignore`：`reports/batch/*` 仅放行 `README.md` 与 `verify-24/`） | 本机全量运行：供 WebUI 样本库覆盖全部序号 + 全量指标复现；体量 58 MB，故不入库 |

`reports/api/`（REST 服务落盘的运行时报告与请求日志）不入库：运行时产物，且可由任意一次
`POST /api/v1/detect/*` 重新生成。

---

## 1. 归档 schema 与质检（F9）

- **report schema 版本：1.1.0**（单一真源 `repo/src/report.py: REPORT_VERSION`；
  运行中的服务通过 `GET /api/v1/version` 的 `report_schema_version` 暴露同一数值，
  `GET /api/v1/health` 亦给出该值，避免代码与流水线漂移）。
- 该版本相对旧版新增/必备的 4 个标记，`verify-24` 中 **24/24 份逐样本报告全部具备**：

| 标记 | 含义 | verify-24 覆盖 |
| --- | --- | --- |
| `explanation` | 关注区域解释块（Grad-CAM 不可用时给出替代方法说明） | 24/24 |
| `localization.available` | 定位分节的可用性标记 | 24/24 |
| `versions.artifact_sha256` | 权重/阈值产物哈希表（可追溯） | 24/24 |
| `latency_ms.L1b` | L1b 域取证独立计时行 | 24/24 |

上述质检结论固化在 `verify-24/metrics.json` 的 `archive_qc` 字段中：

```json
"report_schema_version": "1.1.0",
"archive_qc": {
  "applicable": true, "n_samples": 24,
  "markers_expected": ["explanation", "latency_ms.L1b", "localization.available", "versions.artifact_sha256"],
  "markers_present": {"explanation": 24, "localization.available": 24,
                      "versions.artifact_sha256": 24, "latency_ms.L1b": 24},
  "missing_examples": {}, "unreadable": [], "all_pass": true
}
```

自查/回归（两条命令任一即可重复验证）：

```bash
python scripts/batch_evaluate.py --selftest        # 含归档质检标记的单测
python tests/test_api.py --fast                    # 含 archive.schema_markers 检查（读本归档逐样本 JSON）
```

## 2. 生成命令与环境

```bash
# 在仓库上一级（工作区根）执行；--limit 24 --spread 为确定性均匀抽样（跨类均衡、可复现）
python repo/scripts/batch_evaluate.py --dir data/images --labels data/labels.csv \
       --limit 24 --spread --out repo/reports/batch/verify-24
```

| 项 | 值 |
| --- | --- |
| 生成时间（UTC） | 2026-10-11T03:41:57Z |
| 总耗时 | 46.44 s（含 13.66 s 启动预热） |
| 模型目录 | `models/`（CLIP ViT-L/14 + forensic/combined LR 头 + fusion.json） |
| 设备 | CUDA（RTX 5070 Laptop），python 3.14.7 |
| 抽样方式 | 输入 119 套按文件名排序后 `linspace` 均匀取 24 套（11 ai + 13 real） |
| 归档体量 | 11.71 MB / 28 个文件 |

> 抽样不落在固定前 N 套上：`--spread` 让归档同时覆盖两类样本，避免"单类样本导致
> FPR/TPR 分母为 0、指标不可评估"的情况（该情形脚本会显式标注 `不可评估`，不伪造 0）。

## 3. 归档内容

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 指标 + 协议说明 + 归档质检 + 生成器 argv/时间 + 与 `reports/metrics.json` 的离线参考对照 |
| `metrics.md` | 同上的 Markdown 版（含归档质检分节） |
| `failures.csv` | 失败案例清单（`false_positive`/`false_negative`/`review`/`decode_error`/`error` + 决策首因） |
| `per_sample.json` | 逐样本明细（结论、置信度、模型分、逐段耗时、标签、失败类型） |
| `samples/<name>.json` | **逐样本完整报告**（24 份，当前 schema 1.1.0；与 REST 接口落盘报告同一 schema） |

### 指标摘要（24 套，主口径：`review` 计入 negative/real）

```
Accuracy (review→real) 1.0000    FPR 0.0000    TPR 1.0000    Precision 1.0000
AUC (连续置信度)        1.0000    覆盖率(非弃权) 0.6667（16/24）  保守下界(弃权=错误) 0.6667
混淆矩阵                TP=11  FP=0  FN=0  TN=13   （AI=11 / real=13）
三档分布                suspect=11  normal=5  review=8
review 成因             模型分歧 7、不确定带 1
耗时                    p50=1187.7 ms  p95=1725.3 ms  max=1917.4 ms  mean_wall=1266.7 ms
```

> 口径提醒：本归档用**模型侧冻结阈值**（`models/fusion.json` 的 `low=0.0364 / high=0.6743`）
> 直接在样本上推理；`reports/metrics.json` 的 `accuracy=0.958 / fpr=0.0317 / tpr=0.9464`
> 是**分层 5 折 OOF**口径（每折阈值只在训练折上按目标 FPR≤3% 选出）。两者不可直接互比，
> `metrics.json.supervised.reference_offline_oof` 同时列出离线参考值。

## 4. 大样本运行（本机留存、不入库）

```bash
# 119 套全量一键评测（实测 139.4 s，CUDA）；产物为本地目录，不提交
python repo/scripts/batch_evaluate.py --dir data/images --labels data/labels.csv \
       --out repo/reports/batch/verify-119
```

`verify-119/` 已在 2026-10-11T03:48:10Z 用当前代码重跑（schema **1.1.0**，119 份逐样本报告
`archive_qc.markers_present` 四类标记均 119/119、`all_pass=true`，体量 58 MB），
用途：WebUI 样本库（`demo/webui.html` 按 `reports/batch/<archive>/samples/NNN.json` 探测）
需要覆盖全部 119 个序号。它**不随仓库提交**——干净副本里只有 `verify-24/` 的 24 个序号；
需要全量时按下述命令一键重建，或让页面对 `GET /api/v1/report/NNN`（已支持回退到工作区
`reports/`）取数。

该次全量运行的指标（主口径：`review` 计入 negative/real）：

```
Accuracy 1.0000   FPR 0.0000   TPR 1.0000   AUC 1.0000
混淆矩阵 TP=56  FP=0  FN=0  TN=63     覆盖率 0.7899（94/119）  保守下界 0.7899
三档分布 suspect=56  normal=38  review=25（成因：模型分歧 22、不确定带 3）
耗时     p50=1030.3 ms  p95=1522.2 ms  max=1958.8 ms（含 11.89 s 预热，总 139.41 s）
```

## 5. 变更记录

- 2026-10-11（t8 / F9）：删除旧 schema 归档 `verify-119/`、`verify-balanced24/`（缺
  `latency_ms.L1b`），重建 `verify-24/` 为 schema 1.1.0；同时删除 `reports/api/` 下 18 份
  缺 `versions.artifact_sha256` 的旧 schema 运行时报告；`.gitignore` 改为
  `reports/batch/*` + 仅放行 `reports/batch/README.md`、`reports/batch/verify-24/`，
  并移除已失效的 `robustness_smoke.json` / `video_smoke.json` 白名单行（F7）。
