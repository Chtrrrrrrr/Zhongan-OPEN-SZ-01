# demo/webui_assets —— WebUI 静态资产（由 webui-engineer 维护）

本目录是 `demo/webui.html`（`api/server.py` 的首页）使用的静态资产。所有内容都是
**从仓库内既有产物离线导出**的，不含任何编造数据；每个数字都能回溯到来源文件。

## 目录

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `oof_119.json` | 119 套命题样本的逐样本折外记录：真实标签、折外融合分、二值判定、三档结论、各路温度标定分、所属折与折阈值，以及汇总统计（混淆矩阵 / 三档分布 / 护栏指标） | `<ws>/models/oof_bundle_119.json`（`scripts/evaluate_system.py` 同协议导出）+ `<ws>/data/manifest.json` + `reports/metrics.json` |
| `thumbs/NNN.svg` | 150 px 缩略图（内部是一张 base64 JPEG 的 `<image>`），用于样本库网格 | `<ws>/data/images/NNN.png` |
| `build_assets.py` | 上述资产的生成器（可重复执行，`--check` 校验一致性） | —— |
| `verify_webui.py` | WebUI 端到端验收自检（stdlib，30 项检查） | 对运行中的服务发请求 |
| `verify_webui_browser.mjs` | 真实浏览器（Chrome + CDP，无 npm 依赖）验收自检：渲染真实数据、样本弹窗、图片实时检测、超时降级、布局无溢出 | 对运行中的服务打开页面 |

## 指标看板证据卡清单（14 张，全部 `reports/` 下实时读取）

| # | 证据卡标题 | 文件 | 渲染方式 |
| --- | --- | --- | --- |
| 1 | 特征消融 | `reports/feature_ablation.json` | 专用表 |
| 2 | 融合规则消融 | `reports/fusion_ablation.json` | 专用表 |
| 3 | NPR 通道消融 | `reports/npr_ablation.json` | 专用表 |
| 4 | 鲁棒性矩阵 | `reports/robustness_matrix.json` | 专用表 |
| 5 | 冷启动与端到端冒烟 | `reports/smoke_test.json` | 专用键值 |
| 6 | 置信度校准 / ECE | `reports/calibration.json` | 通用展开 |
| 7 | 成员相关性 / 错误重叠 / 留一法 | `reports/member_analysis.json` | 通用展开 |
| 8 | 阈值 bootstrap 稳定性 | `reports/threshold_stability.json` | 通用展开 |
| 9 | 分层诊断拆解 | `reports/stratified_analysis.json` | 通用展开 |
| 10 | 各环节耗时与显存画像 | `reports/resource_profile.json` | 通用展开 |
| 11 | **跨生成器泛化（整类生成器从训练移除）** | `reports/cross_generator.json` | 专用表（对照 AUC + 3 个留出 fold + 逐生成器 recall + 方法论说明） |
| 12 | **AEROBLADE 消融（可选成员，结论：剔除）** | `reports/aeroblade_ablation.json` | 专用表（预注册门槛 / 单路变体 / 加入融合 Δ / 与各成员相关性 / 判定） |
| 13 | **分维度指标（人工标注）** | `reports/stratified_manual.json` | 专用表（外观+内容两维 + 捷径检验 + 两处错误归因） |
| 14 | 错误分析清单 | `reports/error_analysis.csv` | CSV 摘要表 |

* 第 11–13 张是第二轮合规审计（F1–F9）后新增的证据，页面同时把关键结论提到功能矩阵 ⑤ 卡片与
  ④ 指标看板的「分维度指标」卡片（含「误报预算应按内容类别分别管控」的工程结论）。
* 任一张缺失都只在该卡内显示「未生成 / 不可用（HTTP 404）」，不影响其余卡片；页脚与看板顶部
  会显示 `已加载 N/14` 的计数。
* 渲染原则：已知结构用专用渲染器（数值条形 + 表格），未知/新增结构退回通用展开（截断 + 原始 JSON 折叠）。

## 报告回取（与归档布局解耦）

样本库点击卡片后，页面按以下顺序探测该样本的完整报告，命中即渲染，全部 404 才降级为说明文案：

1. `GET /api/v1/report/{NNN}` —— 服务端回退链，依次查 `reports/api/<id>.json` → `repo/reports/<id>.json`
   → 工作区 `<ws>/reports/<id>.json` → 同名 `.html`（与 `GET /api/v1/health` 的 `report_search_dirs` 同源）
2. `GET /reports/{NNN}.json` —— 静态回退
3. `reports/batch/verify-119|verify-24|verify-balanced24|full119/samples/{NNN}.json` —— 各批量评测归档兜底

因此 `reports/batch/` 被清理或重建（例如归档 schema 刷新）不会影响演示：只要归档报告在服务端
回退链的搜索目录内即可取到，页面无需改动。

## 为什么缩略图是 `.svg`

仓库根 `.gitignore` 全局忽略 `*.png / *.jpg / *.jpeg / *.webp / *.gif`（只对 `assets/**` 放行），
位图缩略图不会被提交，克隆仓库后页面会出现坏图。`.svg` 属文本资产、不受该规则影响，
浏览器渲染结果与直接引用 JPEG 一致（就是一张 JPEG 的 data URI），因此这里用 SVG 包装。

## 口径与一致性

* `oof_119.json` 里的折外判定与 `reports/metrics.json → oof_overall` **逐项一致**：
  `accuracy=0.958 / fpr=0.0317 / tpr=0.9464 / precision=0.9636`，
  混淆矩阵 `TP53 FP2 FN3 TN61`，三档分布 `suspect 55 / review 32 / normal 32`。
* 评测口径：`pred = 1 if 折外融合分 ≥ 该折 high 阈值`，**review（弃权）计入负类**，
  因此真实样本的 review 计入 TN、AI 样本的 review 计入 FN。这一点在页面上也做了显式说明。
* `oof_119.json.sources` 记录了三个来源文件的路径与 sha256，便于审计比对。
  **注意这是「磁盘字节」的 sha256**：仓库 `core.autocrlf=true`，`reports/metrics.json` 工作区为 CRLF、
  git blob 为 LF，因此字节 sha 与 `git show HEAD:reports/metrics.json` 的 blob sha 不同属正常
  （归一化换行后一致，`git diff` 为空）。审计比对时请统一口径。
* 同一行尾效应也落在接口对外报的哈希字段上：`GET /api/v1/version` 的 `code.code_hash`（对跟踪文本文件求字节哈希）
  与 `metrics.sha256` 在本机（CRLF）与干净 clone/Linux（LF）上会不同；模型权重 `*.pt/*.joblib`、
  逐样本报告里的 `input_sha256`（图片/视频字节）不受影响，`fusion.json` 未被 git 跟踪也不受影响。
  跨环境比对哈希前请 `git show HEAD:<path>` 或先 CRLF→LF 归一化。
* `--check` 的判定口径：**数据内容**必须一致（samples / counts / deployed_metrics / protocol …），
  来源指纹漂移只打印提示不算失败（他人重跑评测脚本、或上述 CRLF 差异都不应让检查变红）。
  来源文件被重新生成后（例如 `reports/metrics.json` 由队长提交 c8c8184 更新了 latency 分位数），
  重跑 `build_assets.py` 即可刷新 `sources` 里的指纹记录。

## 重建

```bash
python demo/webui_assets/build_assets.py          # 生成/更新全部资产
python demo/webui_assets/build_assets.py --check  # 仅校验（CI 可用，不一致时退出码非 0）
```

生成器只读 `<ws>/models/`、`<ws>/data/`，只写本目录，不触碰其他成员的文件。

## 验收自检

```bash
python api/server.py --port 8080                 # 终端 A：启动服务（默认真实模型 + 预热 ~11 s）
python demo/webui_assets/verify_webui.py         # 终端 B：27 项端点/页面/资产检查
```

`verify_webui.py` 覆盖：首页 200 且含 5 个板块锚点、非回退页、无外部 CDN、静态资产（119 条折外记录 /
缩略图 / README）可达、`/api/v1/health|version|metrics` 可达、样本完整报告数据源可达、
`/api/v1/report/{id}` 404 统一错误体、**图片实时检测（原始字节模式）**返回结论与定位图、
落盘报告可按 id 回取、**超时降级为 200+review+degraded**、批量端点返回 count/summary/results。

页面的浏览器侧验证（渲染真实数据、样本弹窗、图片/视频检测、批量、超时降级、布局无横向溢出）
由 WebUI 作者在 Chrome（headless + CDP）中执行，结论记录在 t4 任务交付说明中。
