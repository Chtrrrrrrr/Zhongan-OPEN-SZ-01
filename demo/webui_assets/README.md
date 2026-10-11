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
  git blob 为 LF，因此字节 sha（`398dfa4f…`）与 `git show HEAD:reports/metrics.json` 的 blob sha
  （`bc412ced…`）不同属正常——归一化换行后两者一致，`git diff` 为空。审计比对时请用同一种口径。
* `--check` 的判定口径：**数据内容**必须一致（samples / counts / deployed_metrics / protocol …），
  来源指纹漂移只打印提示不算失败（他人重跑评测脚本、或上述 CRLF 差异都不应让检查变红）。

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
