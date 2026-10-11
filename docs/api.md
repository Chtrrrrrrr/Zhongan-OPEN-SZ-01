# 对外接口说明（OPEN-SZ-01 REST 服务）

本文件描述 `api/server.py`（版本化 REST 服务）与 `scripts/batch_evaluate.py`（批处理一键评测脚本）
的对外契约，对应验证手册 **16.1 服务化 / 批处理** 与 **16.2 模型常驻、预热、请求日志**、
**16.3 降级策略** 的要求。

- 服务实现：`api/server.py`（**只用 Python 标准库**：`http.server` / `json` / `threading` /
  `hashlib` / `multipart 手工解析`，零额外依赖，可直接运行）
- 接口文档（机器可读）：运行中实例的 `GET /api/v1/openapi.json`（OpenAPI 3.0.3，覆盖全部端点）
- 端点自测：`python tests/test_api.py`（或 `python api/server.py --selftest`）

> 结论口径：所有响应中的 `verdict` 为**材料层面**的技术判定
> （`normal` 正常 / `review` 待复核 / `suspect` 疑似生成或篡改），不构成业务欺诈定性。

---

## 1. 启动

```bash
# 主交付：stdlib 服务（默认加载真实模型并启动预热）
python api/server.py --port 8080

# 无模型占位模式（自测 / CI / 无权重环境；端点契约完全一致）
python api/server.py --port 8080 --no-model

# 启动并跑通端点自测后退出
python api/server.py --selftest
python api/server.py --selftest --no-model
```

常用参数（完整列表见 `python api/server.py --help`）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8080` | 监听地址/端口；非本机监听会自动禁用服务端路径输入 |
| `--no-model` | 关 | 占位实现：端点可用、结论固定 `review`（用于契约自测） |
| `--no-warmup` | 关 | 跳过启动预热（首个请求将承担 10~20 s 冷启动） |
| `--warmup-image` | `data/images` 首张 | 指定预热样本 |
| `--device` | 自动 | `cpu` / `cuda` |
| `--timeout-ms` | `30000` | 单请求默认超时；超时返回 `review` 降级结论并标注 |
| `--max-timeout-ms` | `300000` | `?timeout_ms=` 的上限 |
| `--batch-timeout-ms` | `600000` | 批量端点总预算 |
| `--max-batch-files` | `200` | 单次批量文件数上限（覆盖 100 套样本场景） |
| `--max-upload-mb` | `64` | 请求体上限 |
| `--workers` / `--max-queue` | `1` / `32` | 推理工作线程数与排队上限（GPU 场景建议 1） |
| `--time-budget` | `5.0` | 流水线内部时间预算（秒），超预算的 L2 通道自动跳过 |
| `--no-path-input` | 关 | 禁用 `application/json` 的 `path`/`dir` 服务端路径输入 |
| `--report-dir` | `<reports>/api` | API 落盘报告目录 |
| `--log` | `<reports>/api/requests.jsonl` | 请求日志（JSONL） |

### 服务化三种形态的关系

| 形态 | 入口 | 用途 |
| --- | --- | --- |
| 版本化 REST（主交付） | `api/server.py` | 对外接口、OpenAPI、批处理、报告回取、静态托管 WebUI |
| 单样本 CLI | `scripts/predict.py` | 命令行单图/单视频检测 |
| 批量评测 CLI | `scripts/batch_evaluate.py` | 100 套样本一键评测（离线指标 + 失败案例 + 逐样本报告） |

`GET /` 从 `demo/` 目录静态托管 WebUI 页面（`demo/webui.html`）；页面缺失时返回内置的简易回退页
（仍可直接上传文件并调用 `/api/v1/detect/image`），因此 WebUI 与接口互不阻塞。

---

## 2. 端点总览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | WebUI 页面（`demo/webui.html`，缺失则回退页） |
| GET | `/api/v1/health` | 各证据通道加载状态、预热状态、队列状态 |
| GET | `/api/v1/version` | 模型 / 阈值 / 代码版本与 sha256 |
| POST | `/api/v1/detect/image` | 图片检测 |
| POST | `/api/v1/detect/video` | 视频检测（抽帧聚合） |
| POST | `/api/v1/detect/batch` | 多文件批量检测（逐条摘要 + 汇总） |
| GET | `/api/v1/report/{report_id}` | 回取已落盘报告 |
| GET | `/api/v1/reports` | 已落盘报告列表 |
| GET | `/api/v1/metrics` | 返回 `reports/metrics.json` |
| GET | `/api/v1/openapi.json` | OpenAPI 3.0.3 描述 |
| GET | `/reports/{file}` | 历史报告静态访问（`reports/` 目录） |
| GET | `/demo/{file}` | WebUI 静态资源（`demo/` 目录） |

兼容别名（便于既有脚本/页面平滑迁移）：`GET /api/health` = `/api/v1/health`，
`POST /api/detect` = `/api/v1/detect/image`，`POST /api/detect_video` = `/api/v1/detect/video`。

### 三种上传方式（三个 detect 端点通用）

1. **`multipart/form-data`**：字段名任意（建议 `file`），取带 `filename` 的部件；批量端点取全部文件部件。
2. **原始字节**：`Content-Type: application/octet-stream`（或 `image/*` / `video/*`）+ `?name=xxx.png`
   （或请求头 `X-Filename`）用于确定媒体类型与后缀。
3. **`application/json`**：`{"path": "..."}`（单文件）或 `{"paths": [...]}` / `{"dir": "...", "recursive": true}`（批量）。
   该模式读取服务端路径，**仅在本机监听时默认启用**（`--no-path-input` 可关闭），且必须落在允许的根目录内。

公共查询参数：

| 参数 | 适用 | 说明 |
| --- | --- | --- |
| `timeout_ms` | detect/* | 单请求超时（毫秒，`1..--max-timeout-ms`），超时降级见 §5 |
| `save` | detect/* | 是否落盘报告，默认 `true` |
| `html` | detect/* | 是否额外生成单文件 HTML 报告，默认 `false` |
| `recursive` | detect/batch | `dir` 模式是否递归子目录 |

---

## 3. 端点详解与 curl 示例

### 3.1 `GET /api/v1/health`

返回 `status`（`ok` / `degraded`）、`mode`（`full` / `stub`）、`device`、`warmup`、`queue`、
`limits`，以及**逐通道加载状态** `channels`。真实运行示例（截取自本机 full 模式实例）：

```json
{
 "status": "ok",
 "service": "open-sz-01-api",
 "service_version": "1.0.0",
 "api_version": "v1",
 "mode": "full",
 "started_at": "2026-10-11T03:03:51Z",
 "uptime_s": 12.0,
 "device": "cuda",
 "models_dir": "C:\\Files\\Deepseek Harness\\Zhongan-OPEN-SZ-01\\models",
 "data_dir": "C:\\Files\\Deepseek Harness\\Zhongan-OPEN-SZ-01\\repo\\data",
 "reports_dir": "C:\\Files\\Deepseek Harness\\Zhongan-OPEN-SZ-01\\repo\\reports",
 "warmup": {
  "done": true, "ms": 11901.0,
  "sample": "...\\data\\images\\001.png",
  "started_at": "2026-10-11T03:03:51Z", "finished_at": "2026-10-11T03:04:03Z", "error": null
 },
 "queue": {"workers": 1, "max_queue": 32, "queued": 0, "submitted": 0, "completed": 0,
           "errors": 0, "timed_out": 0, "timeout_discarded": 0, "rejected_busy": 0},
 "requests_served": 1,
 "limits": {"timeout_ms": 30000, "max_timeout_ms": 300000, "max_upload_mb": 64.0,
            "max_batch_files": 200, "allow_path_input": true, "time_budget_s": 5.0},
 "channels": {
  "L0_input":        {"kind": "容器/解码通道（stdlib+cv2）", "loaded": true},
  "L1_forensics":    {"kind": "零权重取证特征（47维）", "loaded": true},
  "L1b_domain":      {"kind": "零权重域取证（相机 vs 渲染）", "loaded": true},
  "L3_localization": {"kind": "零权重异常图（ELA+噪声不一致）", "loaded": true},
  "L2_clip":         {"configured": true, "available": true, "loaded": true,
                      "path": "...\\models\\hf_cache\\models--openai--clip-vit-large-patch14\\snapshots\\32bd64..."},
  "L2_clip_probe":   {"configured": true, "available": true, "loaded": true,
                      "path": "...\\models\\clip_probe_norm.pt"},
  "L2_npr":          {"configured": false, "available": false, "loaded": false},
  "L2_aeroblade":    {"configured": false, "available": false, "loaded": false},
  "head_forensic":   {"configured": true, "available": true, "loaded": true},
  "head_combined":   {"configured": true, "available": true, "loaded": true},
  "fusion":          {"loaded": true, "weights": {"combined": 0.6, "forensic": 0.25},
                      "thr_low": 0.03644, "thr_high": 0.674288,
                      "temperature": {"combined": 1.71, "forensic": 1.1332}}
 }
}
```

`loaded` 表示该通道当前是否已在内存中常驻；`configured`/`available` 表示是否配置/磁盘上存在权重。
health 采用**只读探测**（不会因访问而触发一次惰性加载），因此刚启动、未预热时
`loaded=false, available=true` 是正常状态。

```bash
curl.exe -s http://127.0.0.1:8080/api/v1/health
# PowerShell 等价：Invoke-RestMethod http://127.0.0.1:8080/api/v1/health | ConvertTo-Json -Depth 5
```

### 3.2 `GET /api/v1/version`

模型 / 阈值 / 代码的版本与**内容哈希**（可追溯、可比对、可写进报告）。

```json
{
 "service": "open-sz-01-api", "service_version": "1.0.0",
 "api_version": "v1", "report_schema_version": "1.0.0",
 "pipeline": {"version": "open-sz-01 v0.3", "mode": "full", "device": "cuda",
              "time_budget_s": 5.0, "stub_reason": null},
 "code": {"root": "...\\repo", "n_source_files": 39,
          "code_hash": "dd2c12d2660655cf10cf0a6ca28241aeaebf426d99bb5f635d3fb574a416e2e8",
          "git_commit": "ba1db77071b9a0e8944339215c47d4af5822ca22",
          "git_branch": "main", "git_dirty": true},
 "models": {
   "clip_probe": {"path": "...\\models\\clip_probe_norm.pt", "exists": true, "bytes": 24677,
                  "sha256": "d040ca02207f478d547c79cae27c198e430779c0e71fa8dc7a80315b717adbde"},
   "forensic_lr": {"path": "...\\models\\forensic_lr.joblib", "exists": true, "bytes": 3395,
                   "sha256": "21ac8ed153a68f752e34624b67cfc08f9eed1ebc8b54eaa51b02b13ac02b79a8"},
   "fusion_json": {"path": "...\\models\\fusion.json", "exists": true, "bytes": 246,
                   "sha256": "d7062fe5198eb38c58211d76ef178b6a65196c055b7f426277f249586190a011"}
 },
 "thresholds": {"low": 0.036440409414102856, "high": 0.6742875975794557,
                "disagree_review": 1.25, "weights": {"combined": 0.6, "forensic": 0.25},
                "temperatures": {"combined": 1.7099609375000016, "forensic": 1.133203125},
                "source": "...\\models\\fusion.json",
                "source_sha256": "d7062fe5198eb38c58211d76ef178b6a65196c055b7f426277f249586190a011"},
 "metrics": {"path": "...\\repo\\reports\\metrics.json", "exists": true,
             "sha256": "398dfa4fd545b875abccbfeb551d46ca92c7ef934549d8a76bcef870652f8019",
             "n_samples": 119, "oof_accuracy": 0.958, "oof_fpr": 0.0317, "oof_tpr": 0.9464},
 "runtime": {"python": "3.14.7", "platform": "win32", "executable": "C:\\Python314\\python.exe"},
 "generated_at": "2026-10-11T03:04:05Z"
}
```

```bash
curl.exe -s http://127.0.0.1:8080/api/v1/version
```

### 3.3 `POST /api/v1/detect/image`

| 输入 | 示例 |
| --- | --- |
| multipart | `curl.exe -s -X POST -F "file=@data/images/001.png" http://127.0.0.1:8080/api/v1/detect/image` |
| 原始字节 | `curl.exe -s -X POST --data-binary "@data/images/001.png" "http://127.0.0.1:8080/api/v1/detect/image?name=001.png"` |
| JSON 路径 | `curl.exe -s -X POST -H "Content-Type: application/json" -d "{\"path\":\"data/images/001.png\"}" http://127.0.0.1:8080/api/v1/detect/image` |
| 带超时/落盘控制 | `... ?timeout_ms=20000&save=true&html=false` |

响应 = **完整检测报告 JSON**（`input_hash` / `resolution` / `verdict` / `confidence` /
`model_scores` / `forensic_clues` / `localization` / `explanation` / `uncertainty` /
`versions` / `latency_ms`）+ `meta` 元信息（`report_id` / `report_url` / `elapsed_ms` /
`timeout_ms` / `degraded` / `input`）。真实响应（节选，`localization.heatmap_png_b64` 与
`explanation.heatmap_png_b64` 为 base64 PNG 已省略）：

```json
{
 "input_hash": "f40204f5571717278b8c50b481458caa73cfd9f46fc7ac6db30e86e95747b889",
 "input_type": "image",
 "resolution": "1238x1704",
 "verdict": "suspect",
 "verdict_cn": "疑似生成/篡改",
 "confidence": 0.9657,
 "fused_score": 0.9657,
 "model_scores": {"forensic": 0.8659, "combined": 0.9812},
 "disagreement": 1.0439,
 "forensic_clues": [
  {"name": "PNG容器指纹", "status": "命中", "detail": "IDAT块数=..., sRGB=..., eXIf=... "},
  {"name": "EXIF/XMP元数据", "status": "未命中", "detail": "..."}
 ],
 "localization": {"available": true, "kind": "zero-weight forensic map (ELA+noise inconsistency)",
                  "heatmap_png_b64": "<base64 PNG，此处省略>", "boxes": []},
 "explanation": {"available": true, "method": "tile_contribution_occlusion (leave-one-tile-out on combined head)",
                 "tiles": [{"index": 1, "row": 0, "col": 1, "bbox_xywh": [348, 0, 465, 640],
                            "delta_logit": 1.2948, "weight": 1.0}],
                 "gradcam": {"available": false, "status": "不可用",
                             "reason": "combined 头为 joblib/sklearn 逻辑回归包：线性模型，无卷积特征图",
                             "substitute": "tile_contribution_occlusion"}},
 "uncertainty": ["Grad-CAM 不可用（…）；关注区域解释改由块级贡献归因提供",
                 "fused score 0.966 >= high threshold 0.674"],
 "versions": {"pipeline": "open-sz-01 v0.1", "models": "forensic+combined",
              "thresholds": "low=0.036/high=0.674",
              "thresholds_sha256": "sha256:e9db1a93ac66363dd7e024883dc72f62be23643cea05c9756be3caa6c3c21450",
              "device": "cuda",
              "input_sha256": "f40204f5571717278b8c50b481458caa73cfd9f46fc7ac6db30e86e95747b889",
              "artifact_sha256": {"clip_probe_norm.pt": "sha256:d040ca02…", "forensic_lr.joblib": "sha256:21ac8ed1…"}},
 "latency_ms": {"L0": 20.2, "L1": 77.9, "L2_clip": 39.3, "L2_clip_tiles": 191.3,
                "explain": 215.9, "L3": 23.7, "fusion": 0.2, "total": 967.8, "server_total": 971.6},
 "meta": {"request_id": "6873d4cd5547", "api_version": "v1", "elapsed_ms": 971.6,
          "timeout_ms": 30000, "degraded": false,
          "report_id": "20261011T110405-476d072a",
          "report_url": "/api/v1/report/20261011T110405-476d072a",
          "saved_path": "...\\repo\\reports\\api\\20261011T110405-476d072a.json",
          "input": {"filename": "001.png", "kind": "image"}}
}
```

### 3.4 `POST /api/v1/detect/video`

未提供视频文件时按单请求降级为 `review`（并给出原因），不会 500。真实响应（节选）：

```json
{
 "input_hash": "7cc6a65ddbd756e67e00bd3d7887e917db8d1fddd7b8801c7c04bf52c8ed4614",
 "input_type": "video", "resolution": "640x640", "duration_s": 4.0,
 "verdict": "review", "verdict_cn": "待复核", "confidence": 0.1481, "fused_score": 0.1481,
 "model_scores": {"frame_mean": 0.1025, "frame_median": 0.0903,
                  "frame_p90": 0.1937, "frame_std": 0.0869},
 "localization": {"available": true, "kind": "temporal segments",
                  "suspicious_segments": [], "heatmap_png_b64": null,
                  "note": "视频通道复用图像模型（批量全局CLIP+取证）…该通道不提供空间热力图"},
 "latency_ms": {"L2_clip_batch": 229.9, "total": 3191.7, "server_total": 4140.0},
 "meta": {"elapsed_ms": 4140.0, "timeout_ms": 30000, "degraded": false,
          "report_id": "20261011T110410-fe133882",
          "input": {"filename": "smoke_test.mp4", "kind": "video"}}
}
```

```bash
curl.exe -s -X POST --data-binary "@demo_assets/smoke_test.mp4" \
  "http://127.0.0.1:8080/api/v1/detect/video?name=smoke_test.mp4"
```

### 3.5 `POST /api/v1/detect/batch`

输入同 §3 三种方式（multipart 全部文件部件，或 `{"paths": [...]}` / `{"dir": "..."}`）。
返回 `count` + `summary`（结论分布、失败/超时计数、置信度统计、耗时、落盘数）+ `results`
（逐条摘要：文件名、媒体类型、`status`（`ok`/`timeout`/`error`/`skipped_budget`）、结论、
置信度、`report_id`/`report_url`、首因 `attention`、逐条耗时）。真实响应：

```json
{
 "request_id": "b792fd13c19a", "api_version": "v1", "count": 2,
 "summary": {"n": 2, "verdicts": {"normal": 0, "review": 0, "suspect": 2},
             "ok": 2, "timeout": 0, "errors": 0, "skipped_budget": 0,
             "mean_confidence": 0.9698, "max_confidence": 0.9738, "min_confidence": 0.9657,
             "elapsed_ms": 1854.0, "reports_saved": 2, "source": "uploads", "mode": "full"},
 "results": [
  {"filename": "001.png", "kind": "image", "status": "ok", "verdict": "suspect",
   "verdict_cn": "疑似生成/篡改", "confidence": 0.9657, "fused_score": 0.9657,
   "disagreement": 1.0439, "degraded": false,
   "report_id": "20261011T110411-40f9b91a", "report_url": "/api/v1/report/20261011T110411-40f9b91a",
   "attention": "Grad-CAM 不可用（…）", "latency_ms": {"total": 974.8, "server_total": 978.1},
   "error": null},
  {"filename": "002.png", "kind": "image", "status": "ok", "verdict": "suspect",
   "confidence": 0.9738, "report_id": "20261011T110412-91b3e626", "error": null}
 ],
 "meta": {"timeout_ms": 30000, "batch_timeout_ms": 600000, "saved": true, "degraded": false}
}
```

```bash
# 多文件 multipart
curl.exe -s -X POST -F "file=@data/images/001.png" -F "file=@data/images/002.png" \
  http://127.0.0.1:8080/api/v1/detect/batch
# 服务端目录（本机部署）
curl.exe -s -X POST -H "Content-Type: application/json" \
  -d "{\"dir\":\"data/images\",\"recursive\":false}" http://127.0.0.1:8080/api/v1/detect/batch
```

### 3.6 `GET /api/v1/report/{report_id}`

按 `report_id` 回取已落盘报告；先查 `<reports>/api/<id>.json`，再查 `<reports>/<id>.json`
（历史报告如 `001.json` 亦可回取），最后回退到同名 `.html`（返回 `text/html`）。

```bash
curl.exe -s -o report.json http://127.0.0.1:8080/api/v1/report/20261011T110405-476d072a
```

`GET /api/v1/reports?limit=50` 列出已落盘报告（`report_id` / `bytes` / `modified` / `url`）。

### 3.7 `GET /api/v1/metrics`

直接返回 `<reports>/metrics.json`（离线 OOF 指标、分折结果、ROC 前沿、延迟、错误分析）。
文件不存在时返回 `404 E_METRICS_MISSING`，并在 `details.expected_path` 给出期望路径与生成命令。

```bash
curl.exe -s http://127.0.0.1:8080/api/v1/metrics
```

### 3.8 `GET /api/v1/openapi.json`

OpenAPI 3.0.3 文档，覆盖上述全部端点（含 `requestBody` 的 multipart / octet-stream /
JSON 三种媒体类型、错误响应、以及 `ErrorEnvelope`/`Health`/`Version`/`DetectionResponse`/
`BatchResponse` 等 schema）。可直接导入 Swagger UI / Postman。

```bash
curl.exe -s -o openapi.json http://127.0.0.1:8080/api/v1/openapi.json
```

---

## 4. 统一错误体与错误码表

所有错误响应（HTTP 状态码 ≠ 2xx）都是同一个结构：

```json
{
 "error": {
  "code": "E_NOT_FOUND",
  "message": "报告不存在或未落盘：does-not-exist",
  "details": {"searched": ["...\\reports\\api\\does-not-exist.json", "..."]},
  "request_id": "35409ea856f0"
 }
}
```

- `code` 稳定，供程序判断；`message` 面向人；`details` 可选；`request_id` 与响应头
  `X-Request-Id` 一致，便于与请求日志对齐。
- 错误码表（与 `api/server.py:ERROR_CODES` 保持一致；`GET /api/v1/openapi.json` 的
  `components.schemas.Error.properties.code.enum` 也列出了同一组值）：

| code | HTTP | 含义 | 典型触发 |
| --- | --- | --- | --- |
| `E_BAD_REQUEST` | 400 | 请求格式/参数非法 | `timeout_ms` 非数字、JSON 解析失败、`limit` 非法、报告 id 含非法字符 |
| `E_NO_FILE` | 400 | 未提供文件 | 空请求体、multipart 中没有带 `filename` 的部件 |
| `E_UNSUPPORTED_MEDIA_TYPE` | 415 | 不支持的 `Content-Type` | 如 `application/x-msgpack` |
| `E_FORBIDDEN_PATH` | 403 | 服务端路径越界或已禁用 | `path`/`dir` 在允许根之外；非本机监听且未显式允许 |
| `E_NOT_FOUND` | 404 | 资源不存在 | 未知端点、报告 id 不存在、路径/目录不存在 |
| `E_METHOD_NOT_ALLOWED` | 405 | 方法不匹配 | 对 `detect/*` 发 GET |
| `E_PAYLOAD_TOO_LARGE` | 413 | 请求体过大 | 超过 `--max-upload-mb`；批量文件数超过 `--max-batch-files` |
| `E_BUSY` | 429 | 推理队列已满 | 并发超过 `--max-queue` |
| `E_METRICS_MISSING` | 404 | `metrics.json` 不存在 | 尚未运行 `scripts/evaluate_system.py` |
| `E_MODEL_UNAVAILABLE` | 503 | 模型/通道不可用 | 预留（当前一律降级为 `review` 报告而非 503） |
| `E_INTERNAL` | 500 | 服务内部错误 | 未预期异常（同时写入日志的 `traceback`） |

> 设计约定：**输入不可解码、通道加载失败、推理超时都不是 HTTP 错误**，而是 HTTP 200 +
> `verdict="review"` + `uncertainty` 说明（手册 4.2/16.3 的降级要求）。HTTP 错误只用于
> "请求本身不合法"。

---

## 5. 单请求超时降级（含示例）

- 单请求超时由 `?timeout_ms=`（或 `--timeout-ms`）控制，范围 `1..--max-timeout-ms`。
- 超时后立即返回 HTTP 200 + `verdict="review"` + `confidence=0`，`meta.degraded=true`，
  `uncertainty` 明确写出"单请求超时降级…请转人工复核"。
- 队列中的那一单会继续跑完（GPU 状态不被破坏），其结果记为 `timeout_discarded` 并写入日志，
  不回传给已超时的调用方。
- `GET /api/v1/health` 的 `queue.timed_out` / `queue.timeout_discarded` 可用于运维观测。

真实响应（`?timeout_ms=1`，人为触发降级）：

```json
{
 "input_hash": "f40204f5571717278b8c50b481458caa73cfd9f46fc7ac6db30e86e95747b889",
 "verdict": "review", "verdict_cn": "待复核", "confidence": 0.0, "fused_score": 0.0,
 "model_scores": {}, "forensic_clues": [], "localization": {}, "explanation": {},
 "uncertainty": ["单请求超时降级：推理超过 0.0s 未返回（队列继续完成该单），按要求返回 review 结论，请转人工复核"],
 "versions": {"pipeline": "open-sz-01 v0.3", "channel": "image-degraded", "mode": "full"},
 "latency_ms": {"server_total": 13.9},
 "meta": {"request_id": "291f8368414f", "api_version": "v1", "elapsed_ms": 13.9,
          "timeout_ms": 1, "degraded": true, "report_id": null, "report_url": null,
          "input": {"filename": "upload.png", "kind": "image"}}
}
```

```bash
curl.exe -s -X POST --data-binary "@data/images/001.png" \
  "http://127.0.0.1:8080/api/v1/detect/image?timeout_ms=1&save=0"
```

---

## 6. 模型常驻、预热与请求日志

**模型常驻 + 启动预热**（手册 16.2）：进程启动时构建检测器（一次）并跑一次完整推理作为预热，
权重与 cuDNN 调优成本在启动阶段付清。实测：

| 阶段 | 实测（RTX 5070 Laptop / CUDA） | 说明 |
| --- | --- | --- |
| 预热耗时 | 11.9 ~ 12.0 s | 冷启动含权重加载 + 首次推理 |
| 预热后单张图片 | 总耗时 967.8 ms，`server_total` 971.6 ms（p50 ≈ 1.0 s） | 与 `reports/metrics.json` 的 `latency_image.p50=1007.9ms`、`smoke_test.json` 的 warm 1.23/1.64 s 一致 |
| 视频（4 s / 10 帧） | 3.19 s 推理（`server_total` 4.14 s） | 5 s 预算内 |
| 预热状态 | `GET /api/v1/health` → `warmup.done=true, ms=…` | 未预热时 `warmup.skipped=true` |

**请求日志**：每个请求写一行 JSON（同时输出到 stderr），落盘到 `--log`（默认
`<reports>/api/requests.jsonl`）：

```json
{"ts": "2026-10-11T03:03:31Z", "level": "info", "request_id": "001a9bc27b93",
 "remote": "127.0.0.1", "method": "GET", "path": "/api/v1/health", "status": 200,
 "bytes": 3451, "ms": 0.7, "note": "health", "mode": "full"}
```

stderr 形态（便于 `docker logs` 直接观察）：

```
[api] POST /api/v1/detect/batch -> 200 (1977B, 1869.1ms) req=b792fd13c19a batch:2
[api] GET /api/v1/report/does-not-exist -> 404 (508B, 0.6ms) req=35409ea856f0 error:E_NOT_FOUND
```

---

## 7. 批处理脚本 `scripts/batch_evaluate.py`（100 套样本一键评测）

```bash
# 命题数据集：119 套全部评测
python scripts/batch_evaluate.py --dir data/images --labels data/labels.csv \
    --out reports/batch/full119

# 子集冒烟（前 24 套）
python scripts/batch_evaluate.py --dir data/images --labels data/labels.csv --limit 24 \
    --out reports/batch/verify-24

# 按目录名取标签（images/ai/*, images/real/*）
python scripts/batch_evaluate.py --dir data/images --infer-labels --out reports/batch/bydir

# 视频批量（抽帧聚合）
python scripts/batch_evaluate.py --dir demo_assets --ext mp4 --video

# 无模型环境验证链路（结论固定 review，用于自测）
python scripts/batch_evaluate.py --selftest
python scripts/batch_evaluate.py --dir data/images --no-model --limit 4
```

标签来源优先级：`--labels`（`seq,filename,label` 或 `filename,label`，自动识别表头，支持
`ai/fake/generated/1/是` 与 `real/genuine/0/否` 同义词）→ `--label-dirs a=…,b=…` →
`--infer-labels`（按目录名/文件名推断）。三者都没有时**只输出结论分布与逐样本报告**，
不计算监督指标（避免无标签时给出误导性的 Accuracy）。

产物（默认 `reports/batch/<时间戳>/`）：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 全部指标 + 协议说明 + 逐通道均分 + 延迟分位 + 与 `reports/metrics.json` 的离线参考对照 |
| `metrics.md` | 同上的 Markdown 版（可直接贴进报告） |
| `failures.csv` | 失败案例清单（`false_positive` / `false_negative` / `review` / `decode_error` / `error` + 首因） |
| `per_sample.json` | 逐样本明细（结论、置信度、模型分、逐段耗时、标签、失败类型） |
| `samples/<name>.json` | **逐样本完整报告**（与 API 落盘的报告同 schema；`--html` 时对 FP/FN 追加 HTML） |

指标口径（在 `metrics.json.protocol` 中明示）：

- **主口径**（与 `reports/metrics.json` / `scripts/evaluate_system.py` 一致）：`review` 计入
  负类（real），正类为 `ai`；`Accuracy = (TP+TN)/N`、`FPR = FP/FN+TN 的 real 数`、`TPR = TP/AI 数`。
- 另附：`decided_only`（排除弃权样本的严格口径）、`coverage`（非弃权占比）、
  `accuracy_conservative_review_as_error`（**保守下界**：弃权按最不利方向计）、`auc_confidence`
  （连续置信度 AUC）、三档分布，以及 Accuracy/FPR 的 bootstrap 95% CI（默认 2000 次重采样，seed 固定）。
- 真实运行示例 A（**全部 119 套，一键评测**，CUDA；CUDA 预热 12.4 s，总耗时 137.4 s）：

```
批量评测：119 样本（带标签 119）  耗时 137.44 s
Accuracy (review→real)     1.0000      Accuracy 95% CI [1.0, 1.0]
FPR (误报率)                 0.0000      FPR 95% CI [0.0, 0.0]
TPR / Recall               1.0000      Precision  1.0000
AUC (连续置信度)             1.0000
覆盖率 (非弃权)               0.7899      （94/119）
保守下界 (弃权=错误)           0.7899
混淆矩阵  TP=56  FP=0  FN=0  TN=63      (AI=56 / real=63)
已决样本  Acc=1.0000 FPR=0.0000 TPR=1.0000 (n=94)
三档分布  suspect=56  normal=38  review=25   成因={'model_disagreement': 22, 'uncertainty_band': 3}
耗时      p50=980.7 ms  p95=1482.4 ms  max=1719.6 ms  mean_wall=1027.5 ms
失败案例  {'review': 25}（共 25 条，均来自 real 侧弃权，详见 failures.csv）
```

- 真实运行示例 B（24 套均衡子集：12 张 AI + 12 张 real，用 `--list` 指定）：

```
批量评测：24 样本（带标签 24）  耗时 44.03 s
Accuracy (review→real) 1.0000   FPR 0.0000   TPR 1.0000   AUC 1.0000
覆盖率 (非弃权) 0.7083（17/24）  保守下界 0.7083
混淆矩阵  TP=12  FP=0  FN=0  TN=12      成因={'model_disagreement': 7}
三档分布  suspect=12  normal=5  review=7
```

产物：`reports/batch/verify-119/`（119 套）、`reports/batch/verify-balanced24/`（均衡子集），
含 `metrics.json` / `metrics.md` / `failures.csv` / `per_sample.json` / `samples/*.json`（逐样本 119 份）。

> 口径差异说明：本脚本用**模型侧冻结阈值**（`models/fusion.json` 的 `low=0.0364 / high=0.6743`）
> 直接在全部样本上推理，而 `reports/metrics.json` 的 `accuracy=0.958 / fpr=0.0317 / tpr=0.9464`
> 是**分层 5 折 OOF** 口径（每折阈值只在训练折上按目标 FPR≤3% 选出）。两者不可直接互比；
> `metrics.json.supervised.reference_offline_oof` 会同时列出离线参考值，便于对照。
> 弃权（review）在本脚本中计入 real（与离线口径一致），并另行给出覆盖率与"弃权=错误"的保守下界。
>
> 注意：只给单类样本（例如 `--limit` 截取到前 N 张全是 AI）时，FPR/特异性或 TPR/AUC
> 的分母为 0，脚本会把它们标为 **不可评估**（`null` + `note`），不会伪造成 0 或 nan；
> 需要完整指标时请同时包含 `ai` 与 `real` 两类样本（如使用 `--list` 做均衡抽样）。

---

## 8. 端点自测

`tests/test_api.py`（纯标准库，无 pytest 依赖）：

```bash
python tests/test_api.py            # 先跑 stub 契约套件，再跑 full 真实模型套件
python tests/test_api.py --fast     # 只跑 stub 套件（秒级）
python tests/test_api.py --no-stub  # 只跑 full 套件（含预热/真实推理/超时降级）
python tests/test_api.py --base-url http://127.0.0.1:8080   # 对已运行实例测试
python api/server.py --selftest --no-model                  # 进程内自测
```

覆盖 65 项检查：health/version/openapi/metrics/reports/report 回取/静态托管、三种输入方式、
视频、批量（multipart 与 JSON 路径）、统一错误体与错误码、报告 id 路径穿越防护、静态资源穿越
防护、越界路径拒绝、3 路并发、单请求超时降级、multipart 解析器二进制安全单测。
实测结果：

| 套件 | 结果 |
| --- | --- |
| stub（`--no-model`，无权重环境） | **57 passed / 0 failed** |
| full（真实模型 + 预热 10.1 s） | **65 passed / 0 failed** |

---

## 9. 安全与部署边界

- 默认**只监听 `127.0.0.1`**；若改为非本机地址，服务端路径输入（`path`/`dir`）默认关闭，
  除非显式加 `--allow-path-input`。
- 服务端路径读取限制在允许的根目录内（仓库目录、其上一级、`data/` 与临时目录，可用
  `--allow-root` 追加）；`report_id` 仅接受 `[A-Za-z0-9._-]{1,64}`，`/reports/` 与 `/demo/`
  静态目录做了 `resolve()` 后的包含性校验，防目录穿越。
- 单请求体默认上限 64 MB，批量默认上限 200 文件。
- **本服务不含身份认证**：定位是内网/本机演示与评测服务。对外部署时须前置网关/反向代理
  做认证与限流，或仅暴露 OpenAPI 描述中需要的端点。
- 可选实现：本环境实测 `fastapi 0.143 / uvicorn 0.54 / gradio 6.30` 均可导入，但为满足
  "零额外依赖、可直接运行"的要求，**stdlib 版本是主交付**；OpenAPI 描述已按要求输出
  （`/api/v1/openapi.json`），如需 FastAPI 版本可基于该 schema 做薄封装。
