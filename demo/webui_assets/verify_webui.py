#!/usr/bin/env python3
"""WebUI 验收自检（stdlib，零依赖）。

对**正在运行**的 OPEN-SZ-01 服务做端到端检查，覆盖 m4-webui 交付要求的每一项：

  1. GET /                               首页由 demo/webui.html 托管（HTTP 200 + text/html）
                                         且包含 5 个板块锚点与关键渲染入口
  2. GET /demo/webui_assets/…            静态资产可达：oof_119.json（119 条折外记录）、缩略图、README
  3. GET /api/v1/health|version|metrics  页面依赖的数据端点可达
  4. GET /reports/batch/…/samples/NNN.json  样本库「点击查看完整报告」的真实数据源可达
  5. POST /api/v1/detect/image           实时检测（原始字节模式）返回 200 且含结论/分数/定位图
  6. POST /api/v1/detect/image?timeout_ms=1  超时降级为 review（HTTP 200 + degraded）

用法：
    python api/server.py --port 8080          # 另开一个终端
    python demo/webui_assets/verify_webui.py  # 默认 http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[2]

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  — " + detail) if detail else ""))


def get(base: str, path: str, timeout: float = 30.0):
    req = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)
    except Exception as e:  # noqa: BLE001
        return 0, str(e).encode(), {}


def post_bytes(base: str, path: str, data: bytes, ctype: str, timeout: float = 120.0):
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)
    except Exception as e:  # noqa: BLE001
        return 0, str(e).encode(), {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--sample", default=str(WS / "data" / "images" / "001.png"))
    ap.add_argument("--wait-warmup", type=float, default=60.0,
                    help="等待启动预热完成的最长秒数")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # ---- 0. 等待服务与预热就绪 -------------------------------------------------
    t0 = time.time()
    warmup = None
    while time.time() - t0 < args.wait_warmup:
        st, body, _ = get(base, "/api/v1/health", timeout=10)
        if st == 200:
            warmup = json.loads(body)
            if (warmup.get("warmup") or {}).get("done"):
                break
        time.sleep(2)
    check("服务可达且 GET /api/v1/health = 200", warmup is not None)
    if warmup is None:
        return 1
    mode = warmup.get("mode")
    print("      mode=%s device=%s warmup_done=%s ms=%s"
          % (mode, warmup.get("device"), (warmup.get("warmup") or {}).get("done"),
             (warmup.get("warmup") or {}).get("ms")))

    # ---- 1. 首页 -----------------------------------------------------------------
    st, body, hdr = get(base, "/")
    html = body.decode("utf-8", "replace")
    check("GET / = 200", st == 200, "status=%s" % st)
    check("GET / 由 text/html 返回", "text/html" in (hdr.get("Content-Type") or ""),
          hdr.get("Content-Type", ""))
    check("首页不是内置回退页（含 webui 标记）",
          "功能展示台" in html and "未找到 demo/webui.html" not in html)
    anchors = ["matrix", "live", "samples", "dashboard", "compliance"]
    missing = [a for a in anchors if ('id="%s"' % a) not in html]
    check("首页包含全部板块锚点 %s" % anchors, not missing, "缺失=%s" % missing)
    titles = ["功能矩阵", "实时检测", "样本库", "指标看板", "合规"]
    check("首页包含各板块标题", all(t in html for t in titles))
    check("首页无外部 CDN 依赖",
          not [u for u in re.findall(r'https?://[^"\'\s<>]+', html)
               if "127.0.0.1" not in u and "localhost" not in u])
    check("页面字节数 < 200 KB（单文件）", len(body) < 200 * 1024, "%d bytes" % len(body))

    # ---- 2. 静态资产 -------------------------------------------------------------
    st, body, hdr = get(base, "/demo/webui_assets/oof_119.json")
    oof = json.loads(body) if st == 200 else {}
    n = len(oof.get("samples") or [])
    check("GET /demo/webui_assets/oof_119.json = 200 且含 119 条样本", st == 200 and n == 119,
          "status=%s n=%s" % (st, n))
    m = oof.get("deployed_metrics") or {}
    check("资产内的折外指标与 reports/metrics.json 一致（混淆 TP53 FP2 FN3 TN61）",
          m.get("confusion") == {"tp": 53, "tn": 61, "fp": 2, "fn": 3} and m.get("accuracy") == 0.958,
          json.dumps(m, ensure_ascii=False))
    st, body, hdr = get(base, "/demo/webui_assets/thumbs/001.svg")
    check("GET 缩略图 001.svg = 200 且为 image/svg+xml",
          st == 200 and "image/svg" in (hdr.get("Content-Type") or ""),
          "status=%s type=%s" % (st, hdr.get("Content-Type")))
    check("缩略图是内嵌 JPEG 的 SVG", b"data:image/jpeg;base64," in body)
    missing_thumbs = [s for s in (1, 57, 119) if get(base, "/demo/webui_assets/thumbs/%03d.svg" % s)[0] != 200]
    check("抽查缩略图 001/057/119 全部可访问", not missing_thumbs, "缺失=%s" % missing_thumbs)
    st, _, _ = get(base, "/demo/webui_assets/README.md")
    check("GET 静态资产说明 README.md = 200", st == 200, "status=%s" % st)

    # ---- 3. 页面依赖的数据端点 ---------------------------------------------------
    for path, key in (("/api/v1/version", "thresholds"), ("/api/v1/metrics", "oof_overall")):
        st, body, _ = get(base, path)
        ok = st == 200
        if ok:
            d = json.loads(body)
            ok = key in d
        check("GET %s = 200 且含字段 %s" % (path, key), ok, "status=%s" % st)
    check("health 报告回取链与报告 schema 版本（report_search_dirs / report_schema_version）",
          isinstance((warmup or {}).get("report_search_dirs"), list)
          and bool((warmup or {}).get("report_schema_version")),
          "dirs=%s schema=%s" % ((warmup or {}).get("report_search_dirs"),
                                 (warmup or {}).get("report_schema_version")))

    # 页面首选「服务端报告回退链」，不依赖 batch 归档是否入库（与 api/server.py 解耦）
    page_src = (REPO / "demo" / "webui.html").read_text(encoding="utf-8")
    check("首页报告数据源首选 /api/v1/report/{id} 回退链，batch 归档仅作兜底",
          '"/api/v1/report/" + p' in page_src and "/reports/batch/verify-119/samples/" in page_src
          and page_src.index('"/api/v1/report/" + p') < page_src.index("/reports/batch/verify-119/samples/"))

    # ---- 4. 样本详情数据源（复刻页面 REPORT_SOURCES 的 6 级探测顺序） --------------
    REPORT_SOURCES = [
        "/api/v1/report/%s",
        "/reports/%s.json",
        "/reports/batch/verify-119/samples/%s.json",
        "/reports/batch/verify-24/samples/%s.json",
        "/reports/batch/verify-balanced24/samples/%s.json",
        "/reports/batch/full119/samples/%s.json",
    ]
    found, used = 0, []
    for seq in (1, 57, 119):
        pad = "%03d" % seq
        for tpl in REPORT_SOURCES:
            st, body, _ = get(base, tpl % pad)
            if st != 200:
                continue
            try:
                d = json.loads(body)
            except Exception:  # noqa: BLE001
                d = {}
            if d.get("verdict"):
                found += 1
                used.append("#%s←%s" % (pad, tpl))
                break
    check("样本完整报告 6 级回退链覆盖 #001/#057/#119", found == 3, "命中 %d/3：%s" % (found, used))
    if found:
        st, body, _ = get(base, REPORT_SOURCES[0] % "001")
        d = json.loads(body) if st == 200 else {}
        check("回取到的报告是新 schema（localization.available / explanation / artifact_sha256）",
              bool((d.get("localization") or {}).get("available") is not None
                   and d.get("explanation") and ((d.get("versions") or {}).get("artifact_sha256"))),
              "keys=%s" % (sorted((d.get("explanation") or {}).keys())[:6],))

    st, _, _ = get(base, "/api/v1/report/does-not-exist")
    check("报告回取对不存在 id 返回 404 + 统一错误体", st == 404, "status=%s" % st)

    # ---- 4b. 指标看板证据卡（含第二轮审计新增的三张） ----------------------------
    EVIDENCE = ["feature_ablation", "fusion_ablation", "npr_ablation", "robustness_matrix", "smoke_test",
                "calibration", "member_analysis", "threshold_stability", "stratified_analysis",
                "resource_profile", "cross_generator", "aeroblade_ablation", "stratified_manual"]
    ev_ok, ev_missing = [], []
    for name in EVIDENCE:
        st, body, _ = get(base, "/reports/%s.json" % name)
        (ev_ok if st == 200 else ev_missing).append(name)
    check("13 张证据卡的数据文件全部可达（页面按同路径读取）", not ev_missing,
          "已加载 %d/13 缺失=%s" % (len(ev_ok), ev_missing))

    st, body, _ = get(base, "/reports/cross_generator.json")
    cg = json.loads(body) if st == 200 else {}
    folds = cg.get("folds") or {}
    check("跨生成器证据含对照组 AUC 与 3 个整类留出 fold（含逐生成器 recall）",
          isinstance((cg.get("same_generator_control") or {}).get("auc"), float)
          and len([k for k, v in folds.items() if v.get("held_out_generators")]) == 3
          and all((v.get("per_generator_recall_at_threshold") for v in folds.values())),
          "control=%s folds=%s" % ((cg.get("same_generator_control") or {}).get("auc"),
                                   [v.get("held_out_generators") for v in folds.values()]))

    st, body, _ = get(base, "/reports/aeroblade_ablation.json")
    ab = json.loads(body) if st == 200 else {}
    con = ab.get("conclusions") or {}
    check("AEROBLADE 消融含单路 AUC / 最佳变体 / 加入融合 Δ / 最终判定",
          bool((con.get("primary_variant_auc") is not None) and (con.get("best_variant") or {}).get("name")
               and (con.get("best_add_in") or {}).get("delta_accuracy") is not None
               and "剔除" in str(con.get("verdict"))),
          "verdict=%s auc=%s best=%s" % (con.get("verdict"), con.get("primary_variant_auc"),
                                         (con.get("best_variant") or {}).get("name")))

    st, body, _ = get(base, "/reports/stratified_manual.json")
    sm = json.loads(body) if st == 200 else {}
    sct, strata = sm.get("shortcut_test") or {}, sm.get("strata") or {}
    con_s = (strata.get("content") or {})
    sc_verdict = str(sct.get("verdict") or "")
    check("分维度证据含捷径检验与外观/内容分维度指标（含 scene FPR=1.0 与 document FPR=0.0）",
          (sct.get("appearance_gap") is not None
           and ("无可用" in sc_verdict or "NO usable" in sc_verdict))
          and ((con_s.get("scene") or {}).get("fpr") == 1.0)
          and ((con_s.get("document") or {}).get("fpr") == 0.0),
          "gap=%s verdict=%s scene_fpr=%s doc_fpr=%s" % (sct.get("appearance_gap"), sc_verdict[:40],
                                                        (con_s.get("scene") or {}).get("fpr"),
                                                        (con_s.get("document") or {}).get("fpr")))
    page_txt = (REPO / "demo" / "webui.html").read_text(encoding="utf-8")
    check("首页含三张新证据卡标题、功能矩阵捷径检验与误报预算结论",
          all(t in page_txt for t in ("跨生成器泛化", "AEROBLADE 消融（可选成员，结论：剔除）",
                                      "分维度指标（人工标注）", "捷径检验", "误报预算应按")))


    # ---- 5. 实时检测（原始字节模式，页面实际使用的调用方式） ---------------------
    sample = Path(args.sample)
    if not sample.is_file():
        check("存在可用于实时检测的样本文件 %s" % sample, False)
    else:
        st, body, _ = post_bytes(base, "/api/v1/detect/image?name=%s&save=true&html=false" % sample.name,
                                 sample.read_bytes(), "application/octet-stream")
        rep = json.loads(body) if st == 200 else {}
        ok = (st == 200 and rep.get("verdict") in ("normal", "review", "suspect")
              and isinstance(rep.get("model_scores"), dict) and rep.get("model_scores"))
        check("POST /api/v1/detect/image（原始字节）= 200 且含结论与各路分数", ok,
              "status=%s verdict=%s confidence=%s scores=%s"
              % (st, rep.get("verdict"), rep.get("confidence"),
                 json.dumps(rep.get("model_scores"), ensure_ascii=False)))
        loc = rep.get("localization") or {}
        exp = rep.get("explanation") or {}
        check("检测报告含 L3 定位热力图（页面②渲染）", bool(loc.get("heatmap_png_b64")))
        check("检测报告含关注区域解释（块级归因，与 L3 严格区分）",
              bool(exp.get("heatmap_png_b64") or exp.get("attribution_png_b64") or exp.get("tiles")))
        check("检测报告含 forensic_clues 三态线索", isinstance(rep.get("forensic_clues"), list))
        check("检测报告含版本哈希（可追溯）",
              bool((rep.get("versions") or {}).get("input_sha256")))
        check("检测报告已落盘并给出 report_url（样本库回取链路）",
              bool((rep.get("meta") or {}).get("report_url")))
        rid = (rep.get("meta") or {}).get("report_id")
        if rid:
            st2, body2, _ = get(base, "/api/v1/report/%s" % rid)
            check("落盘报告可按 report_id 回取（=200）", st2 == 200 and json.loads(body2).get("verdict") == rep.get("verdict"))

        # 超时降级：HTTP 200 + review + degraded（不是 HTTP 错误）
        st, body, _ = post_bytes(base, "/api/v1/detect/image?timeout_ms=1&save=0&name=%s" % sample.name,
                                 sample.read_bytes(), "application/octet-stream")
        d = json.loads(body) if st == 200 else {}
        check("单请求超时降级为 200 + verdict=review + meta.degraded",
              st == 200 and d.get("verdict") == "review" and (d.get("meta") or {}).get("degraded") is True,
              "status=%s verdict=%s degraded=%s" % (st, d.get("verdict"), (d.get("meta") or {}).get("degraded")))

    # ---- 6. 批量端点（页面②的批量面板） -----------------------------------------
    if sample.is_file():
        boundary = "----webuicheck"
        part = (b"--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                b"Content-Type: image/png\r\n\r\n" % (boundary.encode(), sample.name.encode()))
        payload = part + sample.read_bytes() + b"\r\n--" + boundary.encode() + b"--\r\n"
        st, body, _ = post_bytes(base, "/api/v1/detect/batch?save=false",
                                 payload, "multipart/form-data; boundary=%s" % boundary)
        d = json.loads(body) if st == 200 else {}
        check("POST /api/v1/detect/batch = 200 且返回 count/summary/results",
              st == 200 and d.get("count") == 1 and "summary" in d and len(d.get("results") or []) == 1,
              "status=%s count=%s" % (st, d.get("count")))

    print("\n===== 汇总：%d 项通过 / %d 项失败 =====" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
