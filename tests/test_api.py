#!/usr/bin/env python3
"""api/server.py 端点自测（stdlib，无 pytest 依赖）。

两种用法:
  1) 直接跑：`python tests/test_api.py` —— 自动拉起 stub 服务跑全套端点检查，
     若检测到模型则再拉起 full 服务跑真实推理/预热/超时降级检查。
  2) 对已运行实例跑：`python tests/test_api.py --base-url http://127.0.0.1:8080`
     （api/server.py --selftest 就是这样在进程内调用 run_suite 的）

覆盖：health / version / openapi / metrics / reports / report 回取 / 静态托管 /
multipart + 原始字节 + JSON 路径三种输入 / 批量 / 统一错误体 / 错误码 /
路径穿越防护 / 并发 / 单请求超时降级。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_PREFIX = "/api/v1"
REPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
VERDICTS = {"normal", "review", "suspect"}


# --------------------------------------------------------------------------- #
# 请求工具
# --------------------------------------------------------------------------- #
class HttpResult:
    def __init__(self, status: int, headers, body: bytes):
        self.status = status
        self.headers = headers
        self.body = body

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def request(method: str, url: str, data: bytes | None = None,
            headers: dict | None = None, timeout: float = 300.0) -> HttpResult:
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return HttpResult(r.status, dict(r.headers), r.read())
    except urllib.error.HTTPError as e:
        return HttpResult(e.code, dict(e.headers or {}), e.read())


def multipart_body(parts: list[tuple], boundary: str) -> bytes:
    out = bytearray()
    for name, filename, data, ctype in parts:
        out += f"--{boundary}\r\n".encode()
        cd = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            cd += f'; filename="{filename}"'
        out += (cd + "\r\n").encode()
        if ctype:
            out += f"Content-Type: {ctype}\r\n".encode()
        out += b"\r\n" + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out)


def post_multipart(base: str, path: str, files: list[tuple[str, bytes]],
                   query: str = "", field: str = "file") -> HttpResult:
    boundary = "----opensz01test" + str(int(time.time() * 1000))
    parts = [(field, name, data, "application/octet-stream") for name, data in files]
    body = multipart_body(parts, boundary)
    url = base + path + (("?" + query) if query else "")
    return request("POST", url, body, {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    })


def envelope_error(res: HttpResult) -> tuple[str | None, str | None]:
    """统一错误体检查：返回 (code, message)；结构不符返回 (None, None)。"""
    try:
        payload = res.json()
    except Exception:  # noqa: BLE001
        return None, None
    err = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(err, dict):
        return None, None
    return err.get("code"), err.get("message")


# --------------------------------------------------------------------------- #
# 测试素材（无外部依赖：自造 PNG；有 cv2 则自造 mp4）
# --------------------------------------------------------------------------- #
def make_png_bytes(w: int = 96, h: int = 96, seed: int = 7) -> bytes:
    """纯 stdlib 生成一张小 PNG（真彩、带细微梯度，便于 L1 特征计算）。"""
    raw = bytearray()
    for y in range(h):
        raw.append(0)                              # filter type 0
        for x in range(w):
            r = (x * 7 + seed * 13) % 256
            g = (y * 5 + seed * 29) % 256
            b = ((x + y) * 3 + seed * 3) % 256
            raw += bytes((r, g, b))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def make_mp4(path: Path, seconds: float = 1.0, fps: int = 8) -> bool:
    try:
        import cv2                                    # noqa: PLC0415
        import numpy as np                            # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    try:
        w = h = 96
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            return False
        for i in range(int(seconds * fps)):
            frame = np.full((h, w, 3), 40 + i * 3, dtype="uint8")
            frame[:, : w // 2] = (200, 180 - i, 60)
            vw.write(frame)
        vw.release()
        return path.is_file() and path.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def find_sample_image() -> Path | None:
    for base in (ROOT.parent / "data" / "images", ROOT / "data" / "images",
                 Path.cwd() / "data" / "images"):
        if base.is_dir():
            for p in sorted(base.iterdir()):
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    return p
    return None


def find_sample_video() -> Path | None:
    for base in (ROOT.parent / "demo_assets", ROOT / "demo_assets", ROOT / "assets",
                 Path.cwd() / "demo_assets"):
        if base.is_dir():
            for p in sorted(base.glob("*.mp4")):
                return p
    return None


# --------------------------------------------------------------------------- #
# 检查收集器
# --------------------------------------------------------------------------- #
class Suite:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        rec = {"name": name, "status": "passed" if ok else "failed", "detail": detail}
        self.checks.append(rec)
        if self.verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    def skip(self, name: str, reason: str):
        rec = {"name": name, "status": "skipped", "detail": reason}
        self.checks.append(rec)
        if self.verbose:
            print(f"  [SKIP] {name} — {reason}", flush=True)

    def ok(self) -> bool:
        return all(c["status"] != "failed" for c in self.checks)

    def summary(self) -> dict:
        return {
            "passed": sum(1 for c in self.checks if c["status"] == "passed"),
            "failed": sum(1 for c in self.checks if c["status"] == "failed"),
            "skipped": sum(1 for c in self.checks if c["status"] == "skipped"),
            "total": len(self.checks),
        }


def _unit_multipart_parser(s: Suite):
    """multipart 解析器直接单测：二进制安全（含 CRLF/零字节/近 boundary 内容）。"""
    name = "unit.multipart_parser"
    try:
        spec = importlib.util.spec_from_file_location("open_sz_api_server",
                                                      ROOT / "api" / "server.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod      # dataclasses 需要模块已注册（_is_type 查 sys.modules）
        spec.loader.exec_module(mod)
        boundary = "----unitboundary"
        payload = b"\x00\x01\r\n--not-the-boundary\r\n\xff" * 3
        body = multipart_body([("file", "a.png", payload, "image/png")], boundary)
        parts = mod.parse_multipart(body, boundary)
        ok = (len(parts) == 1 and parts[0][0].get("filename") == "a.png"
              and parts[0][1] == payload)
        s.check(name, ok, f"parts={len(parts)} binary_safe={ok}")
    except Exception as e:  # noqa: BLE001
        s.check(name, False, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 主测试流程
# --------------------------------------------------------------------------- #
def run_suite(base_url: str, *, full_mode: bool = False, verbose: bool = True,
              timeout_degrade: bool = True) -> dict:
    base = base_url.rstrip("/")
    s = Suite(verbose)
    if verbose:
        print(f"[test] base_url={base} mode={'full' if full_mode else 'stub'}", flush=True)

    img = find_sample_image()
    if img is not None:
        img_bytes, img_name = img.read_bytes(), img.name
        img_source = str(img)
    else:
        img_bytes, img_name, img_source = make_png_bytes(), "synthetic.png", "synthetic"
    if verbose:
        print(f"[test] image = {img_source} ({len(img_bytes)} B)", flush=True)

    # ---------------- meta 端点 ----------------
    r = request("GET", f"{base}{API_PREFIX}/health")
    s.check("health.status", r.status == 200, f"HTTP {r.status}")
    health = {}
    try:
        health = r.json()
    except Exception as e:  # noqa: BLE001
        s.check("health.json", False, str(e))
    if health:
        s.check("health.json", True, "解析成功")
        s.check("health.mode", health.get("mode") in ("full", "stub"),
                f"mode={health.get('mode')}")
        s.check("health.status_field", health.get("status") in ("ok", "degraded"),
                f"status={health.get('status')}")
        ch = health.get("channels") or {}
        want = ["L0_input", "L1_forensics", "L1b_domain", "L3_localization", "L2_clip",
                "L2_clip_probe", "head_forensic", "head_combined", "fusion"]
        missing = [k for k in want if k not in ch]
        s.check("health.channels", not missing, f"missing={missing}")
        s.check("health.limits", isinstance(health.get("limits"), dict)
                and "timeout_ms" in (health.get("limits") or {}),
                f"limits={health.get('limits')}")
        s.check("health.queue", isinstance(health.get("queue"), dict),
                f"queue={health.get('queue')}")
        s.check("health.warmup_field", "warmup" in health, f"warmup={health.get('warmup')}")

    r = request("GET", f"{base}{API_PREFIX}/version")
    s.check("version.status", r.status == 200, f"HTTP {r.status}")
    ver = {}
    try:
        ver = r.json()
    except Exception as e:  # noqa: BLE001
        s.check("version.json", False, str(e))
    if ver:
        s.check("version.json", True, "解析成功")
        code = ver.get("code") or {}
        code_hash = code.get("code_hash")
        s.check("version.code_hash",
                isinstance(code_hash, str) and len(code_hash) == 64,
                f"code_hash={str(code_hash)[:16]}… files={code.get('n_source_files')}")
        s.check("version.models", isinstance(ver.get("models"), dict) and ver["models"],
                f"n_models={len(ver.get('models') or {})}")
        s.check("version.thresholds", isinstance(ver.get("thresholds"), dict),
                f"thresholds keys={sorted((ver.get('thresholds') or {}).keys())}")
        s.check("version.report_schema", bool(ver.get("report_schema_version")),
                f"report_schema_version={ver.get('report_schema_version')}")
        if full_mode:
            s.check("version.thresholds.low_high",
                    isinstance((ver.get("thresholds") or {}).get("low"), (int, float))
                    and isinstance((ver.get("thresholds") or {}).get("high"), (int, float)),
                    f"low/high={(ver.get('thresholds') or {}).get('low')}/"
                    f"{(ver.get('thresholds') or {}).get('high')}")

    r = request("GET", f"{base}{API_PREFIX}/openapi.json")
    s.check("openapi.status", r.status == 200, f"HTTP {r.status}")
    spec = {}
    try:
        spec = r.json()
    except Exception as e:  # noqa: BLE001
        s.check("openapi.json", False, str(e))
    if spec:
        s.check("openapi.json", True, "解析成功")
        s.check("openapi.version", str(spec.get("openapi", "")).startswith("3."),
                f"openapi={spec.get('openapi')}")
        want_paths = ["/api/v1/health", "/api/v1/version", "/api/v1/detect/image",
                      "/api/v1/detect/video", "/api/v1/detect/batch",
                      "/api/v1/report/{report_id}", "/api/v1/metrics", "/api/v1/openapi.json"]
        paths = spec.get("paths") or {}
        missing = [p for p in want_paths if p not in paths]
        s.check("openapi.paths", not missing, f"missing={missing}")
        s.check("openapi.methods",
                paths.get("/api/v1/detect/image", {}).get("post", {}).get("requestBody")
                is not None and paths.get("/api/v1/health", {}).get("get") is not None,
                "detect/image 有 requestBody、health 有 get")
        schemas = ((spec.get("components") or {}).get("schemas") or {})
        s.check("openapi.schemas", "ErrorEnvelope" in schemas and "DetectionResponse" in schemas,
                f"schemas={sorted(schemas)[:8]}…")
        s.check("openapi.servers", bool(spec.get("servers")), f"servers={spec.get('servers')}")

    r = request("GET", f"{base}{API_PREFIX}/metrics")
    if r.status == 200:
        m = r.json()
        s.check("metrics.status", True, f"n_samples={m.get('n_samples')}")
        s.check("metrics.contract",
                isinstance(m, dict) and ("oof_overall" in m or "n_samples" in m),
                f"keys={sorted(m)[:8]}…")
    else:
        code, _msg = envelope_error(r)
        s.check("metrics.status", r.status == 404 and code == "E_METRICS_MISSING",
                f"HTTP {r.status} code={code}（metrics.json 未生成时允许）")

    r = request("GET", f"{base}{API_PREFIX}/reports")
    s.check("reports.list", r.status == 200 and isinstance(r.json().get("reports"), list),
            f"HTTP {r.status} count={r.json().get('count') if r.status == 200 else '-'}")

    r = request("GET", f"{base}/")
    s.check("ui.root", r.status == 200 and "text/html" in (r.headers.get("Content-Type", "")),
            f"HTTP {r.status} ctype={r.headers.get('Content-Type')} "
            f"len={len(r.body)}")

    # ---------------- 输入模式 ----------------
    r = post_multipart(base, f"{API_PREFIX}/detect/image", [(img_name, img_bytes)])
    s.check("detect.image.multipart", r.status == 200, f"HTTP {r.status} {r.text[:200]}")
    rep = {}
    if r.status == 200:
        try:
            rep = r.json()
        except Exception as e:  # noqa: BLE001
            s.check("detect.image.json", False, str(e))
    if rep:
        s.check("detect.image.json", True, "解析成功")
        s.check("detect.image.verdict", rep.get("verdict") in VERDICTS,
                f"verdict={rep.get('verdict')} conf={rep.get('confidence')}")
        s.check("detect.image.confidence",
                isinstance(rep.get("confidence"), (int, float))
                and 0.0 <= float(rep["confidence"]) <= 1.0,
                f"confidence={rep.get('confidence')}")
        s.check("detect.image.input_hash",
                isinstance(rep.get("input_hash"), str) and len(rep["input_hash"]) == 64,
                f"input_hash={str(rep.get('input_hash'))[:16]}…")
        s.check("detect.image.uncertainty", isinstance(rep.get("uncertainty"), list),
                f"uncertainty={rep.get('uncertainty')}")
        meta = rep.get("meta") or {}
        rid = meta.get("report_id")
        s.check("detect.image.report_id", bool(rid) and bool(REPORT_ID_RE.match(str(rid))),
                f"report_id={rid}")
        s.check("detect.image.latency",
                isinstance(rep.get("latency_ms"), dict) and rep["latency_ms"],
                f"latency_ms keys={sorted(rep.get('latency_ms') or {})}")
        s.check("detect.image.meta", meta.get("api_version") == "v1"
                and isinstance(meta.get("elapsed_ms"), (int, float)),
                f"meta={ {k: meta.get(k) for k in ('api_version', 'elapsed_ms', 'timeout_ms', 'degraded')} }")

        if rid:
            r2 = request("GET", f"{base}{API_PREFIX}/report/{rid}")
            s.check("report.get.roundtrip", r2.status == 200, f"HTTP {r2.status}")
            if r2.status == 200:
                stored = r2.json()
                s.check("report.get.identity",
                        stored.get("input_hash") == rep.get("input_hash")
                        and (stored.get("meta") or {}).get("report_id") == rid,
                        f"stored report_id={(stored.get('meta') or {}).get('report_id')}")
                s.check("report.get.verdict_matches",
                        stored.get("verdict") == rep.get("verdict"),
                        f"{stored.get('verdict')} vs {rep.get('verdict')}")
                if full_mode:
                    s.check("full.evidence_present",
                            bool(stored.get("model_scores")),
                            f"model_scores={stored.get('model_scores')}")
                    s.check("full.clues_present",
                            isinstance(stored.get("forensic_clues"), list)
                            and len(stored["forensic_clues"]) >= 3,
                            f"n_clues={len(stored.get('forensic_clues') or [])}")

    # 原始字节 + ?name=
    r = request("POST", f"{base}{API_PREFIX}/detect/image?name={urllib.parse.quote(img_name)}&save=0",
                img_bytes, {"Content-Type": "application/octet-stream",
                            "Content-Length": str(len(img_bytes))})
    s.check("detect.image.rawbytes", r.status == 200 and r.json().get("verdict") in VERDICTS,
            f"HTTP {r.status} verdict={r.json().get('verdict') if r.status == 200 else '-'}")
    if r.status == 200:
        m = (r.json().get("meta") or {})
        s.check("detect.image.save0_not_persisted", m.get("report_id") is None
                and m.get("saved_path") is None, f"meta.report_id={m.get('report_id')}")

    # JSON 路径模式（本机部署可用；非本机监听时服务端会禁用）
    if img is not None and not str(img).startswith("synthetic"):
        body = json.dumps({"path": str(img)}).encode()
        r = request("POST", f"{base}{API_PREFIX}/detect/image?save=0", body,
                    {"Content-Type": "application/json", "Content-Length": str(len(body))})
        if r.status == 200:
            s.check("detect.image.json_path", r.json().get("verdict") in VERDICTS,
                    f"verdict={r.json().get('verdict')}")
        else:
            code, _ = envelope_error(r)
            s.check("detect.image.json_path", code in ("E_FORBIDDEN_PATH", "E_NOT_FOUND"),
                    f"HTTP {r.status} code={code}（路径模式被禁用亦视为符合设计）")

    # 空体 / 非空但不支持的 Content-Type
    r = request("POST", f"{base}{API_PREFIX}/detect/image", b"",
                {"Content-Type": "application/octet-stream", "Content-Length": "0"})
    code, _ = envelope_error(r)
    s.check("error.empty_body", r.status == 400 and code in ("E_NO_FILE", "E_BAD_REQUEST"),
            f"HTTP {r.status} code={code}")

    r = request("POST", f"{base}{API_PREFIX}/detect/image", b"\x00\x01\x02",
                {"Content-Type": "application/x-msgpack", "Content-Length": "3"})
    code, _ = envelope_error(r)
    s.check("error.unsupported_ctype", r.status == 415 and code == "E_UNSUPPORTED_MEDIA_TYPE",
            f"HTTP {r.status} code={code}")

    # ---------------- 视频 ----------------
    vid = find_sample_video()
    vid_bytes = None
    if vid is not None:
        vid_bytes, vid_name = vid.read_bytes(), vid.name
    else:
        tmp_vid = Path(tempfile.gettempdir()) / "opensz01_selftest.mp4"
        if make_mp4(tmp_vid):
            vid_bytes, vid_name = tmp_vid.read_bytes(), tmp_vid.name
    if vid_bytes:
        r = request("POST", f"{base}{API_PREFIX}/detect/video?name={urllib.parse.quote(vid_name)}",
                    vid_bytes, {"Content-Type": "video/mp4",
                                "Content-Length": str(len(vid_bytes))})
        ok = r.status == 200 and r.json().get("verdict") in VERDICTS
        s.check("detect.video.rawbytes", ok,
                f"HTTP {r.status} verdict={r.json().get('verdict') if r.status == 200 else r.text[:120]}")
        if r.status == 200 and full_mode:
            j = r.json()
            s.check("detect.video.structure",
                    isinstance(j.get("localization"), dict) and "latency_ms" in j,
                    f"localization.kind={j.get('localization', {}).get('kind')}")
    else:
        s.skip("detect.video.rawbytes", "无 mp4 素材且 cv2 不可用，无法构造视频")

    # ---------------- 批量 ----------------
    r = post_multipart(base, f"{API_PREFIX}/detect/batch?save=0",
                       [(img_name, img_bytes), ("second.png", make_png_bytes(seed=11))])
    s.check("detect.batch.multipart", r.status == 200, f"HTTP {r.status} {r.text[:200]}")
    if r.status == 200:
        b = r.json()
        s.check("detect.batch.count", b.get("count") == 2 and len(b.get("results") or []) == 2,
                f"count={b.get('count')}")
        summ = b.get("summary") or {}
        s.check("detect.batch.summary",
                isinstance(summ.get("verdicts"), dict) and "elapsed_ms" in summ
                and summ.get("n") == 2,
                f"summary={ {k: summ.get(k) for k in ('n', 'verdicts', 'ok', 'errors', 'elapsed_ms')} }")
        ok_items = all(it.get("verdict") in VERDICTS for it in b["results"])
        s.check("detect.batch.items", ok_items,
                f"verdicts={[it.get('verdict') for it in b['results']]}")

    if img is not None and not str(img).startswith("synthetic"):
        body = json.dumps({"paths": [str(img)]}).encode()
        r = request("POST", f"{base}{API_PREFIX}/detect/batch?save=0", body,
                    {"Content-Type": "application/json", "Content-Length": str(len(body))})
        if r.status == 200:
            b = r.json()
            s.check("detect.batch.json_paths",
                    b.get("count") == 1 and (b.get("summary") or {}).get("source") == "server-side",
                    f"count={b.get('count')} source={(b.get('summary') or {}).get('source')}")
        else:
            code, _ = envelope_error(r)
            s.check("detect.batch.json_paths", code in ("E_FORBIDDEN_PATH", "E_NOT_FOUND"),
                    f"HTTP {r.status} code={code}")

    # ---------------- 错误处理 / 安全 ----------------
    r = request("GET", f"{base}{API_PREFIX}/definitely-not-here")
    code, msg = envelope_error(r)
    s.check("error.not_found", r.status == 404 and code == "E_NOT_FOUND" and bool(msg),
            f"HTTP {r.status} code={code}")

    r = request("GET", f"{base}{API_PREFIX}/detect/image")
    code, _ = envelope_error(r)
    s.check("error.method_not_allowed", r.status == 405 and code == "E_METHOD_NOT_ALLOWED",
            f"HTTP {r.status} code={code}")

    r = request("GET", f"{base}{API_PREFIX}/report/no-such-report-xyz")
    code, _ = envelope_error(r)
    s.check("error.report_missing", r.status == 404 and code == "E_NOT_FOUND",
            f"HTTP {r.status} code={code}")

    r = request("GET", f"{base}{API_PREFIX}/report/..%2f..%2fwin.ini")
    code, _ = envelope_error(r)
    s.check("security.report_id_traversal", r.status in (400, 404) and code in
            ("E_BAD_REQUEST", "E_NOT_FOUND"), f"HTTP {r.status} code={code}")

    r = request("GET", f"{base}/reports/..%2f..%2frequirements.txt")
    s.check("security.static_traversal",
            r.status in (400, 403, 404),
            f"HTTP {r.status}（不允许 200 泄露仓库文件）")

    r = request("POST", f"{base}{API_PREFIX}/detect/batch", b'{"paths": ["/etc/hostname"]}',
                {"Content-Type": "application/json",
                 "Content-Length": str(len(b'{"paths": ["/etc/hostname"]}'))})
    code, _ = envelope_error(r)
    s.check("security.outside_root_path", r.status in (403, 404) and code in
            ("E_FORBIDDEN_PATH", "E_NOT_FOUND"), f"HTTP {r.status} code={code}")

    # ---------------- 并发 ----------------
    results: list[int] = []
    lock = threading.Lock()

    def _one(i: int):
        rr = request("POST", f"{base}{API_PREFIX}/detect/image?save=0&name=c{i}.png",
                     img_bytes, {"Content-Type": "application/octet-stream",
                                 "Content-Length": str(len(img_bytes))})
        with lock:
            results.append(rr.status)

    threads = [threading.Thread(target=_one, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    [t.join(timeout=300) for t in threads]
    s.check("concurrency.3x", results == [200, 200, 200], f"statuses={results}")

    # ---------------- 单请求超时降级（放最后：队列里可能仍有一单在跑） ----------------
    if timeout_degrade:
        r = request("POST", f"{base}{API_PREFIX}/detect/image?timeout_ms=1&save=0",
                    img_bytes, {"Content-Type": "application/octet-stream",
                                "Content-Length": str(len(img_bytes))}, timeout=120)
        if r.status != 200:
            s.check("timeout.degradation", False, f"HTTP {r.status} {r.text[:200]}")
        else:
            j = r.json()
            meta = j.get("meta") or {}
            degraded = bool(meta.get("degraded"))
            if full_mode:
                ok = (degraded and j.get("verdict") == "review"
                      and any("超时" in str(u) or "降级" in str(u)
                              for u in (j.get("uncertainty") or [])))
                s.check("timeout.degradation", ok,
                        f"degraded={degraded} verdict={j.get('verdict')} "
                        f"uncertainty={(j.get('uncertainty') or [])[:1]}")
                s.check("timeout.meta_timeout_ms", meta.get("timeout_ms") == 1,
                        f"meta.timeout_ms={meta.get('timeout_ms')}")
            else:
                # stub 模式推理极快：要么未触发超时、要么降级为 review，二者都合规
                ok = j.get("verdict") in VERDICTS and (not degraded or j.get("verdict") == "review")
                s.check("timeout.degradation", ok,
                        f"degraded={degraded} verdict={j.get('verdict')}（stub 模式允许未触发）")

    if full_mode:
        r = request("GET", f"{base}{API_PREFIX}/health")
        h = r.json()
        warm = h.get("warmup") or {}
        s.check("full.warmup_done", bool(warm.get("done")) and isinstance(warm.get("ms"), (int, float)),
                f"warmup={ {k: warm.get(k) for k in ('done', 'ms', 'sample', 'error')} }")
        ch = h.get("channels") or {}
        s.check("full.channels_loaded",
                bool((ch.get("L2_clip") or {}).get("loaded"))
                and bool((ch.get("head_forensic") or {}).get("loaded")),
                f"L2_clip.loaded={(ch.get('L2_clip') or {}).get('loaded')} "
                f"head_forensic.loaded={(ch.get('head_forensic') or {}).get('loaded')}")
        s.check("full.mode", h.get("mode") == "full" and h.get("status") == "ok",
                f"mode={h.get('mode')} status={h.get('status')}")

    _unit_multipart_parser(s)

    summary = s.summary()
    if verbose:
        print(f"[test] 结果：{summary['passed']} passed / {summary['failed']} failed / "
              f"{summary['skipped']} skipped", flush=True)
    return {"ok": s.ok(), "summary": summary, "checks": s.checks, "base_url": base,
            "mode": "full" if full_mode else "stub"}


# --------------------------------------------------------------------------- #
# 命令行：自己拉起服务进程再跑
# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


def wait_ready(base: str, timeout_s: float, proc: subprocess.Popen) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            r = request("GET", f"{base}{API_PREFIX}/health", timeout=5)
            if r.status == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def spawn_and_test(extra_args: list[str], *, full_mode: bool, verbose: bool,
                   start_timeout: float) -> dict:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / f"opensz01_api_{port}.log"
    log = open(log_path, "w", encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "api" / "server.py"), "--port", str(port),
           "--host", "127.0.0.1"] + extra_args
    print(f"[test] 启动服务：{' '.join(cmd)}\n[test] 日志：{log_path}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT.parent), stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(base, start_timeout, proc):
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            except OSError:
                pass
            return {"ok": False, "summary": {"passed": 0, "failed": 1, "skipped": 0, "total": 1},
                    "checks": [{"name": "server.startup", "status": "failed",
                                "detail": f"服务未在 {start_timeout}s 内就绪；日志尾部：{tail}"}],
                    "base_url": base, "mode": "full" if full_mode else "stub"}
        return run_suite(base, full_mode=full_mode, verbose=verbose)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OPEN-SZ-01 API 端点自测")
    ap.add_argument("--base-url", default=None, help="对已运行实例测试（不再拉起进程）")
    ap.add_argument("--fast", action="store_true", help="只跑 stub 模式，不跑真实模型")
    ap.add_argument("--no-stub", action="store_true", help="跳过 stub，只跑真实模型")
    ap.add_argument("--no-degrade-check", action="store_true", help="跳过超时降级检查")
    ap.add_argument("--start-timeout", type=float, default=300.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    verbose = not args.quiet
    results = []

    if args.base_url:
        results.append(run_suite(args.base_url, full_mode=False, verbose=verbose,
                                 timeout_degrade=not args.no_degrade_check))
    else:
        if not args.no_stub:
            results.append(spawn_and_test(["--no-model", "--no-warmup"], full_mode=False,
                                          verbose=verbose, start_timeout=args.start_timeout))
        if not args.fast:
            models = None
            for cand in (ROOT / "models", ROOT.parent / "models"):
                if cand.is_dir() and any((cand / n).exists() for n in
                                         ("fusion.json", "clip_probe_norm.pt", "clip_probe.pt",
                                          "forensic_lr.joblib", "combined_lr.joblib")):
                    models = cand
                    break
            if models is None:
                print("[test] 未发现含模型产物的 models/ 目录，跳过 full 模式检查", flush=True)
            else:
                results.append(spawn_and_test([], full_mode=True, verbose=verbose,
                                              start_timeout=args.start_timeout))

    ok = all(r["ok"] for r in results)
    print("\n===== 自测汇总 =====")
    for r in results:
        print(f"  mode={r['mode']:5s} ok={r['ok']} {r['summary']} base={r['base_url']}")
        for c in r["checks"]:
            if c["status"] == "failed":
                print(f"    FAILED {c['name']}: {c['detail']}")
    print(f"===== {'全部通过' if ok else '存在失败'} =====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
