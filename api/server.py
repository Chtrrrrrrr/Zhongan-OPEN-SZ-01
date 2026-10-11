#!/usr/bin/env python3
"""OPEN-SZ-01 对外 REST 服务：版本化端点 + OpenAPI + 批处理（零额外依赖）。

手册 16.1「服务化」要求把检测能力以服务形式对外提供；本模块只用 Python
标准库（http.server / json / threading / hashlib ...），因此在 gradio /
fastapi / uvicorn 不可用的环境里也能直接运行（本环境实测三者可导入，但
stdlib 版本仍是主交付，见 docs/api.md）。

端点（base = /api/v1）:
  GET  /                          -> 静态托管 demo/webui.html（缺失则回退页）
  GET  /api/v1/health             各证据通道加载状态 / 预热状态 / 队列状态
  GET  /api/v1/version            模型、阈值、代码版本与 sha256
  POST /api/v1/detect/image       图片检测（multipart / 原始字节 / JSON 路径）
  POST /api/v1/detect/video       视频检测（同上）
  POST /api/v1/detect/batch       多文件批量检测（逐条摘要 + 汇总）
  GET  /api/v1/report/{id}        回取已落盘报告
  GET  /api/v1/reports            已落盘报告列表（便于 WebUI 展示）
  GET  /api/v1/metrics            返回 reports/metrics.json
  GET  /api/v1/openapi.json       OpenAPI 3.0 描述

工程约束（手册 16.2/16.3）:
  * 统一错误体 {"error": {"code", "message", ...}}
  * 单请求超时降级：推理超过 timeout 时返回 review 结论并标注 degraded=true
  * 模型常驻 + 启动预热（首个请求不再支付 ~14-20s 冷启动）
  * 请求日志（stderr + JSONL 落盘）
  * 证据生产者互为并行通道：单个通道失败不影响整体结论（不抛 500）

用法:
  python api/server.py --port 8080                 # 真实模型（默认预热）
  python api/server.py --port 8080 --no-model      # 无模型占位（自测/CI 用）
  python api/server.py --selftest                  # 启动自测并跑通端点
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import importlib.util
import json
import os
import queue
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
SERVICE_NAME = "open-sz-01-api"
SERVICE_VERSION = "1.0.0"
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
REPORT_SCHEMA_VERSION = "1.1.0"          # 与 src/report.py:REPORT_VERSION 对齐（见 pipeline_report_schema_version）
REPORT_SCHEMA_VERSION_FALLBACK = REPORT_SCHEMA_VERSION
PIPELINE_VERSION = "open-sz-01 v0.3"

ROOT = Path(__file__).resolve().parent.parent          # <repo>/api/server.py -> <repo>
DOCS_SRC_EXT = {".py", ".md", ".json", ".html", ".js", ".css", ".txt"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}

# 错误码表：code -> (HTTP status, 默认消息)。docs/api.md 与本表保持同步。
ERROR_CODES: dict[str, tuple[int, str]] = {
    "E_BAD_REQUEST": (400, "请求格式不正确（缺少/非法参数）"),
    "E_NO_FILE": (400, "未提供文件：请用 multipart 字段 file、原始字节体或 JSON path"),
    "E_UNSUPPORTED_MEDIA_TYPE": (415, "不支持的 Content-Type"),
    "E_FORBIDDEN_PATH": (403, "禁止的服务端路径（超出允许的根目录或已禁用路径输入）"),
    "E_NOT_FOUND": (404, "资源不存在"),
    "E_METHOD_NOT_ALLOWED": (405, "该端点不支持此 HTTP 方法"),
    "E_PAYLOAD_TOO_LARGE": (413, "请求体超过上限"),
    "E_BUSY": (429, "推理队列已满，请稍后重试"),
    "E_METRICS_MISSING": (404, "reports/metrics.json 不存在"),
    "E_MODEL_UNAVAILABLE": (503, "模型或证据通道不可用"),
    "E_INTERNAL": (500, "服务内部错误"),
}

MEDIA_BY_EXT = {**{e: "image" for e in IMAGE_EXTS}, **{e: "video" for e in VIDEO_EXTS}}


class ApiError(Exception):
    """携带错误码的业务异常，由 Handler 统一渲染为错误体。"""

    def __init__(self, code: str, message: str | None = None, *, details=None,
                 status: int | None = None):
        default_status, default_msg = ERROR_CODES.get(code, (500, "服务内部错误"))
        self.code = code
        self.message = message or default_msg
        self.status = status or default_status
        self.details = details
        super().__init__(f"{code}: {self.message}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, _cache: dict | None = None) -> str | None:
    """文件 sha256（带缓存；失败返回 None，绝不影响主流程）。"""
    key = str(path)
    if _cache is not None and key in _cache:
        return _cache[key]
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    except OSError:
        return None
    if _cache is not None:
        _cache[key] = digest
    return digest


def is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


# --------------------------------------------------------------------------- #
# 路径发现：模型 / 数据 / 报告 可能在仓库内，也可能在仓库上一级（本工程实测）
# --------------------------------------------------------------------------- #
def _uniq(paths) -> list[Path]:
    out, seen = [], set()
    for p in paths:
        try:
            key = str(Path(p).resolve())
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            out.append(Path(p))
    return out


def models_dir_has_artifacts(d: Path) -> bool:
    """models/ 目录里是否真的有模型产物（repo/models 可能只有 .gitkeep）。"""
    if not d.is_dir():
        return False
    for name in ("fusion.json", "clip_probe_norm.pt", "clip_probe.pt", "forensic_lr.joblib",
                 "combined_lr.joblib", "clip_domain_lr.joblib", "l1_lr.joblib", "npr.pt"):
        if (d / name).exists():
            return True
    if (d / "hf_cache").is_dir():
        return True
    return any(p.suffix in (".pt", ".joblib", ".pth") for p in d.glob("*") if p.is_file())


def find_models_dir(explicit: str | None = None) -> Path | None:
    cands = [Path(explicit)] if explicit else []
    cands += [ROOT / "models", ROOT.parent / "models", Path.cwd() / "models"]
    uniq = _uniq(cands)
    for c in uniq:                     # 优先选真的有模型产物的目录
        if models_dir_has_artifacts(c):
            return c
    for c in uniq:                     # 退而求其次：任何存在的 models 目录
        if c.is_dir():
            return c
    return None


def data_dir_has_samples(d: Path) -> bool:
    """data/ 里是否真有样本（repo/data 可能只有 .gitkeep）。"""
    if not d.is_dir():
        return False
    for name in ("labels.csv", "manifest.json", "splits.json"):
        if (d / name).exists():
            return True
    img = d / "images"
    if img.is_dir() and any(p.is_file() for p in img.iterdir()):
        return True
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS for p in d.iterdir())


def find_data_dir(explicit: str | None = None) -> Path | None:
    cands = [Path(explicit)] if explicit else []
    cands += [ROOT / "data", ROOT.parent / "data", Path.cwd() / "data"]
    uniq = _uniq(cands)
    for c in uniq:
        if data_dir_has_samples(c):
            return c
    for c in uniq:
        if c.is_dir():
            return c
    return None


def find_reports_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    for c in _uniq([ROOT / "reports", ROOT.parent / "reports", Path.cwd() / "reports"]):
        if c.is_dir():
            return c
    return ROOT / "reports"


def report_search_dirs(primary: Path, explicit: list[str] | None = None) -> list[Path]:
    """报告查找目录（按优先级）：主目录 -> 仓库 reports/ -> 工作区 reports/ 及其 api 子目录。

    变更是为了让 WebUI 样本库能回取到"实际存在的"报告：仓库内的 reports/ 只保留
    聚合指标与 batch 归档，逐样本 0xx.json 由流水线写在工作区 reports/；两者都查，
    避免出现"repo 里没有该 id 就 404"的情况（F9 接口刷新）。
    """
    cands = [Path(primary)]
    cands += [Path(p) for p in (explicit or [])]
    cands += [ROOT / "reports", ROOT.parent / "reports",
              ROOT.parent / "reports" / "api", Path.cwd() / "reports"]
    out: list[Path] = []
    seen: set[str] = set()
    for c in cands:
        try:
            key = str(Path(c).resolve())
        except OSError:
            continue
        if key in seen or not Path(c).is_dir():
            continue
        seen.add(key)
        out.append(Path(c))
    return out


def resolve_clip_dir(base: Path | None) -> Path | None:
    """HF snapshots 目录 -> 具体 snapshot（含 config.json 的优先）。"""
    if base is None or not base.is_dir():
        return None
    if (base / "config.json").exists():
        return base
    subs = [d for d in base.iterdir() if d.is_dir()]
    for d in subs:
        if (d / "config.json").exists():
            return d
    return subs[0] if subs else None


# --------------------------------------------------------------------------- #
# 推理队列：模型常驻 + 串行化 GPU 访问 + 单请求超时降级
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Job:
    fn: object
    label: str = ""
    event: threading.Event = dataclasses.field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None
    t_submit: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def ms(self) -> float:
        return max(0.0, (self.t_end - self.t_start) * 1000.0) if self.t_end else 0.0


class InferenceQueue:
    """后台工作线程消费队列；请求方按 deadline 等待，超时即降级返回。

    与「每请求一个线程 + 超时丢弃」的做法相比，这里的关键差别是：超时后
    工作线程仍会把当前这一单跑完（GPU 状态不被破坏），但结果被记录为
    timeout_discarded，调用方立即拿到降级结论。
    """

    def __init__(self, workers: int = 1, max_queue: int = 32):
        self.workers = max(1, int(workers))
        self.max_queue = max(1, int(max_queue))
        self.q: "queue.Queue[Job | None]" = queue.Queue(maxsize=self.max_queue)
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.submitted = 0
        self.completed = 0
        self.timed_out = 0
        self.errors = 0
        self.discarded = 0
        self.rejected = 0

    def start(self):
        for i in range(self.workers):
            t = threading.Thread(target=self._loop, name=f"infer-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            job.t_start = time.perf_counter()
            try:
                job.result = job.fn()
            except BaseException as e:  # noqa: BLE001 - 任何模型异常都降级，不杀死工作线程
                job.error = e
                with self._lock:
                    self.errors += 1
            finally:
                job.t_end = time.perf_counter()
                with self._lock:
                    self.completed += 1
                    if job.event.is_set():
                        # 调用方已超时返回，这单结果被丢弃
                        self.discarded += 1
                job.event.set()

    def submit(self, fn, label: str = "") -> Job:
        job = Job(fn=fn, label=label, t_submit=time.perf_counter())
        try:
            self.q.put_nowait(job)
        except queue.Full:
            with self._lock:
                self.rejected += 1
            raise ApiError("E_BUSY", f"推理队列已满（max_queue={self.max_queue}）")
        with self._lock:
            self.submitted += 1
        if not self._threads:      # 兜底：未 start() 时同步执行，端点仍可用
            self._loop_once(job)
        return job

    def _loop_once(self, job: Job):
        job.t_start = time.perf_counter()
        try:
            job.result = job.fn()
        except BaseException as e:  # noqa: BLE001
            job.error = e
        finally:
            job.t_end = time.perf_counter()
            job.event.set()

    def run(self, fn, timeout_s: float, label: str = "") -> tuple[object, bool]:
        """返回 (result, timed_out)；timed_out=True 时 result 为 None。"""
        job = self.submit(fn, label=label)
        got = job.event.wait(timeout=max(0.001, timeout_s))
        if not got:
            with self._lock:
                self.timed_out += 1
            return None, True
        if job.error is not None:
            raise job.error
        return job.result, False

    def stats(self) -> dict:
        with self._lock:
            return {
                "workers": self.workers,
                "max_queue": self.max_queue,
                "queued": self.q.qsize(),
                "submitted": self.submitted,
                "completed": self.completed,
                "errors": self.errors,
                "timed_out": self.timed_out,
                "timeout_discarded": self.discarded,
                "rejected_busy": self.rejected,
            }


# --------------------------------------------------------------------------- #
# 检测服务：真实流水线（src/）或占位（--no-model / 模型缺失）
# --------------------------------------------------------------------------- #
class DetectorService:
    def __init__(self, cfg: "ServerConfig", logger):
        self.cfg = cfg
        self.log = logger
        self.pipeline = load_pipeline_modules(logger)
        self.mode = "stub"
        self.detector = None
        self.stub_reason = ""
        self.model_paths: dict[str, Path | None] = {}
        self.hash_cache: dict[str, str] = {}
        self.warmup = {"done": False, "ms": None, "sample": None, "error": None,
                       "started_at": None, "finished_at": None}
        self._build()

    # ---------------- 构建 ----------------
    def _model_candidates(self) -> dict[str, Path | None]:
        m = self.cfg.models_dir
        def opt(*rel):
            if m is None:
                return None
            for r in rel:
                p = m / r
                if p.exists():
                    return p
            return None
        return {
            "clip_dir": resolve_clip_dir(opt("hf_cache/models--openai--clip-vit-large-patch14/snapshots",
                                             "clip-vit-large-patch14", "clip")) if m else None,
            "clip_probe": opt("clip_probe_norm.pt", "clip_probe.pt"),
            "l1_lr_model": opt("l1_lr.joblib"),
            "forensic_lr": opt("forensic_lr.joblib"),
            "clip_domain_lr": opt("clip_domain_lr.joblib"),
            "combined_lr": opt("combined_lr.joblib"),
            "npr_weights": opt("npr.pt") if self.cfg.enable_npr else None,
            "vae_dir": opt("vae", "sd-vae") if self.cfg.enable_aeroblade else None,
            "fusion_json": opt("fusion.json"),
        }

    def _build(self):
        self.model_paths = self._model_candidates()
        if self.cfg.no_model:
            self.stub_reason = "以 --no-model 启动（占位实现，仅用于端点自测）"
            return
        if self.pipeline is None:
            self.stub_reason = "无法导入 src/ 流水线模块（占位实现）"
            return
        available = [k for k, v in self.model_paths.items() if v is not None]
        if not available:
            self.stub_reason = (f"未找到模型文件（models_dir="
                                f"{self.cfg.models_dir}），以占位实现运行")
            return
        try:
            PipelineConfig = self.pipeline["detector"].PipelineConfig
            AIGCDetector = self.pipeline["detector"].AIGCDetector
            pc = self.model_paths
            cfg = PipelineConfig(
                clip_dir=str(pc["clip_dir"]) if pc["clip_dir"] else None,
                clip_probe=str(pc["clip_probe"]) if pc["clip_probe"] else None,
                npr_weights=str(pc["npr_weights"]) if pc["npr_weights"] else None,
                vae_dir=str(pc["vae_dir"]) if pc["vae_dir"] else None,
                fusion_json=str(pc["fusion_json"]) if pc["fusion_json"] else None,
                l1_lr_model=str(pc["l1_lr_model"]) if pc["l1_lr_model"] else None,
                clip_domain_lr=str(pc["clip_domain_lr"]) if pc["clip_domain_lr"] else None,
                combined_lr=str(pc["combined_lr"]) if pc["combined_lr"] else None,
                forensic_lr=str(pc["forensic_lr"]) if pc["forensic_lr"] else None,
                enable_aeroblade=self.cfg.enable_aeroblade,
                time_budget_s=self.cfg.time_budget_s,
                device=self.cfg.device,
            )
            self.detector = AIGCDetector(cfg)
            self.mode = "full"
            self.log(f"[startup] 检测器已构建：device={self.detector.device} "
                     f"通道={','.join(sorted(available))}")
        except Exception as e:  # noqa: BLE001 - 构建失败也必须让服务起来（降级）
            self.stub_reason = f"检测器构建失败：{type(e).__name__}: {e}"
            self.mode = "stub"
            self.detector = None
            self.log(f"[startup][warn] {self.stub_reason}")
            self.log(traceback.format_exc(limit=3))

    # ---------------- 预热 ----------------
    def warmup_sample(self) -> Path | None:
        if self.cfg.warmup_image:
            p = Path(self.cfg.warmup_image)
            if p.is_file():
                return p
        for base in _uniq([self.cfg.data_dir, ROOT.parent / "data", ROOT / "data"]):
            if base is None:
                continue
            img_dir = Path(base) / "images" if Path(base).name != "images" else Path(base)
            if img_dir.is_dir():
                for p in sorted(img_dir.iterdir()):
                    if p.suffix.lower() in IMAGE_EXTS:
                        return p
        return None

    def warm_up(self):
        """启动预热：把 L0/L1/L2 权重与 cudnn 调优成本提前付掉。"""
        if self.cfg.no_warmup or self.mode == "stub":
            self.warmup["error"] = None if self.cfg.no_warmup else self.stub_reason
            self.warmup["skipped"] = True
            return
        sample = self.warmup_sample()
        self.warmup["started_at"] = utc_now_iso()
        t0 = time.perf_counter()
        if sample is None:
            self.warmup["error"] = "未找到预热样本（--warmup-image 或 data/images/*）"
            self.warmup["skipped"] = True
            self.log("[startup][warn] " + self.warmup["error"])
            return
        try:
            self.detect_image(str(sample))          # 直接调用，不经过队列
            self.warmup.update(done=True, sample=str(sample),
                               ms=round((time.perf_counter() - t0) * 1000, 1),
                               error=None)
            self.log(f"[startup] 预热完成 {self.warmup['ms']} ms（{sample.name}）")
        except Exception as e:  # noqa: BLE001 - 预热失败不阻塞服务
            self.warmup.update(done=False, error=f"{type(e).__name__}: {e}",
                               ms=round((time.perf_counter() - t0) * 1000, 1))
            self.log(f"[startup][warn] 预热失败（继续提供服务）：{self.warmup['error']}")
        finally:
            self.warmup["finished_at"] = utc_now_iso()

    # ---------------- 报告构造 ----------------
    def _ReportClass(self):
        return self.pipeline["report"].DetectionReport if self.pipeline else None

    def _new_report(self, kind: str, path: str):
        cls = self._ReportClass()
        if cls is not None:
            return cls(input_type=kind)
        return {"input_type": kind, "verdict": "review", "model_scores": {},
                "forensic_clues": [], "localization": {}, "uncertainty": [],
                "versions": {}, "latency_ms": {}, "confidence": 0.0,
                "fused_score": 0.0, "disagreement": 0.0, "input_hash": "",
                "resolution": "", "duration_s": None, "verdict_cn": "待复核"}

    @staticmethod
    def report_to_dict(rep) -> dict:
        if hasattr(rep, "finalize"):
            try:
                rep.finalize()
            except Exception:  # noqa: BLE001
                pass
        if dataclasses.is_dataclass(rep):
            return dataclasses.asdict(rep)
        return dict(rep)

    def degraded_report(self, kind: str, path: str, reason: str,
                        extra: dict | None = None) -> dict:
        """降级结论：任何通道不可用都返回 review + 不确定性说明，而不是 500。"""
        rep = self._new_report(kind, path)
        d = self.report_to_dict(rep) if not isinstance(rep, dict) else rep
        d["verdict"] = "review"
        d["verdict_cn"] = "待复核"
        d["uncertainty"] = list(d.get("uncertainty") or []) + [reason]
        if extra:
            d.setdefault("latency_ms", {})
            d["latency_ms"].update(extra)
        d["versions"] = dict(d.get("versions") or {})
        d["versions"].update({"pipeline": PIPELINE_VERSION, "channel": f"{kind}-degraded",
                              "mode": self.mode})
        if path and Path(path).exists() and not d.get("input_hash"):
            d["input_hash"] = sha256_file(Path(path), self.hash_cache) or ""
        return d

    # ---------------- 检测 ----------------
    def detect_image(self, path: str, *, save_html: bool = False,
                     report_dir: Path | None = None) -> dict:
        if self.mode != "full":
            return self.degraded_report("image", path,
                                        f"模型未加载（{self.stub_reason}）；结论保持待复核")
        rep, _conf = self.detector.detect_image(path)
        d = self.report_to_dict(rep)
        if save_html and report_dir is not None:
            try:
                self._save_html(rep, path, report_dir)
            except Exception as e:  # noqa: BLE001
                self.log(f"[warn] HTML 报告落盘失败：{e}")
        return d

    def detect_video(self, path: str) -> dict:
        if self.mode != "full":
            return self.degraded_report("video", path,
                                        f"模型未加载（{self.stub_reason}）；结论保持待复核")
        rep = self.detector.detect_video(path)
        return self.report_to_dict(rep)

    def _save_html(self, rep, media_path: str, report_dir: Path):
        save_html = self.pipeline["report"].save_html
        try:
            with open(media_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except OSError:
            b64 = None
        stem = Path(media_path).stem
        out = report_dir / f"{stem}.html"
        save_html(rep, str(out), image_b64=b64)

    def run_detection(self, kind: str, path: str, timeout_s: float,
                      queue_: InferenceQueue, *, save_html: bool = False,
                      report_dir: Path | None = None) -> tuple[dict, bool]:
        """(report_dict, degraded)。超时 -> 队列继续跑，这里立即返回 review 降级结论。"""
        if kind == "video":
            fn = lambda: self.detect_video(path)                      # noqa: E731
        else:
            fn = lambda: self.detect_image(path, save_html=save_html,  # noqa: E731
                                           report_dir=report_dir)
        try:
            res, timed_out = queue_.run(fn, timeout_s, label=f"{kind}:{Path(path).name}")
        except ApiError:
            raise
        except Exception as e:  # noqa: BLE001 - 单通道失败 -> 降级结论
            self.log(f"[warn] 检测异常 {kind} {path}: {type(e).__name__}: {e}")
            return self.degraded_report(
                kind, path, f"检测过程异常（{type(e).__name__}: {e}）；结论保持待复核"), False
        if timed_out:
            return self.degraded_report(
                kind, path,
                f"单请求超时降级：推理超过 {timeout_s:.1f}s 未返回（队列继续完成该单），"
                f"按要求返回 review 结论，请转人工复核"), True
        return res, False

    # ---------------- 状态 ----------------
    def _loaded(self, attr: str) -> bool:
        return getattr(self.detector, attr, None) is not None

    def channel_status(self) -> dict:
        """健康检查用：只读已加载状态，绝不触发惰性加载。"""
        ch: dict[str, dict] = {}
        ch["L0_input"] = {"kind": "容器/解码通道（stdlib+cv2）", "loaded": True,
                          "role": "输入解析、sha256/phash、长边归一化"}
        ch["L1_forensics"] = {"kind": "零权重取证特征（47维）", "loaded": True,
                              "role": "PNG 容器指纹 / 块效应 / 残差自相关"}
        ch["L1b_domain"] = {"kind": "零权重域取证（相机 vs 渲染）", "loaded": True,
                            "role": "噪声底/光照梯度/白点质量"}
        ch["L3_localization"] = {"kind": "零权重异常图（ELA+噪声不一致）", "loaded": True,
                                 "role": "伪造区域热力图"}
        specs = [
            ("L2_clip", "_clip", "clip_dir", "CLIP ViT-L/14 特征骨干（常驻）"),
            ("L2_clip_probe", "_clip_head", "clip_probe", "CLIP 线性探针（外部冻结模型）"),
            ("L2_npr", "_npr", "npr_weights", "NPR 邻域像素回归（默认消融）"),
            ("L2_aeroblade", "_blade", "vae_dir", "AEROBLADE 重建误差"),
            ("head_l1", "_l1_pack", "l1_lr_model", "L1 LR 头"),
            ("head_forensic", "_forensic_pack", "forensic_lr", "取证 LR 头（L1+L1b）"),
            ("head_clip_domain", "_clip_domain_pack", "clip_domain_lr", "CLIP 域 LR 头"),
            ("head_combined", "_combined_pack", "combined_lr", "联合 LR 头（L1+L1b+CLIP+tiles）"),
        ]
        for name, attr, cfg_key, role in specs:
            p = self.model_paths.get(cfg_key)
            entry = {"kind": role, "configured": p is not None,
                     "available": bool(p and Path(p).exists()),
                     "loaded": self._loaded(attr), "path": str(p) if p else None}
            if self.mode == "stub":
                entry["loaded"] = False
            ch[name] = entry
        fusion = getattr(self.detector, "fusion", None)
        ch["fusion"] = {"kind": "加权 logit 融合 + 温度标定 + 三档决策",
                        "loaded": fusion is not None, "source":
                        str(self.model_paths.get("fusion_json") or "") or None}
        if fusion is not None:
            ch["fusion"].update({"weights": getattr(fusion, "weights", {}),
                                 "thr_low": round(float(getattr(fusion, "thr_low", 0.0)), 6),
                                 "thr_high": round(float(getattr(fusion, "thr_high", 0.0)), 6),
                                 "temperature": {k: round(v.temperature, 4) for k, v
                                                 in getattr(fusion, "scalers", {}).items()}})
        if self.mode == "stub":
            ch["_note"] = {"kind": "占位实现", "loaded": False, "reason": self.stub_reason}
        return ch

    def version_info(self) -> dict:
        models = {}
        for key, p in self.model_paths.items():
            if p is None:
                models[key] = {"path": None, "exists": False}
                continue
            models[key] = {
                "path": str(p),
                "exists": Path(p).exists(),
                "bytes": Path(p).stat().st_size if Path(p).exists() and Path(p).is_file() else None,
                "sha256": sha256_file(Path(p), self.hash_cache) if Path(p).is_file() else None,
            }
        fusion = getattr(self.detector, "fusion", None)
        thresholds = {}
        if fusion is not None:
            thresholds = {
                "low": float(getattr(fusion, "thr_low", 0.0)),
                "high": float(getattr(fusion, "thr_high", 0.0)),
                "disagree_review": float(getattr(fusion, "disagree_review", 0.0)),
                "weights": getattr(fusion, "weights", {}),
                "temperatures": {k: v.temperature for k, v in getattr(fusion, "scalers", {}).items()},
                "source": str(self.model_paths.get("fusion_json") or "") or None,
                "source_sha256": sha256_file(self.model_paths["fusion_json"], self.hash_cache)
                if self.model_paths.get("fusion_json") else None,
            }
        metrics_path = self.cfg.metrics_path
        metrics_summary = {"path": str(metrics_path), "exists": metrics_path.is_file()}
        if metrics_path.is_file():
            summary = read_metrics(metrics_path) or {}
            metrics_summary.update({
                "sha256": sha256_file(metrics_path, self.hash_cache),
                "n_samples": summary.get("n_samples"),
                "oof_accuracy": (summary.get("oof_overall") or {}).get("accuracy"),
                "oof_fpr": (summary.get("oof_overall") or {}).get("fpr"),
                "oof_tpr": (summary.get("oof_overall") or {}).get("tpr"),
            })
        code = code_fingerprint(self.cfg)
        return {
            "service": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "api_version": API_VERSION,
            "report_schema_version": pipeline_report_schema_version(),
            "pipeline": {"version": PIPELINE_VERSION, "mode": self.mode,
                         "device": getattr(self.detector, "device", None),
                         "time_budget_s": self.cfg.time_budget_s,
                         "stub_reason": self.stub_reason or None},
            "code": code,
            "models": models,
            "thresholds": thresholds,
            "metrics": metrics_summary,
            "runtime": {"python": sys.version.split()[0], "platform": sys.platform,
                        "executable": sys.executable},
            "generated_at": utc_now_iso(),
        }


def pipeline_report_schema_version() -> str:
    """报告 schema 版本以 src/report.py 的 REPORT_VERSION 为准（单一真源）。

    F9 教训：API 侧曾硬编码 1.0.0，与流水线实际 schema 漂移；这里改为运行时读取，
    读不到时才回退到已知版本，且 health/version 都会显式暴露该值供评审比对。
    """
    try:
        src = ROOT / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from report import REPORT_VERSION          # noqa: PLC0415
        return str(REPORT_VERSION)
    except Exception:  # noqa: BLE001
        return REPORT_SCHEMA_VERSION_FALLBACK


def load_pipeline_modules(logger):
    """导入 src/ 下的流水线模块；失败返回 None（服务退化为占位实现）。"""
    src = ROOT / "src"
    if not src.is_dir():
        logger(f"[startup][warn] 未见 {src}，使用占位实现")
        return None
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import detector as _detector          # noqa: PLC0415
        import report as _report              # noqa: PLC0415
        return {"detector": _detector, "report": _report}
    except Exception as e:  # noqa: BLE001
        logger(f"[startup][warn] 导入 src/ 失败（{type(e).__name__}: {e}），使用占位实现")
        return None


def code_fingerprint(cfg: "ServerConfig") -> dict:
    """代码版本与哈希：git commit（若有）+ 源文件内容 sha256。"""
    files: list[Path] = []
    for sub in ("api", "src", "scripts", "tests"):
        d = ROOT / sub
        if d.is_dir():
            files += [p for p in sorted(d.rglob("*"))
                      if p.is_file() and p.suffix in DOCS_SRC_EXT and "__pycache__" not in p.parts]
    for extra in ("demo/webui.html", "docs/api.md", "requirements.txt", "README.md"):
        p = ROOT / extra
        if p.is_file():
            files.append(p)
    h = hashlib.sha256()
    entries = []
    for p in files:
        digest = sha256_file(p)
        if digest is None:
            continue
        rel = p.relative_to(ROOT).as_posix()
        entries.append({"path": rel, "sha256": digest})
        h.update(rel.encode("utf-8"))
        h.update(digest.encode("ascii"))
    out = {
        "root": str(ROOT),
        "n_source_files": len(entries),
        "code_hash": h.hexdigest() if entries else None,
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    git = ROOT / ".git"
    if git.exists():
        import subprocess
        def _git(*argv):
            try:
                r = subprocess.run(["git", *argv], cwd=str(ROOT), capture_output=True,
                                   text=True, timeout=10)
                return r.stdout.strip() if r.returncode == 0 else None
            except Exception:  # noqa: BLE001
                return None
        out["git_commit"] = _git("rev-parse", "HEAD")
        out["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        status = _git("status", "--porcelain")
        out["git_dirty"] = bool(status) if status is not None else None
    return out


def read_metrics(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# 报告存储
# --------------------------------------------------------------------------- #
REPORT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ReportStore:
    def __init__(self, primary: Path, extra_dirs: list[Path], logger):
        self.primary = Path(primary)
        self.extra_dirs = [Path(p) for p in extra_dirs]
        self.log = logger
        try:
            self.primary.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.log(f"[warn] 报告目录不可写（{self.primary}）：{e}")

    def new_id(self, media_name: str = "") -> str:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def save(self, report_id: str, payload: dict) -> str | None:
        if not REPORT_ID_RE.match(report_id):
            raise ApiError("E_BAD_REQUEST", f"非法报告 id：{report_id!r}")
        path = self.primary / f"{report_id}.json"
        try:
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
            return str(path)
        except OSError as e:
            self.log(f"[warn] 报告落盘失败 {path}: {e}")
            return None

    def candidates(self, report_id: str) -> list[Path]:
        if not REPORT_ID_RE.match(report_id):
            raise ApiError("E_BAD_REQUEST", f"非法报告 id：{report_id!r}")
        out = [self.primary / f"{report_id}.json", self.primary / f"{report_id}.html"]
        for d in self.extra_dirs:
            out += [d / f"{report_id}.json", d / f"{report_id}.html"]
        return out

    def get(self, report_id: str) -> tuple[bytes, str] | None:
        for p in self.candidates(report_id):
            if p.is_file():
                ctype = ("text/html; charset=utf-8" if p.suffix == ".html"
                         else "application/json; charset=utf-8")
                return p.read_bytes(), ctype
        return None

    def list(self, limit: int = 50) -> list[dict]:
        items = []
        for p in sorted(self.primary.glob("*.json"), reverse=True):
            try:
                st = p.stat()
            except OSError:
                continue
            items.append({"report_id": p.stem, "bytes": st.st_size,
                          "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "url": f"{API_PREFIX}/report/{p.stem}"})
            if len(items) >= max(1, limit):
                break
        return items


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    no_model: bool = False
    no_warmup: bool = False
    warmup_image: str | None = None
    device: str | None = None
    time_budget_s: float = 5.0
    timeout_ms: int = 30000
    max_timeout_ms: int = 300000
    batch_timeout_ms: int = 600000
    max_batch_files: int = 200
    max_upload_mb: float = 64.0
    workers: int = 1
    max_queue: int = 32
    enable_npr: bool = False
    enable_aeroblade: bool = True
    allow_path_input: bool = True
    allow_root: list[str] = dataclasses.field(default_factory=list)
    models_dir: Path | None = None
    data_dir: Path | None = None
    reports_dir: Path = dataclasses.field(default_factory=lambda: ROOT / "reports")
    report_dir: Path = dataclasses.field(default_factory=lambda: ROOT / "reports" / "api")
    metrics_path: Path = dataclasses.field(default_factory=lambda: ROOT / "reports" / "metrics.json")
    log_path: Path | None = None
    demo_dir: Path = dataclasses.field(default_factory=lambda: ROOT / "demo")
    extra_reports_dirs: list[str] = dataclasses.field(default_factory=list)
    report_dirs: list[Path] = dataclasses.field(default_factory=list)

    @property
    def report_dirs_or_default(self) -> list[Path]:
        return self.report_dirs or [self.report_dir, self.reports_dir]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def allowed_roots(self) -> list[Path]:
        roots = [ROOT, ROOT.parent, Path.cwd(), Path(tempfile.gettempdir())]
        if self.data_dir is not None:
            roots.append(Path(self.data_dir))
        roots += [Path(p) for p in self.allow_root]
        return _uniq([r for r in roots if r.exists()])


def build_config(args: argparse.Namespace) -> ServerConfig:
    cfg = ServerConfig(
        host=args.host, port=args.port,
        no_model=bool(args.no_model), no_warmup=bool(args.no_warmup),
        warmup_image=args.warmup_image, device=args.device,
        time_budget_s=args.time_budget,
        timeout_ms=int(args.timeout_ms), max_timeout_ms=int(args.max_timeout_ms),
        batch_timeout_ms=int(args.batch_timeout_ms),
        max_batch_files=int(args.max_batch_files),
        max_upload_mb=float(args.max_upload_mb),
        workers=int(args.workers), max_queue=int(args.max_queue),
        enable_npr=bool(args.enable_npr), enable_aeroblade=not args.disable_aeroblade,
        allow_path_input=not args.no_path_input,
        allow_root=list(args.allow_root or []),
        extra_reports_dirs=list(getattr(args, "extra_reports_dir", None) or []),
    )
    cfg.models_dir = find_models_dir(args.models_dir)
    cfg.data_dir = find_data_dir(args.data_dir)
    cfg.reports_dir = find_reports_dir(args.reports_dir)
    report_dir = Path(args.report_dir) if args.report_dir else cfg.reports_dir / "api"
    cfg.report_dir = report_dir
    cfg.metrics_path = (Path(args.metrics) if args.metrics
                        else cfg.reports_dir / "metrics.json")
    cfg.log_path = (Path(args.log) if args.log else cfg.reports_dir / "api" / "requests.jsonl")
    cfg.demo_dir = Path(args.demo_dir)
    # 报告查找目录：主目录 + 仓库 reports/ + 工作区 reports/（逐样本报告所在）
    cfg.report_dirs = report_search_dirs(cfg.report_dir, cfg.extra_reports_dirs)
    # 非本机监听时，自动关闭服务端路径读取（避免把文件系统暴露给远端）
    loopback = cfg.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and args.allow_path_input is None:
        cfg.allow_path_input = False
    elif args.allow_path_input is True:
        cfg.allow_path_input = True
    return cfg


# --------------------------------------------------------------------------- #
# multipart / 输入解析
# --------------------------------------------------------------------------- #
_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;]+))', re.I)
_PARAM_RE = re.compile(r'([a-zA-Z0-9_-]+)\s*=\s*"([^"]*)"')


def parse_boundary(content_type: str) -> str | None:
    m = _BOUNDARY_RE.search(content_type or "")
    if not m:
        return None
    return (m.group(1) or m.group(2) or "").strip()


def parse_multipart(body: bytes, boundary: str) -> list[tuple[dict, bytes]]:
    """最小 multipart/form-data 解析（stdlib，兼容 cgi 移除后的 Python 3.13+）。"""
    delim = b"--" + boundary.encode("latin-1")
    parts: list[tuple[dict, bytes]] = []
    for section in body.split(delim):
        if section in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        if section.startswith(b"--"):        # 结束标记
            break
        if section.startswith(b"\r\n"):
            section = section[2:]
        elif section.startswith(b"\n"):
            section = section[1:]
        if section.endswith(b"\r\n"):
            section = section[:-2]
        elif section.endswith(b"\n"):
            section = section[:-1]
        head, sep, data = section.partition(b"\r\n\r\n")
        if not sep:
            head, sep, data = section.partition(b"\n\n")
            if not sep:
                continue
        headers: dict[str, str] = {}
        for line in head.split(b"\r\n"):
            if b":" in line:
                k, _, v = line.partition(b":")
                headers[k.decode("latin-1").strip().lower()] = v.decode("latin-1").strip()
        params = {k: v for k, v in _PARAM_RE.findall(headers.get("content-disposition", ""))}
        params["content_type"] = headers.get("content-type", "")
        parts.append((params, data))
    return parts


@dataclasses.dataclass
class Upload:
    name: str
    data: bytes
    content_type: str = "application/octet-stream"
    source: str = "bytes"        # bytes | multipart | path


def guess_kind(name: str, default: str = "image") -> str:
    return MEDIA_BY_EXT.get(Path(name).suffix.lower(), default)


SUPPORTED_UPLOAD_CTYPES = ("multipart/form-data", "application/json", "application/octet-stream",
                           "image/", "video/", "binary/octet-stream", "text/plain", "")


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
FALLBACK_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>OPEN-SZ-01 API</title><style>
body{font-family:'Microsoft YaHei',system-ui,sans-serif;margin:28px;background:#0f1420;color:#e8edf7}
code{background:#0c111b;padding:2px 6px;border-radius:4px;color:#9ecbff}
a{color:#4c8dff} li{line-height:1.9}.mut{color:#93a1bd}
</style></head><body>
<h1>OPEN-SZ-01 对外接口服务</h1>
<p class="mut">未找到 <code>demo/webui.html</code>，当前展示简易回退页（HTTP 8090 直连服务）。</p>
<ul>
<li><a href="/api/v1/health">GET /api/v1/health</a> — 证据通道加载状态</li>
<li><a href="/api/v1/version">GET /api/v1/version</a> — 模型/阈值/代码版本与哈希</li>
<li><a href="/api/v1/metrics">GET /api/v1/metrics</a> — reports/metrics.json</li>
<li><a href="/api/v1/reports">GET /api/v1/reports</a> — 已落盘报告列表</li>
<li><a href="/api/v1/openapi.json">GET /api/v1/openapi.json</a> — OpenAPI 3.0</li>
<li><code>POST /api/v1/detect/image|video|batch</code> — 上传字节或 multipart</li>
</ul>
<form id="f"><input type="file" id="file"><button>检测</button></form>
<pre id="out" class="mut">等待输入…</pre>
<script>
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
 const f=document.getElementById('file').files[0]; if(!f)return;
 const out=document.getElementById('out'); out.textContent='推理中…';
 const r=await fetch('/api/v1/detect/image?name='+encodeURIComponent(f.name),
                     {method:'POST',body:f}); out.textContent=await r.text();};
</script></body></html>"""


OPENAPI_PATHS_FOR_DOCS = [
    "/api/v1/health", "/api/v1/version", "/api/v1/metrics", "/api/v1/reports",
    "/api/v1/report/{report_id}", "/api/v1/detect/image", "/api/v1/detect/video",
    "/api/v1/detect/batch", "/api/v1/openapi.json",
]


def _err_ref() -> dict:
    return {"$ref": "#/components/schemas/ErrorEnvelope"}


def build_openapi(cfg: ServerConfig, base_url: str, mode: str) -> dict:
    """OpenAPI 3.0.3 描述（覆盖全部对外端点）。"""
    binary = {"type": "string", "format": "binary"}
    detect_params = [
        {"name": "name", "in": "query", "required": False,
         "schema": {"type": "string"},
         "description": "原始字节体模式下的文件名（决定媒体类型与后缀）"},
        {"name": "timeout_ms", "in": "query", "required": False,
         "schema": {"type": "integer", "minimum": 1, "maximum": cfg.max_timeout_ms},
         "description": "单请求超时（毫秒）；超时返回 review 降级结论并标注 degraded=true"},
        {"name": "save", "in": "query", "required": False,
         "schema": {"type": "boolean", "default": True}, "description": "是否落盘报告"},
        {"name": "html", "in": "query", "required": False,
         "schema": {"type": "boolean", "default": False},
         "description": "是否同时生成单文件 HTML 报告"},
    ]
    common_errors = {
        "400": {"description": "请求格式错误", "content": {"application/json": {"schema": _err_ref()}}},
        "404": {"description": "资源不存在", "content": {"application/json": {"schema": _err_ref()}}},
        "413": {"description": "请求体过大", "content": {"application/json": {"schema": _err_ref()}}},
        "415": {"description": "不支持的媒体类型", "content": {"application/json": {"schema": _err_ref()}}},
        "429": {"description": "推理队列已满", "content": {"application/json": {"schema": _err_ref()}}},
        "500": {"description": "服务内部错误", "content": {"application/json": {"schema": _err_ref()}}},
    }
    timeout_desc = ("单请求超时降级：超过 timeout_ms 未返回时，本端点返回 HTTP 200 + "
                    "verdict=review，并置 meta.degraded=true、meta.timeout_ms，"
                    "队列中的该单仍会跑完（结果记入日志，不回传）。")
    detect_request = {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {"type": "object",
                           "properties": {"file": binary,
                                          "file[]": {"type": "array", "items": binary}},
                           "required": ["file"]},
                "encoding": {"file": {"contentType": "image/png, image/jpeg, video/mp4"}},
            },
            "application/octet-stream": {"schema": binary},
            "image/png": {"schema": binary},
            "image/jpeg": {"schema": binary},
            "video/mp4": {"schema": binary},
            "application/json": {
                "schema": {"type": "object",
                           "properties": {"path": {"type": "string",
                                                   "description": "服务端可见路径（本机部署时可用）"},
                                          "name": {"type": "string"}},
                           "required": ["path"]},
            },
        },
    }
    batch_request = {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {"type": "object",
                           "properties": {"file": {"type": "array", "items": binary}},
                           "required": ["file"]},
            },
            "application/octet-stream": {"schema": binary},
            "application/json": {
                "schema": {"type": "object",
                           "properties": {
                               "paths": {"type": "array", "items": {"type": "string"}},
                               "dir": {"type": "string"},
                               "recursive": {"type": "boolean", "default": False}},
                           "oneOf": [{"required": ["paths"]}, {"required": ["dir"]}]},
            },
        },
    }
    resp_report = {
        "description": "完整检测报告（报告字段 + meta 元信息）",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DetectionResponse"}}},
    }
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "OPEN-SZ-01 AI 生成材料反欺诈检测服务",
            "version": SERVICE_VERSION,
            "description": (
                "L0→L1/L1b→L2（CLIP/NPR/AEROBLADE）→L3 定位→融合三档决策的服务化接口。"
                "零额外依赖实现（Python 标准库 http.server）。所有结论均为材料层面的技术判定，"
                "不构成业务欺诈定性。当前服务模式："
                f"`{mode}`（full=真实模型，stub=占位实现）。"),
            "license": {"name": "见仓库 LICENSE"},
        },
        "servers": [{"url": base_url, "description": "本实例"}],
        "tags": [{"name": "meta"}, {"name": "detect"}, {"name": "report"}, {"name": "metrics"}],
        "paths": {
            "/api/v1/health": {
                "get": {
                    "tags": ["meta"], "operationId": "getHealth", "summary": "健康检查与证据通道状态",
                    "responses": {
                        "200": {"description": "服务状态",
                                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Health"}}}},
                        **common_errors},
                }},
            "/api/v1/version": {
                "get": {
                    "tags": ["meta"], "operationId": "getVersion", "summary": "模型/阈值/代码版本与哈希",
                    "responses": {"200": {
                        "description": "版本与哈希",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Version"}}}},
                        **common_errors}},
            },
            "/api/v1/openapi.json": {
                "get": {
                    "tags": ["meta"], "operationId": "getOpenApi", "summary": "本 OpenAPI 描述",
                    "responses": {"200": {"description": "OpenAPI 3.0.3 文档",
                                          "content": {"application/json": {"schema": {"type": "object"}}}},
                                  **common_errors}},
            },
            "/api/v1/detect/image": {
                "post": {
                    "tags": ["detect"], "operationId": "detectImage", "summary": "图片检测",
                    "description": timeout_desc,
                    "parameters": detect_params, "requestBody": detect_request,
                    "responses": {"200": resp_report, **common_errors},
                }},
            "/api/v1/detect/video": {
                "post": {
                    "tags": ["detect"], "operationId": "detectVideo", "summary": "视频检测（抽帧聚合）",
                    "description": timeout_desc,
                    "parameters": detect_params, "requestBody": detect_request,
                    "responses": {"200": resp_report, **common_errors},
                }},
            "/api/v1/detect/batch": {
                "post": {
                    "tags": ["detect"], "operationId": "detectBatch",
                    "summary": "批量检测（多文件，逐条摘要 + 汇总）",
                    "description": timeout_desc + " 批量按序执行，受 batch_timeout 总预算约束。",
                    "parameters": detect_params + [
                        {"name": "dir", "in": "query", "required": False, "schema": {"type": "string"},
                         "description": "服务端目录（本机部署可用），与上传互斥"},
                        {"name": "recursive", "in": "query", "required": False,
                         "schema": {"type": "boolean", "default": False}}],
                    "requestBody": batch_request,
                    "responses": {"200": {
                        "description": "批量结果",
                        "content": {"application/json": {
                            "schema": {"$ref": "#/components/schemas/BatchResponse"}}}},
                        **common_errors},
                }},
            "/api/v1/reports": {
                "get": {
                    "tags": ["report"], "operationId": "listReports", "summary": "已落盘报告列表",
                    "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer",
                                                                               "default": 50,
                                                                               "maximum": 500}}],
                    "responses": {"200": {"description": "报告列表",
                                          "content": {"application/json": {"schema": {"type": "object"}}}},
                                  **common_errors}},
            },
            "/api/v1/report/{report_id}": {
                "get": {
                    "tags": ["report"], "operationId": "getReport", "summary": "回取已落盘报告",
                    "description": (
                        "按 report_id 回取报告。查找顺序：`<reports>/api/<id>.json` → "
                        "`<reports>/<id>.json` → 工作区 `reports/<id>.json` → 同名 `.html`。"
                        "因此 WebUI 样本库用 0xx.json 的 id（如 `001`）即可取到工作区里"
                        "流水线生成的逐样本报告。当前 report schema 版本见 "
                        "`GET /api/v1/version` 的 `report_schema_version`。"),
                    "parameters": [{"name": "report_id", "in": "path", "required": True,
                                    "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "报告 JSON（或 HTML，若只有 HTML 落盘）"},
                                  "404": {"description": "报告不存在（details.searched 列出查找过的路径）",
                                          "content": {"application/json": {"schema": _err_ref()}}},
                                  **common_errors},
                }},
            "/api/v1/metrics": {
                "get": {
                    "tags": ["metrics"], "operationId": "getMetrics", "summary": "返回 reports/metrics.json",
                    "responses": {"200": {"description": "离线指标（OOF/分折/延迟/ROC 前沿）",
                                          "content": {"application/json": {"schema": {"type": "object"}}}},
                                  "404": common_errors["404"], **common_errors}},
            },
        },
        "components": {"schemas": {
            "Error": {
                "type": "object", "required": ["code", "message"],
                "properties": {"code": {"type": "string", "enum": sorted(ERROR_CODES)},
                               "message": {"type": "string"},
                               "details": {}},
            },
            "ErrorEnvelope": {
                "type": "object", "required": ["error"],
                "properties": {"error": {"$ref": "#/components/schemas/Error"}},
            },
            "ChannelStatus": {
                "type": "object",
                "properties": {"kind": {"type": "string"}, "loaded": {"type": "boolean"},
                               "configured": {"type": "boolean"}, "available": {"type": "boolean"},
                               "path": {"type": "string", "nullable": True}, "role": {"type": "string"}},
            },
            "Health": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok", "degraded"]},
                    "service": {"type": "string"}, "api_version": {"type": "string"},
                    "mode": {"type": "string", "enum": ["full", "stub"]},
                    "uptime_s": {"type": "number"}, "started_at": {"type": "string"},
                    "device": {"type": "string", "nullable": True},
                    "channels": {"type": "object", "additionalProperties": {"$ref": "#/components/schemas/ChannelStatus"}},
                    "warmup": {"type": "object"}, "queue": {"type": "object"},
                    "requests_served": {"type": "integer"}, "stub_reason": {"type": "string", "nullable": True},
                    "models_dir": {"type": "string", "nullable": True},
                    "reports_dir": {"type": "string"},
                    "report_search_dirs": {"type": "array", "items": {"type": "string"}},
                    "report_schema_version": {"type": "string"},
                },
            },
            "Version": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"}, "service_version": {"type": "string"},
                    "api_version": {"type": "string"}, "report_schema_version": {"type": "string"},
                    "pipeline": {"type": "object"}, "code": {"type": "object"},
                    "models": {"type": "object"}, "thresholds": {"type": "object"},
                    "metrics": {"type": "object"}, "runtime": {"type": "object"},
                },
            },
            "DetectionMeta": {
                "type": "object",
                "properties": {"request_id": {"type": "string"}, "api_version": {"type": "string"},
                               "report_id": {"type": "string", "nullable": True},
                               "report_url": {"type": "string", "nullable": True},
                               "elapsed_ms": {"type": "number"}, "timeout_ms": {"type": "integer"},
                               "degraded": {"type": "boolean"},
                               "input": {"type": "object"}, "saved_path": {"type": "string", "nullable": True}},
            },
            "DetectionResponse": {
                "type": "object",
                "description": "DetectionReport（input_hash/verdict/confidence/model_scores/forensic_clues/localization/uncertainty/versions/latency_ms）+ meta",
                "properties": {
                    "input_hash": {"type": "string"}, "input_type": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["normal", "review", "suspect"]},
                    "verdict_cn": {"type": "string"}, "confidence": {"type": "number"},
                    "fused_score": {"type": "number"}, "disagreement": {"type": "number"},
                    "model_scores": {"type": "object"}, "forensic_clues": {"type": "array", "items": {"type": "object"}},
                    "localization": {"type": "object"}, "uncertainty": {"type": "array", "items": {"type": "string"}},
                    "versions": {"type": "object"}, "latency_ms": {"type": "object"},
                    "meta": {"$ref": "#/components/schemas/DetectionMeta"},
                },
            },
            "BatchItem": {
                "type": "object",
                "properties": {"filename": {"type": "string"}, "status": {"type": "string",
                                                                          "enum": ["ok", "timeout", "error", "skipped_budget"]},
                               "kind": {"type": "string"}, "verdict": {"type": "string"},
                               "verdict_cn": {"type": "string"}, "confidence": {"type": "number"},
                               "fused_score": {"type": "number"}, "degraded": {"type": "boolean"},
                               "report_id": {"type": "string", "nullable": True},
                               "report_url": {"type": "string", "nullable": True},
                               "error": {"type": "object", "nullable": True},
                               "latency_ms": {"type": "object"}},
            },
            "BatchResponse": {
                "type": "object",
                "properties": {"request_id": {"type": "string"}, "api_version": {"type": "string"},
                               "count": {"type": "integer"}, "summary": {"type": "object"},
                               "results": {"type": "array", "items": {"$ref": "#/components/schemas/BatchItem"}}},
            },
        }},
    }


def make_handler(ctx: "ServerContext"):
    _ctx = ctx          # 类体内不能对闭包名赋值（否则 LOAD_NAME 找不到闭包），故用别名

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"{SERVICE_NAME}/{SERVICE_VERSION}"
        sys_version = ""
        ctx = _ctx

        # ---------------- 基础设施 ----------------
        def log_message(self, fmt, *a):        # 覆盖默认实现：走结构化日志
            self.ctx.logger("[http] " + (fmt % a))

        def log_error(self, fmt, *a):
            self.ctx.logger("[http][error] " + (fmt % a))

        def handle_one_request(self):
            self._t0 = time.perf_counter()
            self._request_id = uuid.uuid4().hex[:12]
            return BaseHTTPRequestHandler.handle_one_request(self)

        def _respond(self, status: int, body: bytes, ctype: str,
                     extra_headers: dict | None = None, note: str = ""):
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Request-Id", self._request_id)
                self.send_header("Access-Control-Allow-Origin", "*")
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.ctx.log_request(self, status, len(body), note)

        def _json(self, status: int, payload, extra_headers: dict | None = None, note: str = ""):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._respond(status, body, "application/json; charset=utf-8",
                          extra_headers, note or "json")

        def _error(self, err: ApiError, note: str = ""):
            payload = {"error": {"code": err.code, "message": err.message}}
            if err.details is not None:
                payload["error"]["details"] = err.details
            payload["error"]["request_id"] = self._request_id
            if err.status == 413:
                self.close_connection = True
            self._json(err.status, payload, note=f"error:{err.code} {note}".strip())

        def _internal_error(self, exc: BaseException):
            self.ctx.logger(f"[error] request_id={self._request_id} "
                            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
            self._error(ApiError("E_INTERNAL", f"{type(exc).__name__}: {exc}"))

        # ---------------- 请求解析 ----------------
        def _read_body(self) -> bytes:
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in te:
                raise ApiError("E_BAD_REQUEST",
                               "暂不支持 Transfer-Encoding: chunked，请带 Content-Length")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError("E_BAD_REQUEST", "Content-Length 非法")
            if length < 0:
                raise ApiError("E_BAD_REQUEST", "Content-Length 非法")
            if length > self.ctx.cfg.max_upload_bytes:
                raise ApiError("E_PAYLOAD_TOO_LARGE",
                               f"请求体 {length} 字节 > 上限 {self.ctx.cfg.max_upload_bytes} 字节"
                               f"（--max-upload-mb {self.ctx.cfg.max_upload_mb}）")
            if length == 0:
                return b""
            return self.rfile.read(length)

        def _timeout_s(self, q: dict) -> float:
            raw = (q.get("timeout_ms") or [None])[0]
            if raw in (None, ""):
                ms = self.ctx.cfg.timeout_ms
            else:
                try:
                    ms = int(float(raw))
                except ValueError:
                    raise ApiError("E_BAD_REQUEST", f"timeout_ms 非法：{raw!r}")
            ms = max(1, min(int(ms), self.ctx.cfg.max_timeout_ms))
            return ms / 1000.0

        def _query_bool(self, q: dict, key: str, default: bool) -> bool:
            raw = (q.get(key) or [None])[0]
            if raw is None:
                return default
            return str(raw).lower() not in ("0", "false", "no", "off")

        @staticmethod
        def _wants_json(ctype: str) -> bool:
            return "application/json" in (ctype or "").lower()

        def _uploads(self, body: bytes, ctype: str, q: dict,
                     default_kind: str) -> tuple[list[Upload], dict | None]:
            """返回 (uploads, json_payload)。json_payload 非空表示走服务端路径模式。"""
            ctype_l = (ctype or "").lower()
            if ctype_l and not any(ctype_l.startswith(x) for x in SUPPORTED_UPLOAD_CTYPES if x):
                raise ApiError("E_UNSUPPORTED_MEDIA_TYPE",
                               f"不支持的 Content-Type: {ctype!r}（支持 multipart/form-data、"
                               f"application/json、application/octet-stream、image/*、video/*）")
            if ctype_l.startswith("multipart/form-data"):
                boundary = parse_boundary(ctype)
                if not boundary:
                    raise ApiError("E_BAD_REQUEST", "multipart 缺少 boundary")
                parts = parse_multipart(body, boundary)
                files = []
                for params, data in parts:
                    fname = params.get("filename")
                    if not fname:
                        continue
                    files.append(Upload(name=Path(fname).name or f"upload.{default_kind}",
                                        data=data,
                                        content_type=params.get("content_type", ""),
                                        source="multipart"))
                if not files:
                    raise ApiError("E_NO_FILE",
                                   "multipart 中未找到带 filename 的文件部件（字段名建议 file）")
                return files, None
            if self._wants_json(ctype_l):
                if not body:
                    raise ApiError("E_BAD_REQUEST", "application/json 请求体为空")
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise ApiError("E_BAD_REQUEST", f"JSON 解析失败：{e}")
                if not isinstance(payload, dict):
                    raise ApiError("E_BAD_REQUEST", "JSON 必须是对象")
                return [], payload
            # 原始字节
            if not body:
                raise ApiError("E_NO_FILE", "请求体为空：请上传文件字节（?name=xxx）或使用 multipart")
            name = (q.get("name") or [None])[0] or self.headers.get("X-Filename") \
                or f"upload.{'mp4' if default_kind == 'video' else 'png'}"
            return [Upload(name=Path(name).name, data=body, content_type=ctype_l or "application/octet-stream",
                           source="bytes")], None

        def _resolve_local_path(self, raw: str) -> Path:
            cfg = self.ctx.cfg
            if not cfg.allow_path_input:
                raise ApiError("E_FORBIDDEN_PATH",
                               "服务已禁用服务端路径输入（非本机监听或 --no-path-input），请上传字节")
            if not raw or not isinstance(raw, str):
                raise ApiError("E_BAD_REQUEST", "path 必须是非空字符串")
            p = Path(raw)
            if not p.is_absolute():
                for cand in (ROOT / p, ROOT.parent / p, Path.cwd() / p):
                    if cand.exists():
                        p = cand
                        break
                else:
                    p = ROOT / p
            try:
                p = p.resolve()
            except OSError as e:
                raise ApiError("E_BAD_REQUEST", f"路径无法解析：{e}")
            if not p.exists():
                raise ApiError("E_NOT_FOUND", f"路径不存在：{p}")
            if not p.is_file():
                raise ApiError("E_BAD_REQUEST", f"不是文件（目录请用 dir 参数）：{p}")
            roots = cfg.allowed_roots()
            if not any(is_within(p, r) for r in roots):
                raise ApiError("E_FORBIDDEN_PATH",
                               f"路径不在允许的根目录内：{p}",
                               details={"allowed_roots": [str(r) for r in roots]})
            return p

        def _expand_dir(self, raw: str, recursive: bool) -> list[Path]:
            cfg = self.ctx.cfg
            if not cfg.allow_path_input:
                raise ApiError("E_FORBIDDEN_PATH", "服务已禁用服务端目录输入")
            d = Path(raw)
            if not d.is_absolute():
                for cand in (ROOT / d, ROOT.parent / d, Path.cwd() / d):
                    if cand.is_dir():
                        d = cand
                        break
            try:
                d = d.resolve()
            except OSError as e:
                raise ApiError("E_BAD_REQUEST", f"目录无法解析：{e}")
            if not d.is_dir():
                raise ApiError("E_NOT_FOUND", f"目录不存在：{d}")
            if not any(is_within(d, r) for r in cfg.allowed_roots()):
                raise ApiError("E_FORBIDDEN_PATH", f"目录不在允许的根目录内：{d}")
            it = d.rglob("*") if recursive else d.glob("*")
            files = sorted(p for p in it
                           if p.is_file() and p.suffix.lower() in MEDIA_BY_EXT)
            if not files:
                raise ApiError("E_NOT_FOUND", f"目录内没有可检测的图片/视频：{d}")
            return files

        # ---------------- 检测执行 ----------------
        def _materialize(self, up: Upload, kind: str) -> Path:
            suffix = Path(up.name).suffix or (".mp4" if kind == "video" else ".png")
            tmp_dir = Path(self.ctx.cfg.report_dir) / "_tmp"
            try:
                tmp_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                tmp_dir = Path(tempfile.gettempdir())
            path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
            path.write_bytes(up.data)
            return path

        @staticmethod
        def _first_bytes(path: Path, n: int = 12) -> bytes:
            try:
                with open(path, "rb") as f:
                    return f.read(n)
            except OSError:
                return b""

        def _sniff_kind(self, path: Path, fallback: str) -> str:
            head = self._first_bytes(path)
            if head[4:8] == b"ftyp":            # mp4/mov
                return "video"
            if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8") or head.startswith(b"RIFF"):
                return "image"
            return fallback

        def _detect_one(self, path: Path, kind: str, timeout_s: float, *, save: bool,
                        want_html: bool, display_name: str | None = None) -> dict:
            cfg = self.ctx.cfg
            name = display_name or path.name
            t0 = time.perf_counter()
            report, degraded = self.ctx.service.run_detection(
                kind, str(path), timeout_s, self.ctx.queue,
                save_html=want_html, report_dir=Path(cfg.report_dir))
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            report.setdefault("latency_ms", {})
            report["latency_ms"]["server_total"] = elapsed
            rid = None
            saved_path = None
            meta = {
                "request_id": self._request_id,
                "api_version": API_VERSION,
                "elapsed_ms": elapsed,
                "timeout_ms": int(round(timeout_s * 1000)),
                "degraded": bool(degraded),
                "report_id": None,
                "report_url": None,
                "saved_path": None,
                "input": {"filename": name, "kind": kind},
            }
            if save:
                rid = self.ctx.store.new_id(name)
                payload = dict(report)
                payload["meta"] = {**meta, "report_id": rid,
                                   "report_url": f"{API_PREFIX}/report/{rid}"}
                saved_path = self.ctx.store.save(rid, payload)
                meta["report_id"] = rid
                meta["report_url"] = f"{API_PREFIX}/report/{rid}"
                meta["saved_path"] = saved_path
            report["meta"] = meta
            self.ctx.counters["detections"] = self.ctx.counters.get("detections", 0) + 1
            if degraded:
                self.ctx.counters["degraded"] = self.ctx.counters.get("degraded", 0) + 1
            return report

        # ---------------- 路由 ----------------
        def do_GET(self):
            try:
                self._route_get()
            except ApiError as e:
                self._error(e)
            except Exception as e:  # noqa: BLE001
                self._internal_error(e)

        def do_POST(self):
            try:
                self._route_post()
            except ApiError as e:
                self._error(e)
            except Exception as e:  # noqa: BLE001
                self._internal_error(e)

        def do_HEAD(self):
            self.do_GET()

        def do_OPTIONS(self):
            self._respond(204, b"", "text/plain; charset=utf-8",
                          {"Allow": "GET, POST, HEAD, OPTIONS",
                           "Access-Control-Allow-Headers": "Content-Type, X-Filename",
                           "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS"},
                          note="preflight")

        def _route_get(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            q = urllib.parse.parse_qs(parsed.query)
            cfg = self.ctx.cfg

            if path in ("/", "/index.html"):
                return self._serve_webui()
            if path in ("/favicon.ico",):
                return self._respond(204, b"", "image/x-icon", note="favicon")
            if path in ("/api/health", f"{API_PREFIX}/health"):
                return self._health()
            if path == f"{API_PREFIX}/version":
                return self._version()
            if path == f"{API_PREFIX}/openapi.json":
                base = f"http://{self.headers.get('Host') or f'{cfg.host}:{cfg.port}'}"
                return self._json(200, self.ctx.openapi(base), note="openapi")
            if path == f"{API_PREFIX}/metrics":
                return self._metrics()
            if path == f"{API_PREFIX}/reports":
                try:
                    limit = int((q.get("limit") or [50])[0])
                except ValueError:
                    raise ApiError("E_BAD_REQUEST", "limit 非法")
                items = self.ctx.store.list(limit=max(1, min(limit, 500)))
                return self._json(200, {"count": len(items), "reports": items,
                                        "api_version": API_VERSION}, note="reports")
            m = re.fullmatch(rf"{API_PREFIX}/report/([^/]+)", path)
            if m:
                return self._report(m.group(1))
            if path.startswith("/reports/"):
                return self._serve_static(cfg.reports_dir, path[len("/reports/"):],
                                          extra_bases=list(cfg.report_dirs_or_default))
            if path.startswith("/demo/"):
                return self._serve_static(cfg.demo_dir, path[len("/demo/"):])
            if path.startswith(f"{API_PREFIX}/detect/"):
                raise ApiError("E_METHOD_NOT_ALLOWED",
                               f"{path} 仅支持 POST（当前 GET）")
            raise ApiError("E_NOT_FOUND", f"未知端点：{path}",
                           details={"hint": f"可用端点见 {API_PREFIX}/openapi.json"})

        def _route_post(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            q = urllib.parse.parse_qs(parsed.query)
            if path in (f"{API_PREFIX}/detect/image", "/api/detect"):
                return self._detect_endpoint(q, "image")
            if path in (f"{API_PREFIX}/detect/video", "/api/detect_video"):
                return self._detect_endpoint(q, "video")
            if path == f"{API_PREFIX}/detect/batch":
                return self._detect_batch(q)
            if path in ("/api/health", f"{API_PREFIX}/health"):
                raise ApiError("E_METHOD_NOT_ALLOWED", f"{path} 仅支持 GET")
            raise ApiError("E_NOT_FOUND", f"未知端点：{path}",
                           details={"hint": f"可用端点见 {API_PREFIX}/openapi.json"})

        # ---------------- 具体端点 ----------------
        def _serve_webui(self):
            page = self.ctx.cfg.demo_dir / "webui.html"
            if page.is_file():
                return self._respond(200, page.read_bytes(), "text/html; charset=utf-8",
                                     note="webui")
            self.ctx.logger(f"[warn] 未找到 {page}，返回简易回退页")
            return self._respond(200, FALLBACK_PAGE.encode("utf-8"),
                                 "text/html; charset=utf-8", note="fallback-page")

        def _serve_static(self, base: Path, rel: str, extra_bases: list[Path] | None = None):
            """静态文件服务：主目录优先，其次报告查找目录（逐样本报告在工作区 reports/）。"""
            rel = urllib.parse.unquote(rel)
            bases = [base] + [b for b in (extra_bases or [])
                              if Path(b).resolve() != Path(base).resolve()]
            while bases:
                b = bases.pop(0)
                target = (Path(b) / rel).resolve()
                if is_within(target, b) and target.is_file():
                    ctype = {
                        ".html": "text/html; charset=utf-8",
                        ".json": "application/json; charset=utf-8",
                        ".js": "application/javascript; charset=utf-8",
                        ".css": "text/css; charset=utf-8",
                        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".svg": "image/svg+xml", ".ico": "image/x-icon", ".mp4": "video/mp4",
                        ".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8",
                    }.get(target.suffix.lower(), "application/octet-stream")
                    return self._respond(200, target.read_bytes(), ctype, note="static")
            raise ApiError("E_NOT_FOUND", f"静态资源不存在：{rel}",
                           details={"searched_dirs": [str(b) for b in [base] + (extra_bases or [])]})

        def _health(self):
            svc = self.ctx.service
            up = time.time() - self.ctx.started_ts
            channels = svc.channel_status()
            any_loaded = any(bool(v.get("loaded")) for k, v in channels.items() if not k.startswith("_"))
            status = "ok" if svc.mode == "full" and any_loaded else "degraded"
            payload = {
                "status": status,
                "service": SERVICE_NAME,
                "service_version": SERVICE_VERSION,
                "api_version": API_VERSION,
                "mode": svc.mode,
                "stub_reason": svc.stub_reason or None,
                "started_at": self.ctx.started_at,
                "uptime_s": round(up, 1),
                "device": getattr(svc.detector, "device", None),
                "models_dir": str(self.ctx.cfg.models_dir) if self.ctx.cfg.models_dir else None,
                "data_dir": str(self.ctx.cfg.data_dir) if self.ctx.cfg.data_dir else None,
                "reports_dir": str(self.ctx.cfg.reports_dir),
                "report_search_dirs": [str(p) for p in self.ctx.cfg.report_dirs_or_default],
                "report_schema_version": pipeline_report_schema_version(),
                "channels": channels,
                "warmup": svc.warmup,
                "queue": self.ctx.queue.stats(),
                "requests_served": self.ctx.counters.get("requests", 0),
                "detections": self.ctx.counters.get("detections", 0),
                "degraded_responses": self.ctx.counters.get("degraded", 0),
                "limits": {"timeout_ms": self.ctx.cfg.timeout_ms,
                           "max_timeout_ms": self.ctx.cfg.max_timeout_ms,
                           "max_upload_mb": self.ctx.cfg.max_upload_mb,
                           "max_batch_files": self.ctx.cfg.max_batch_files,
                           "allow_path_input": self.ctx.cfg.allow_path_input,
                           "time_budget_s": self.ctx.cfg.time_budget_s},
                "time": utc_now_iso(),
            }
            return self._json(200, payload, note="health")

        def _version(self):
            info = self.ctx.service.version_info()
            info["generated_at"] = utc_now_iso()
            return self._json(200, info, note="version")

        def _metrics(self):
            path = Path(self.ctx.cfg.metrics_path)
            if not path.is_file():
                raise ApiError("E_METRICS_MISSING",
                               f"未找到 {path}；可先运行 "
                               f"`python scripts/evaluate_system.py` 生成",
                               details={"expected_path": str(path)})
            data = read_metrics(path)
            if data is None:
                raise ApiError("E_INTERNAL", f"{path} 不是合法 JSON")
            return self._json(200, data, note="metrics")

        def _report(self, report_id: str):
            got = self.ctx.store.get(report_id)
            if got is None:
                raise ApiError("E_NOT_FOUND", f"报告不存在或未落盘：{report_id}",
                               details={"searched": [str(p) for p in
                                                     self.ctx.store.candidates(report_id)]})
            body, ctype = got
            return self._respond(200, body, ctype, note=f"report:{report_id}")

        def _detect_endpoint(self, q: dict, default_kind: str):
            cfg = self.ctx.cfg
            ctype = self.headers.get("Content-Type") or ""
            body = self._read_body()
            uploads, payload = self._uploads(body, ctype, q, default_kind)
            timeout_s = self._timeout_s(q)
            save = self._query_bool(q, "save", True)
            want_html = self._query_bool(q, "html", False)

            if payload is not None:              # JSON 路径模式
                if "path" not in payload:
                    raise ApiError("E_BAD_REQUEST", "JSON 模式需要 path 字段")
                src = self._resolve_local_path(payload["path"])
                kind = payload.get("kind") or self._sniff_kind(src, default_kind)
                if kind not in ("image", "video"):
                    raise ApiError("E_BAD_REQUEST", f"kind 非法：{kind!r}")
                report = self._detect_one(src, kind, timeout_s, save=save, want_html=want_html,
                                          display_name=payload.get("name") or src.name)
                return self._json(200, report, note=f"detect:{kind}")

            up = uploads[0]
            if len(uploads) > 1:
                self.ctx.logger(f"[warn] 单文件端点收到 {len(uploads)} 个文件，只处理第一个 "
                                f"（批量请用 {API_PREFIX}/detect/batch）")
            kind = self._sniff_kind_path_name(up, default_kind)
            tmp = self._materialize(up, kind)
            try:
                report = self._detect_one(tmp, kind, timeout_s, save=save, want_html=want_html,
                                          display_name=up.name)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            return self._json(200, report, note=f"detect:{kind}")

        def _sniff_kind_path_name(self, up: Upload, default_kind: str) -> str:
            ext_kind = MEDIA_BY_EXT.get(Path(up.name).suffix.lower())
            ct = (up.content_type or "").lower()
            if ct.startswith("video/"):
                return "video"
            if ct.startswith("image/"):
                return "image"
            return ext_kind or default_kind

        def _detect_batch(self, q: dict):
            cfg = self.ctx.cfg
            ctype = self.headers.get("Content-Type") or ""
            body = self._read_body()
            uploads, payload = self._uploads(body, ctype, q, "image")
            timeout_s = self._timeout_s(q)
            save = self._query_bool(q, "save", True)
            want_html = self._query_bool(q, "html", False)
            batch_budget = max(1.0, cfg.batch_timeout_ms / 1000.0)
            recursive = self._query_bool(q, "recursive", False)

            items: list[tuple[str, str, Path, bool]] = []   # (display, kind, path, tmp?)
            if payload is not None:
                payloads: list[str] = []
                if isinstance(payload.get("paths"), list):
                    payloads = [str(p) for p in payload["paths"]]
                if payload.get("dir"):
                    items += [(p.name, self._sniff_kind(p, "image"), p, False)
                              for p in self._expand_dir(str(payload["dir"]), recursive)]
                for raw in payloads:
                    p = self._resolve_local_path(raw)
                    items.append((p.name, self._sniff_kind(p, "image"), p, False))
                if not items:
                    raise ApiError("E_BAD_REQUEST", "JSON 批量模式需要 paths 数组或 dir 字段")
            else:
                for up in uploads:
                    kind = self._sniff_kind_path_name(up, "image")
                    tmp = self._materialize(up, kind)
                    items.append((up.name, kind, tmp, True))
            if q.get("dir") and payload is None and not uploads:
                items += [(p.name, self._sniff_kind(p, "image"), p, False)
                          for p in self._expand_dir(q["dir"][0], recursive)]

            if not items:
                raise ApiError("E_NO_FILE", "批量检测未收到任何文件")
            if len(items) > cfg.max_batch_files:
                raise ApiError("E_PAYLOAD_TOO_LARGE",
                               f"批量文件数 {len(items)} > 上限 {cfg.max_batch_files}"
                               f"（--max-batch-files）")

            t_start = time.perf_counter()
            results = []
            for name, kind, path, is_tmp in items:
                elapsed_total = time.perf_counter() - t_start
                remaining = batch_budget - elapsed_total
                if remaining <= 0.05:
                    results.append({"filename": name, "kind": kind, "status": "skipped_budget",
                                    "verdict": "review", "verdict_cn": "待复核",
                                    "degraded": True, "confidence": 0.0, "fused_score": 0.0,
                                    "report_id": None, "report_url": None,
                                    "error": {"code": "E_TIMEOUT_BUDGET",
                                              "message": f"批量总预算 {batch_budget:.0f}s 已用尽，"
                                                         f"剩余样本未执行"}})
                    continue
                item_timeout = max(0.001, min(timeout_s, remaining))
                try:
                    report = self._detect_one(path, kind, item_timeout, save=save,
                                              want_html=want_html, display_name=name)
                    summary = {
                        "filename": name, "kind": kind,
                        "status": "timeout" if report["meta"]["degraded"] else "ok",
                        "verdict": report.get("verdict"),
                        "verdict_cn": report.get("verdict_cn"),
                        "confidence": report.get("confidence"),
                        "fused_score": report.get("fused_score"),
                        "disagreement": report.get("disagreement"),
                        "degraded": report["meta"]["degraded"],
                        "report_id": report["meta"]["report_id"],
                        "report_url": report["meta"]["report_url"],
                        "attention": (report.get("uncertainty") or [None])[0],
                        "latency_ms": report.get("latency_ms", {}),
                        "error": None,
                    }
                except ApiError as e:
                    summary = {"filename": name, "kind": kind, "status": "error",
                               "verdict": "review", "verdict_cn": "待复核", "degraded": True,
                               "confidence": 0.0, "fused_score": 0.0, "report_id": None,
                               "report_url": None,
                               "error": {"code": e.code, "message": e.message}}
                except Exception as e:  # noqa: BLE001
                    summary = {"filename": name, "kind": kind, "status": "error",
                               "verdict": "review", "verdict_cn": "待复核", "degraded": True,
                               "confidence": 0.0, "fused_score": 0.0, "report_id": None,
                               "report_url": None,
                               "error": {"code": "E_INTERNAL",
                                         "message": f"{type(e).__name__}: {e}"}}
                finally:
                    if is_tmp:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
                results.append(summary)

            counts = {"normal": 0, "review": 0, "suspect": 0}
            for r in results:
                counts[r.get("verdict", "review")] = counts.get(r.get("verdict", "review"), 0) + 1
            confs = [float(r["confidence"]) for r in results
                     if isinstance(r.get("confidence"), (int, float)) and r.get("status") != "error"]
            summary = {
                "n": len(results),
                "verdicts": counts,
                "ok": sum(1 for r in results if r.get("status") == "ok"),
                "timeout": sum(1 for r in results if r.get("status") == "timeout"),
                "errors": sum(1 for r in results if r.get("status") == "error"),
                "skipped_budget": sum(1 for r in results if r.get("status") == "skipped_budget"),
                "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
                "max_confidence": round(max(confs), 4) if confs else None,
                "min_confidence": round(min(confs), 4) if confs else None,
                "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "reports_saved": sum(1 for r in results if r.get("report_id")),
                "source": "uploads" if uploads else "server-side",
                "mode": self.ctx.service.mode,
            }
            payload_out = {
                "request_id": self._request_id,
                "api_version": API_VERSION,
                "count": len(results),
                "summary": summary,
                "results": results,
                "meta": {"timeout_ms": int(round(timeout_s * 1000)),
                         "batch_timeout_ms": cfg.batch_timeout_ms,
                         "saved": save, "degraded": summary["timeout"] > 0 or summary["errors"] > 0},
            }
            return self._json(200, payload_out, note=f"batch:{len(results)}")

    return Handler


# --------------------------------------------------------------------------- #
# 服务上下文
# --------------------------------------------------------------------------- #
class ServerContext:
    def __init__(self, cfg: ServerConfig, service: DetectorService):
        self.cfg = cfg
        self.service = service
        self.queue = InferenceQueue(workers=cfg.workers, max_queue=cfg.max_queue)
        dirs = [p for p in cfg.report_dirs_or_default if Path(p).resolve() != Path(cfg.report_dir).resolve()]
        self.store = ReportStore(cfg.report_dir, dirs, self.logger)
        self.started_ts = time.time()
        self.started_at = utc_now_iso()
        self.counters: dict[str, int] = {}
        self._openapi_cache: dict | None = None
        self._log_lock = threading.Lock()
        self._log_file = None
        if cfg.log_path is not None:
            try:
                cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = open(cfg.log_path, "a", encoding="utf-8")
            except OSError as e:
                print(f"[warn] 请求日志文件不可写（{cfg.log_path}）：{e}", file=sys.stderr)

    def logger(self, msg: str):
        line = f"{utc_now_iso()} {msg}"
        with self._log_lock:
            print(line, file=sys.stderr, flush=True)
            if self._log_file is not None:
                try:
                    self._log_file.write(line + "\n")
                except OSError:
                    pass

    def log_request(self, handler, status: int, nbytes: int, note: str):
        self.counters["requests"] = self.counters.get("requests", 0) + 1
        try:
            elapsed_ms = round((time.perf_counter() - getattr(handler, "_t0", time.perf_counter())) * 1000, 1)
        except Exception:  # noqa: BLE001
            elapsed_ms = None
        level = "warn" if status >= 400 else "info"
        rec = {
            "ts": utc_now_iso(),
            "level": level,
            "request_id": getattr(handler, "_request_id", "-"),
            "remote": handler.client_address[0] if handler.client_address else "-",
            "method": handler.command,
            "path": handler.path,
            "status": status,
            "bytes": nbytes,
            "ms": elapsed_ms,
            "note": note,
            "mode": self.service.mode,
        }
        with self._log_lock:
            print(f"[api] {rec['method']} {rec['path']} -> {status} "
                  f"({nbytes}B, {elapsed_ms}ms) req={rec['request_id']} {note}",
                  file=sys.stderr, flush=True)
            if self._log_file is not None:
                try:
                    self._log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._log_file.flush()
                except OSError:
                    pass

    def openapi(self, base_url: str) -> dict:
        if self._openapi_cache is None:
            self._openapi_cache = build_openapi(self.cfg, base_url, self.service.mode)
            self._openapi_cache["x-endpoint-base"] = base_url
        return self._openapi_cache


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def build_service(cfg: ServerConfig, logger) -> DetectorService:
    svc = DetectorService(cfg, logger)
    return svc


def serve(cfg: ServerConfig, *, service: DetectorService | None = None,
          ready_callback=None) -> ThreadingHTTPServer:
    """构建服务（含模型常驻 + 启动预热）并返回已绑定端口的 HTTP server。

    返回的 server 尚未 serve_forever()，便于自测在后台线程里跑。
    """
    holder: dict = {}

    def _logger(msg):
        ctx_ = holder.get("ctx")
        if ctx_ is not None:
            ctx_.logger(msg)
        else:
            print(f"{utc_now_iso()} {msg}", file=sys.stderr, flush=True)

    svc = service if service is not None else build_service(cfg, _logger)
    ctx = ServerContext(cfg, svc)
    holder["ctx"] = ctx
    handler_cls = make_handler(ctx)

    ctx.logger(f"[startup] {SERVICE_NAME} v{SERVICE_VERSION} api={API_VERSION} "
               f"mode={svc.mode} host={cfg.host}:{cfg.port} workers={cfg.workers}")
    if svc.mode == "stub":
        ctx.logger(f"[startup][warn] 占位实现：{svc.stub_reason}")
    ctx.logger(f"[startup] models_dir={cfg.models_dir} data_dir={cfg.data_dir} "
               f"reports={cfg.reports_dir} log={cfg.log_path}")
    svc.warm_up()
    ctx.queue.start()

    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler_cls)
    httpd.daemon_threads = True
    if ready_callback is not None:
        ready_callback(httpd, ctx)
    return httpd


def run_selftest(args: argparse.Namespace) -> int:
    """启动服务（本机随机端口）并跑通 tests/test_api.py 的端点自测。"""
    cfg = build_config(args)
    cfg.host = "127.0.0.1"
    cfg.port = 0
    cfg.allow_path_input = True          # 自测需要校验 JSON 路径模式
    print("[selftest] 构建服务…", flush=True)
    holder: dict = {}
    httpd = serve(cfg, ready_callback=lambda h, c: holder.update(httpd=h, ctx=c))
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    print(f"[selftest] 已启动 {base} mode={holder['ctx'].service.mode}", flush=True)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                              daemon=True)
    thread.start()
    try:
        spec = importlib.util.spec_from_file_location("open_sz_test_api",
                                                      ROOT / "tests" / "test_api.py")
        if spec is None or spec.loader is None:
            print("[selftest][error] 无法加载 tests/test_api.py", flush=True)
            return 2
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, mod)   # dataclasses 的 _is_type 依赖 sys.modules
        spec.loader.exec_module(mod)
        result = mod.run_suite(base_url=base, full_mode=holder["ctx"].service.mode == "full",
                               verbose=True)
        print("\n[selftest] " + json.dumps(result["summary"], ensure_ascii=False))
        return 0 if result["ok"] else 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="OPEN-SZ-01 版本化 REST 检测服务（stdlib，无额外依赖）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-model", action="store_true",
                    help="不加载真实模型，用占位实现（端点自测/CI 用）")
    ap.add_argument("--no-warmup", action="store_true", help="跳过启动预热")
    ap.add_argument("--warmup-image", default=None, help="预热样本（默认 data/images/ 首张）")
    ap.add_argument("--device", default=None, help="cpu / cuda（默认自动）")
    ap.add_argument("--time-budget", type=float, default=5.0,
                    help="流水线内部时间预算（秒），超预算的 L2 通道自动跳过")
    ap.add_argument("--timeout-ms", type=int, default=30000,
                    help="单请求默认超时（毫秒），超时返回 review 降级结论")
    ap.add_argument("--max-timeout-ms", type=int, default=300000)
    ap.add_argument("--batch-timeout-ms", type=int, default=600000)
    ap.add_argument("--max-batch-files", type=int, default=200)
    ap.add_argument("--max-upload-mb", type=float, default=64.0)
    ap.add_argument("--workers", type=int, default=1, help="推理工作线程数（GPU 建议 1）")
    ap.add_argument("--max-queue", type=int, default=32)
    ap.add_argument("--enable-npr", action="store_true", help="启用 NPR 通道（默认消融）")
    ap.add_argument("--disable-aeroblade", action="store_true")
    ap.add_argument("--no-path-input", action="store_true",
                    help="禁用 JSON path/dir 服务端路径输入")
    ap.add_argument("--allow-path-input", dest="allow_path_input", action="store_true",
                    default=None, help="即使非本机监听也允许路径输入")
    ap.add_argument("--allow-root", action="append", default=[],
                    help="额外允许的根目录（可重复）")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--report-dir", default=None, help="API 落盘报告目录（默认 <reports>/api）")
    ap.add_argument("--extra-reports-dir", action="append", default=[],
                    help="额外报告查找目录（可重复）；默认已包含仓库与工作区的 reports/，"
                         "用于回取逐样本报告 0xx.json")
    ap.add_argument("--metrics", default=None, help="metrics.json 路径")
    ap.add_argument("--log", default=None, help="请求日志 JSONL 路径")
    ap.add_argument("--demo-dir", default=str(ROOT / "demo"))
    ap.add_argument("--selftest", action="store_true", help="启动并跑通端点自测后退出")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.selftest:
        return run_selftest(args)
    cfg = build_config(args)
    httpd = serve(cfg)
    ctx = httpd.RequestHandlerClass.ctx
    try:
        print(f"[api] http://{cfg.host}:{cfg.port}/  (Ctrl+C 停止) "
              f"OpenAPI: http://{cfg.host}:{cfg.port}{API_PREFIX}/openapi.json", flush=True)
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n[api] 收到中断，正在停止…", flush=True)
    finally:
        try:
            ctx.logger("[shutdown] 服务停止")
        except Exception:  # noqa: BLE001
            pass
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
