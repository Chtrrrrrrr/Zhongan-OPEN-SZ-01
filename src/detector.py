"""End-to-end detection pipeline: L0 -> L1 -> L2 -> L3 -> fusion -> report.

All members are evidence producers, not serial gates: if one fails or times out,
the remaining evidence still forms a conclusion (manual ch.4.2, 16.3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fusion import FusionEngine, ModelScore
from l0_input import parse_image, ParsedInput
from l1_forensics import extract_l1_features
from report import DetectionReport, heatmap_to_b64_png


@dataclass
class PipelineConfig:
    clip_dir: str | None = None          # local snapshot of openai/clip-vit-large-patch14
    clip_probe: str | None = None        # trained linear probe .pt
    npr_weights: str | None = None       # trained NPR .pt
    vae_dir: str | None = None           # SD VAE for AEROBLADE
    fusion_json: str | None = None       # fitted FusionEngine
    l1_lr_model: str | None = None       # joblib-dumped {scaler, clf} for L1 score
    enable_aeroblade: bool = True
    time_budget_s: float = 5.0
    device: str | None = None


class AIGCDetector:
    """Unified image/video detection service (models kept resident & warm)."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self._clip = None
        self._clip_head = None
        self._npr = None
        self._blade = None
        self._l1_pack = None
        self.fusion = FusionEngine.load(cfg.fusion_json) if cfg.fusion_json else FusionEngine()

    # ---- lazy, resident model loading ----
    @property
    def clip(self):
        if self._clip is None and self.cfg.clip_dir:
            from l2_clip import CLIPFeatureExtractor
            self._clip = CLIPFeatureExtractor(self.cfg.clip_dir, self.device)
            if self.cfg.clip_probe:
                import torch
                from l2_clip import LinearProbe
                sd = torch.load(self.cfg.clip_probe, map_location=self.device)
                head = LinearProbe(sd["weight"].shape[1]).to(self.device).eval()
                head.load_state_dict(sd)
                self._clip_head = head
        return self._clip

    @property
    def npr(self):
        if self._npr is None and self.cfg.npr_weights:
            from l2_npr import NPRFeatureExtractor
            self._npr = NPRFeatureExtractor(self.cfg.npr_weights, self.device)
        return self._npr

    @property
    def blade(self):
        if self._blade is None and self.cfg.vae_dir and self.cfg.enable_aeroblade:
            try:
                from l2_aeroblade import AerobladeScorer
                self._blade = AerobladeScorer(self.cfg.vae_dir, self.device)
            except Exception as e:
                print(f"[degrade] AEROBLADE unavailable: {e}")
        return self._blade

    @property
    def l1_pack(self):
        if self._l1_pack is None and self.cfg.l1_lr_model:
            import joblib
            self._l1_pack = joblib.load(self.cfg.l1_lr_model)
        return self._l1_pack

    # ---- forensic clue list (L1 -> human-readable) ----
    def _l1_clues(self, feat: dict, parsed: ParsedInput) -> list:
        clues = []
        clues.append({"name": "PNG容器指纹", "status": "命中" if feat["nidat"] > 200 else "未命中",
                      "detail": f"IDAT块数={int(feat['nidat'])}, sRGB={int(feat['hasSrgb'])}, "
                                f"eXIf={int(feat['hasExif'])}, iTXt={int(feat['hasItxt'])}, "
                                f"gAMA={int(feat['hasGama'])}"})
        clues.append({"name": "EXIF/XMP元数据", "status": "命中" if (feat["hasExif"] or feat["hasItxt"]) else "未命中",
                      "detail": "真实拍摄件常携带设备/软件元数据；AI渲染件通常缺失"})
        clues.append({"name": "8x8块效应", "status": "命中" if feat["blockRatio"] > 1.05 else "未命中",
                      "detail": f"blockRatio={feat['blockRatio']:.3f}（>1表示存在压缩网格痕迹）"})
        clues.append({"name": "残差自相关", "status": "命中" if abs(feat["ac2"]) > 0.15 else "未命中",
                      "detail": f"ac2={feat['ac2']:.3f}（上采样伪影会留下周期性相关）"})
        clues.append({"name": "多尺度残差", "status": "命中" if feat["msResRatio"] < 1.3 else "未命中",
                      "detail": f"msResRatio={feat['msResRatio']:.3f}"})
        return clues

    # ---- image path ----
    def detect_image(self, path: str, want_report: bool = True,
                     report_dir: str | None = None) -> tuple[DetectionReport, float]:
        t0 = time.perf_counter()
        latency = {}
        rep = DetectionReport(input_type="image")

        parsed = parse_image(path)
        if not parsed.ok:
            rep.verdict = "review"
            rep.uncertainty.append(f"输入无法解码：{parsed.error}（转人工复核）")
            return rep.finalize(), 0.5
        rep.input_hash = parsed.sha256
        rep.resolution = f"{parsed.orig_width}x{parsed.orig_height}"
        latency["L0"] = (time.perf_counter() - t0) * 1000

        # L1
        t = time.perf_counter()
        feat = extract_l1_features(parsed)
        rep.forensic_clues = self._l1_clues(feat, parsed)
        latency["L1"] = (time.perf_counter() - t) * 1000

        scores: list[ModelScore] = []
        # L1 logistic score (if trained)
        if self.l1_pack is not None:
            from l1_forensics import feature_vector
            X = feature_vector(feat).reshape(1, -1)
            Xs = self.l1_pack["scaler"].transform(X)
            p = float(self.l1_pack["clf"].predict_proba(Xs)[0, 1])
            logit = float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
            scores.append(ModelScore("l1_forensics", logit))

        # L2 members with per-member timeout guard
        budget = self.cfg.time_budget_s

        def _elapsed():
            return time.perf_counter() - t0

        if self.clip is not None and _elapsed() < budget - 1.0:
            t = time.perf_counter()
            import torch
            featv = torch.from_numpy(self.clip(parsed.bgr)).float().to(self.device)
            with torch.no_grad():
                logit = float(self._clip_head(featv.unsqueeze(0)).item())
            scores.append(ModelScore("clip_probe", logit))
            latency["L2_clip"] = (time.perf_counter() - t) * 1000

        if self.npr is not None and _elapsed() < budget - 0.8:
            t = time.perf_counter()
            scores.append(ModelScore("npr", self.npr.logit(parsed.bgr)))
            latency["L2_npr"] = (time.perf_counter() - t) * 1000

        if self.blade is not None and _elapsed() < budget - 0.5:
            t = time.perf_counter()
            err = self.blade.recon_error(parsed.bgr)
            # AEROBLADE: low error -> AI. Convert to a raw score with direction=-1
            scores.append(ModelScore("aeroblade", err, direction=-1))
            latency["L2_aeroblade"] = (time.perf_counter() - t) * 1000

        # L3 localization (only if needed: suspect/review path or always for demo)
        t = time.perf_counter()
        try:
            from l3_localization import combined_anomaly_map, top_anomaly_boxes
            amap = combined_anomaly_map(parsed.bgr)
            boxes = top_anomaly_boxes(amap)
            rep.localization = {
                "kind": "zero-weight forensic map (ELA+noise inconsistency)",
                "heatmap_png_b64": heatmap_to_b64_png(amap, parsed.bgr),
                "boxes": boxes,
            }
        except Exception as e:
            rep.uncertainty.append(f"L3定位分支不可用：{e}")
        latency["L3"] = (time.perf_counter() - t) * 1000

        # fusion & decision
        t = time.perf_counter()
        dec = self.fusion.fuse(scores)
        rep.verdict = dec.verdict
        rep.confidence = round(dec.confidence, 4)
        rep.fused_score = round(dec.fused_score, 4)
        rep.model_scores = dec.model_scores
        rep.disagreement = dec.disagreement
        rep.uncertainty.extend(dec.reasons)
        latency["fusion"] = (time.perf_counter() - t) * 1000
        latency["total"] = (time.perf_counter() - t0) * 1000
        rep.latency_ms = {k: round(v, 1) for k, v in latency.items()}
        rep.versions = {
            "pipeline": "open-sz-01 v0.1",
            "models": "+".join(rep.model_scores.keys()) or "none",
            "thresholds": f"low={self.fusion.thr_low:.3f}/high={self.fusion.thr_high:.3f}",
            "device": self.device,
        }
        rep.finalize()
        return rep, dec.confidence

    # ---- video path ----
    def detect_video(self, path: str) -> DetectionReport:
        import video as vid
        t0 = time.perf_counter()
        rep = DetectionReport(input_type="video")
        try:
            frames, fps, total = vid.extract_frames(path)
        except Exception as e:
            rep.verdict = "review"
            rep.uncertainty.append(f"视频抽帧失败：{e}（转人工复核）")
            return rep.finalize()
        rep.resolution = f"{frames[0][1].shape[1]}x{frames[0][1].shape[0]}" if frames else ""
        rep.duration_s = round(total / fps, 2) if fps else None

        frame_scores = []
        for idx, fr in frames:
            ts = idx / fps if fps else 0.0
            # reuse image evidence per frame (L2 members only, keep it fast)
            s: list[ModelScore] = []
            if self.clip is not None:
                import torch
                featv = torch.from_numpy(self.clip(fr)).float().to(self.device)
                with torch.no_grad():
                    s.append(ModelScore("clip_probe", float(self._clip_head(featv.unsqueeze(0)).item())))
            if self.npr is not None:
                s.append(ModelScore("npr", self.npr.logit(fr)))
            if not s:
                rep.uncertainty.append("无可用L2模型，视频无法评分")
                continue
            dec = self.fusion.fuse(s)
            frame_scores.append(vid.VideoFrameScore(idx, ts, dec.confidence))

        agg = vid.aggregate([fs.score for fs in frame_scores]) if frame_scores else None
        if agg:
            final = agg["mean"] * 0.5 + agg["p90"] * 0.5   # mean/p90 fusion (Stage-1 rule)
            rep.fused_score = round(final, 4)
            rep.confidence = round(final, 4)
            rep.verdict = ("suspect" if final >= self.fusion.thr_high
                           else "normal" if final <= self.fusion.thr_low else "review")
            rep.model_scores = {f"frame_{k}": round(v, 4) for k, v in agg.items()}
            rep.localization = {"kind": "temporal segments",
                                "suspicious_segments": vid.suspicious_ranges(
                                    frame_scores, self.fusion.thr_high)}
        rep.latency_ms = {"total": round((time.perf_counter() - t0) * 1000, 1)}
        rep.versions = {"pipeline": "open-sz-01 v0.1", "channel": "video-stage1",
                        "frames": str(len(frame_scores))}
        return rep.finalize()
