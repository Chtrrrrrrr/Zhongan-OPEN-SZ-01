"""End-to-end detection pipeline: L0 -> L1 -> L2 -> L3 -> fusion -> report.

All members are evidence producers, not serial gates: if one fails or times out,
the remaining evidence still forms a conclusion (manual ch.4.2, 16.3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fusion import FusionEngine, ModelScore
from l0_input import parse_image, ParsedInput
from l1_forensics import extract_l1_features
from report import (DetectionReport, heatmap_to_b64_png, model_hash_table,
                    sha256_file, thresholds_sha256)


@dataclass
class PipelineConfig:
    clip_dir: str | None = None          # local snapshot of openai/clip-vit-large-patch14
    clip_probe: str | None = None        # trained linear probe .pt
    npr_weights: str | None = None       # trained NPR .pt
    vae_dir: str | None = None           # SD VAE for AEROBLADE
    fusion_json: str | None = None       # fitted FusionEngine
    l1_lr_model: str | None = None       # joblib-dumped {scaler, clf} for L1 score
    clip_domain_lr: str | None = None    # joblib-dumped {scaler, clf} on CLIP features
    combined_lr: str | None = None       # joblib-dumped {scaler, clf} on [L1 (+) L1b (+) CLIP (+ tiles)]
    forensic_lr: str | None = None       # joblib-dumped {scaler, clf} on [L1 (+) L1b]
    enable_aeroblade: bool = False       # AEROBLADE was REMOVED by its ablation
                                         # (reports/aeroblade_ablation.json: AUC 0.51,
                                         # ΔTPR@FPR=5% below the gate) - off by default,
                                         # and it only loads when the frozen fusion table
                                         # actually carries a weight for it (manual ch.12.1)
    feature_spec: str | None = None      # models/feature_spec.json: block layout the heads expect
    time_budget_s: float = 5.0
    device: str | None = None


# The decision head an attention explanation would have to explain, in one string so
# every report section (available or not) describes the same head.
ATTENTION_HEAD_DESC = ("combined (joblib 逻辑回归 on [L1 | L1b | CLIP global | CLIP tiles])")


class AIGCDetector:
    """Unified image/video detection service (models kept resident & warm)."""
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self._clip = None
        self._clip_head = None
        self._npr = None
        self._blade = None
        self._blade_skipped = False      # AEROBLADE gate: report the reason only once
        self._l1_pack = None
        self._clip_domain_pack = None
        self._combined_pack = None
        self._forensic_pack = None
        # One-time setup cost (weights + joblib loads). It must not be charged to the
        # per-item time budget, otherwise the very first item of a cold process (CLI,
        # first demo/API request) silently drops CLIP/tiles/explanation and reports a
        # degraded result for a perfectly healthy model.
        self._load_s = 0.0
        self.fusion = FusionEngine.load(cfg.fusion_json) if cfg.fusion_json else FusionEngine()
        # The block layout the heads were fitted on. Read from a frozen artifact so the
        # online path cannot drift from the offline extractor (src/feature_spec.py).
        if cfg.feature_spec:
            from feature_spec import FeatureSpec
            self.spec = FeatureSpec.load(cfg.feature_spec)
        else:
            from feature_spec import FeatureSpec
            default = Path(__file__).resolve().parent.parent / "models" / "feature_spec.json"
            self.spec = FeatureSpec.load(default) if default.exists() else FeatureSpec()

    # ---- lazy, resident model loading ----
    @property
    def clip(self):
        if self._clip is None and self.cfg.clip_dir:
            t_load = time.perf_counter()
            from l2_clip import CLIPFeatureExtractor
            self._clip = CLIPFeatureExtractor(self.cfg.clip_dir, self.device)
            if self.cfg.clip_probe:
                import torch
                from l2_clip import LinearProbe
                ck = torch.load(self.cfg.clip_probe, map_location=self.device,
                                weights_only=False)
                # accept both {"state_dict": {...}} bundles and flat state dicts,
                # and both fc.weight/fc.bias and weight/bias key spellings
                sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
                norm = {("weight" if k == "fc.weight" else "bias" if k == "fc.bias" else k): v
                        for k, v in sd.items()}
                head = LinearProbe(int(norm["weight"].shape[1])).to(self.device).eval()
                head.load_state_dict({"fc.weight": norm["weight"], "fc.bias": norm["bias"]})
                self._clip_head = head
            self._load_s += time.perf_counter() - t_load
        return self._clip

    @property
    def npr(self):
        if self._npr is None and self.cfg.npr_weights:
            t_load = time.perf_counter()
            from l2_npr import NPRFeatureExtractor
            self._npr = NPRFeatureExtractor(self.cfg.npr_weights, self.device)
            self._load_s += time.perf_counter() - t_load
        return self._npr

    @property
    def blade(self):
        """AEROBLADE (optional, training-free VAE reconstruction error).

        Off by default. The member's ablation (repo/reports/aeroblade_ablation.json)
        measured a single-route AUC of ~0.51 and a non-positive Δ when added to the
        deployed fusion, so per manual ch.12.1 it is REMOVED from the default
        pipeline: an optional member only enters when it has passed the ablation and
        therefore has a frozen weight in models/fusion.json. Gating on the weight
        also avoids paying its cost (frozen SD-VAE: ~11 s load, ~0.6 s per image)
        for evidence the decision layer would discard anyway.
        """
        if self._blade is None and self.cfg.vae_dir and self.cfg.enable_aeroblade:
            if self.fusion.weights.get("aeroblade", 0.0) <= 0:
                if not self._blade_skipped:
                    print("[degrade] AEROBLADE enabled but has no frozen fusion weight "
                          "(see repo/reports/aeroblade_ablation.json) -> not loaded")
                    self._blade_skipped = True
                return None
            t_load = time.perf_counter()
            try:
                from l2_aeroblade import AerobladeScorer
                self._blade = AerobladeScorer(self.cfg.vae_dir, self.device)
            except Exception as e:
                print(f"[degrade] AEROBLADE unavailable: {e}")
            self._load_s += time.perf_counter() - t_load
        return self._blade

    @property
    def l1_pack(self):
        if self._l1_pack is None and self.cfg.l1_lr_model:
            import joblib
            self._l1_pack = joblib.load(self.cfg.l1_lr_model)
        return self._l1_pack

    @property
    def clip_domain_pack(self):
        if self._clip_domain_pack is None and self.cfg.clip_domain_lr:
            import joblib
            self._clip_domain_pack = joblib.load(self.cfg.clip_domain_lr)
        return self._clip_domain_pack

    @property
    def combined_pack(self):
        if self._combined_pack is None and self.cfg.combined_lr:
            import joblib
            self._combined_pack = joblib.load(self.cfg.combined_lr)
        return self._combined_pack

    @property
    def forensic_pack(self):
        if self._forensic_pack is None and self.cfg.forensic_lr:
            import joblib
            self._forensic_pack = joblib.load(self.cfg.forensic_lr)
        return self._forensic_pack

    @staticmethod
    def _pack_logit(pack, x: "np.ndarray") -> float:
        """Apply a {scaler, clf} bundle -> logit of the positive class."""
        xs = pack["scaler"].transform(x.reshape(1, -1))
        p = float(pack["clf"].predict_proba(xs)[0, 1])
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    def _pack_logit_clip_head(self, clip_feat: "np.ndarray") -> float:
        """Frozen external linear probe applied to an already-computed CLIP feature."""
        if self._clip_head is None:
            return 0.0
        import torch
        v = torch.from_numpy(np.asarray(clip_feat)).float().to(self.device)
        with torch.no_grad():
            return float(self._clip_head(v.unsqueeze(0)).item())

    # ---- traceability: hashes of the frozen artifacts behind this output ----
    def artifact_paths(self) -> dict:
        """Configured weights/threshold files whose hashes make a run reproducible."""
        return {
            "clip_probe": self.cfg.clip_probe,
            "npr": self.cfg.npr_weights,
            "l1_lr": self.cfg.l1_lr_model,
            "forensic_lr": self.cfg.forensic_lr,
            "clip_domain_lr": self.cfg.clip_domain_lr,
            "combined_lr": self.cfg.combined_lr,
            "fusion_json": self.cfg.fusion_json,
        }

    def _traceability(self, rep: DetectionReport, extra: dict | None = None) -> dict:
        """Report `versions` block: versions + weight/threshold/input SHA-256.

        Manual ch.9.2/16.x asks for a verifiable chain from output back to the exact
        frozen artifacts. The input hash is already carried by the report; the model
        and threshold hashes are added here (joblib/pt/fusion.json are hashed on
        disk, fusion parameters are additionally hashed as a canonical value set).
        """
        v = {
            "pipeline": "open-sz-01 v0.1",
            "models": "+".join(rep.model_scores.keys()) or "none",
            "thresholds": f"low={self.fusion.thr_low:.3f}/high={self.fusion.thr_high:.3f}",
            "thresholds_sha256": thresholds_sha256(self.fusion),
            "device": self.device,
            "input_sha256": rep.input_hash or "不可用：该通道未计算输入哈希",
            "artifact_sha256": model_hash_table(self.artifact_paths()),
        }
        if extra:
            v.update(extra)
        return v

    # ---- classifier attention explanation (manual ch.9.2 "Grad-CAM (if available)") ----
    @staticmethod
    def _unavailable_explanation(reason: str, *, method: str | None = None,
                                 gradcam: dict | None = None,
                                 disclaimer: str | None = None) -> dict:
        """A complete, non-blank explanation block for a case where no map exists."""
        from l3_localization import (ATTRIBUTION_DISCLAIMER, ATTRIBUTION_KIND,
                                     ATTRIBUTION_LABEL_CN, ATTRIBUTION_METHOD)
        meth = method or ATTRIBUTION_METHOD
        return {
            "kind": ATTRIBUTION_KIND,
            "label_cn": ATTRIBUTION_LABEL_CN,
            "available": False,
            "status": "不可用",
            "method": meth,
            "head": ATTENTION_HEAD_DESC,
            "heatmap_png_b64": None,
            "tiles": [],
            "reason": reason,
            "disclaimer": disclaimer or ATTRIBUTION_DISCLAIMER,
            "gradcam": gradcam if gradcam is not None else {
                "available": False, "status": "不可用", "reason": reason,
                "substitute": meth},
        }

    def _attention_explanation(self, rep: DetectionReport, parsed: ParsedInput, x1, xb,
                               clip_feat, tile_feat, combined_logit, patch_feat=None,
                               tile_detail=None) -> dict:
        """Attention map of the decision head - EXPLANATION, never localization.

        The deployed `combined` head is a joblib logistic regression: no feature map
        and no backward pass, so Grad-CAM cannot be computed for it. The usable
        substitute is a tile-level leave-one-out contribution attribution over the
        very CLIP tile features that head consumes. If neither path can produce a
        figure, the field is explicitly reported as 不可用 with a reason - a blank
        field is never left behind (manual ch.9.2).
        """
        from l3_localization import (ATTRIBUTION_METHOD, gradcam_supported,
                                     tile_contribution_attribution)
        gradcam_ok, gradcam_reason = gradcam_supported(self.combined_pack)
        gradcam = {"available": bool(gradcam_ok),
                   "status": "可用" if gradcam_ok else "不可用",
                   "reason": gradcam_reason,
                   "substitute": ATTRIBUTION_METHOD}
        exp = self._unavailable_explanation("关注区域解释未生成", gradcam=gradcam)
        exp["reason"] = None
        if not gradcam_ok:
            rep.uncertainty.append(
                f"Grad-CAM 不可用（{gradcam_reason}）；关注区域解释改由块级贡献归因提供")
        if self.combined_pack is None:
            exp["reason"] = "combined 头未配置或加载失败，无关注区域可解释"
            return exp
        if self.clip is None or clip_feat is None:
            exp["reason"] = "CLIP 骨干不可用，无法计算分块特征"
            return exp
        if tile_feat is None:
            exp["reason"] = "分块CLIP特征不可用，combined 头缺少块特征输入"
            return exp
        try:
            context = self.spec.forensic_vector(x1, xb, patch_feat)
            bgr_full = parsed.bgr_full if parsed.bgr_full is not None else parsed.bgr
            res = tile_contribution_attribution(
                self.clip, bgr_full, np.concatenate([context, clip_feat]), tile_feat,
                lambda v: self._pack_logit(self.combined_pack, v),
                max_side=max(int(sc.get("max_side") or 1600) for sc in self.spec.scales),
                spec=self.spec, detail=tile_detail)
        except Exception as e:
            exp["reason"] = f"块级贡献归因计算失败：{e}"
            return exp
        if not res.get("ok"):
            exp["reason"] = res.get("reason") or "块级贡献归因不可用"
            return exp
        delta = None
        if combined_logit is not None:
            delta = abs(float(res["full_logit"]) - float(combined_logit))
        exp.update({
            "available": True,
            "status": "可用",
            "reason": None,
            "heatmap_png_b64": heatmap_to_b64_png(res["heatmap"], parsed.bgr),
            "tiles": res["tiles"],
            "geometry": res["geometry"],
            "delta_logit_max": res["delta_max"],
            "delta_logit_min": res["delta_min"],
            "score_consistency_delta": None if delta is None else round(delta, 9),
        })
        if delta is not None and delta > 1e-6:
            rep.uncertainty.append(
                f"块级归因基准分数与 combined 头分数不一致（Δ={delta:.2e}），归因图仅供参考")
        if res["flat_positive"]:
            exp["note"] = ("所有块的留一贡献均 ≤ 0：分类器在粒度上未表现出正向依赖，"
                           "该图仅示意权重分布，不代表关注区域")
        return exp

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
        load_before = self._load_s
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
        from l1_forensics import feature_vector
        from l1b_domain import extract_l1b_features, feature_vector_b
        x1 = feature_vector(feat)
        # L1b is its own budget line: it was previously invisible in the per-stage
        # latency breakdown (its cost only showed up in `total`), which breaks the
        # manual ch.16.2 requirement to break the budget down per stage.
        t = time.perf_counter()
        xb = feature_vector_b(extract_l1b_features(parsed))
        latency["L1b"] = (time.perf_counter() - t) * 1000

        # Optional CLIP patch-token block (feature spec decides whether the heads want it).
        # Computed from the same frozen forward pass that produces the global feature, so
        # it adds no extra backbone call.
        patch_feat = None
        if self.spec.patch_tokens and self.clip is not None:
            t = time.perf_counter()
            try:
                from l3_localization import patch_token_features
                bgr_full = parsed.bgr_full if parsed.bgr_full is not None else parsed.bgr
                patch_feat = patch_token_features(self.clip, bgr_full,
                                                  max_side=self.spec.patch_max_side)
                latency["L2_patch"] = (time.perf_counter() - t) * 1000
            except Exception as e:
                rep.uncertainty.append(f"分块标记统计不可用（降级运行）：{e}")

        # domain forensics (L1 + L1b): always available, no deep model needed
        if self.forensic_pack is not None:
            try:
                scores.append(ModelScore("forensic", self._pack_logit(
                    self.forensic_pack, self.spec.forensic_vector(x1, xb, patch_feat))))
            except Exception as e:
                rep.uncertainty.append(f"forensic 头不可用（降级运行）：{e}")

        # L2 members with per-member timeout guard
        budget = self.cfg.time_budget_s

        def _elapsed():
            # per-item work time only; model loading that happened inside this call is
            # one-time setup, so it must not eat the per-item time budget
            return time.perf_counter() - t0 - (self._load_s - load_before)

        clip_feat = None
        tile_feat = None
        tile_detail = None
        combined_logit = None
        if self.clip is not None and _elapsed() < budget - 1.5:
            t = time.perf_counter()
            import torch
            clip_feat = self.clip(parsed.bgr)
            featv = torch.from_numpy(clip_feat).float().to(self.device)
            with torch.no_grad():
                logit = float(self._clip_head(featv.unsqueeze(0)).item())
            scores.append(ModelScore("clip_external", logit))
            latency["L2_clip"] = (time.perf_counter() - t) * 1000

            # native-resolution tiled CLIP evidence (scales come from the feature spec).
            # The per-tile features are kept so the attention explanation reuses this
            # pass instead of running the backbone over the same tiles a second time.
            if self.forensic_pack is not None or self.combined_pack is not None:
                try:
                    t = time.perf_counter()
                    bgr_full = parsed.bgr_full if parsed.bgr_full is not None else parsed.bgr
                    tile_detail = self.spec.tile_detail(self.clip, bgr_full)
                    tile_feat = self.spec.aggregate_detail(tile_detail)
                    latency["L2_clip_tiles"] = (time.perf_counter() - t) * 1000
                except Exception as e:
                    rep.uncertainty.append(f"分块CLIP证据不可用：{e}")

            if self.clip_domain_pack is not None:
                scores.append(ModelScore("clip_domain",
                                         self._pack_logit(self.clip_domain_pack, clip_feat)))
            if self.combined_pack is not None:
                try:
                    combined_logit = self._pack_logit(self.combined_pack, self.spec.combined_vector(
                        x1, xb, clip_feat, tile_feat, patch_feat))
                    scores.append(ModelScore("combined", combined_logit))
                except Exception as e:
                    # a member is an evidence producer, not a gate (manual ch.4.2/16.3):
                    # a missing tile block must degrade this member, not abort the item
                    rep.uncertainty.append(f"combined 头不可用（降级运行）：{e}")

        if self.npr is not None and _elapsed() < budget - 0.8:
            t = time.perf_counter()
            scores.append(ModelScore("npr", self.npr.logit(parsed.bgr)))
            latency["L2_npr"] = (time.perf_counter() - t) * 1000

        if self.blade is not None and _elapsed() < budget - 0.5:
            t = time.perf_counter()
            # AI-side logit from the frozen SD-VAE round trip: higher = more likely
            # latent-diffusion generated - the same scale and direction the ablation
            # (repo/scripts/aeroblade_ablation.py) measured. Using the raw error with
            # direction=-1 would leave this member uncalibratable (no temperature in
            # models/fusion.json -> sigmoid(error) ~= 0.5 for every image).
            scores.append(ModelScore("aeroblade", self.blade.logit(parsed.bgr)))
            latency["L2_aeroblade"] = (time.perf_counter() - t) * 1000

        # Classifier attention explanation (manual ch.9.2 "Grad-CAM (if available)").
        # Kept strictly separate from L3: this only says which tiles the classifier
        # relied on, it is NOT a tamper region. Skipped when the time budget is
        # nearly exhausted; the report then says 不可用 with the reason.
        t = time.perf_counter()
        if _elapsed() < budget - 0.8:
            try:
                rep.explanation = self._attention_explanation(
                    rep, parsed, x1, xb, clip_feat, tile_feat, combined_logit, patch_feat,
                    tile_detail)
            except Exception as e:
                rep.explanation = self._unavailable_explanation(
                    f"关注区域解释生成失败：{e}",
                    gradcam={"available": False, "status": "不可用",
                             "reason": f"关注区域解释生成失败：{e}"})
                rep.uncertainty.append(f"关注区域解释生成失败：{e}")
        else:
            from l3_localization import ATTRIBUTION_METHOD, gradcam_supported
            g_ok, g_reason = gradcam_supported(self.combined_pack)
            rep.explanation = self._unavailable_explanation(
                f"时间预算不足（已用 {_elapsed():.2f}s / {budget}s），跳过关注区域解释",
                gradcam={"available": bool(g_ok), "status": "可用" if g_ok else "不可用",
                         "reason": g_reason or "时间预算不足，归因未执行",
                         "substitute": ATTRIBUTION_METHOD})
        latency["explain"] = (time.perf_counter() - t) * 1000

        # L3 localization (only if needed: suspect/review path or always for demo)
        t = time.perf_counter()
        loc_kind = "zero-weight forensic map (ELA+noise inconsistency)"
        try:
            from l3_localization import combined_anomaly_map, top_anomaly_boxes
            amap = combined_anomaly_map(parsed.bgr)
            boxes = top_anomaly_boxes(amap)
            rep.localization = {
                "available": True,
                "kind": loc_kind,
                "heatmap_png_b64": heatmap_to_b64_png(amap, parsed.bgr),
                "boxes": boxes,
                "reason": None,
            }
        except Exception as e:
            rep.localization = {"available": False, "kind": loc_kind,
                                "heatmap_png_b64": None, "boxes": [],
                                "reason": f"L3定位分支不可用：{e}"}
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
        if self._load_s - load_before > 1e-9:
            # only reported when this item paid the cold-start cost (manual ch.9.2
            # timing is per item; the load row keeps `total` explainable)
            latency["setup_load"] = (self._load_s - load_before) * 1000
        rep.latency_ms = {k: round(v, 1) for k, v in latency.items()}
        rep.versions = self._traceability(rep)
        rep.finalize()
        return rep, dec.confidence

    # ---- video path ----
    def detect_video(self, path: str) -> DetectionReport:
        import video as vid
        t0 = time.perf_counter()
        latency = {}
        rep = DetectionReport(input_type="video")
        try:
            frames, fps, total = vid.extract_frames(path)
        except Exception as e:
            rep.verdict = "review"
            rep.uncertainty.append(f"视频抽帧失败：{e}（转人工复核）")
            return rep.finalize()
        rep.resolution = f"{frames[0][1].shape[1]}x{frames[0][1].shape[0]}" if frames else ""
        rep.duration_s = round(total / fps, 2) if fps else None

        # Batch the whole frame set through the frozen backbone in one pass, and use
        # only the members that need no native-resolution tiles, so the <=5s video
        # budget holds. Tiles are reserved for the suspicious-frame re-check below.
        frame_scores = []
        budget = self.cfg.time_budget_s
        imgs = [fr for _, fr in frames]
        clip_batch = None
        if self.clip is not None and imgs:
            t = time.perf_counter()
            try:
                clip_batch = self.clip.batch(imgs, batch_size=8)
                latency["L2_clip_batch"] = (time.perf_counter() - t) * 1000
            except Exception as e:
                rep.uncertainty.append(f"批量CLIP推理失败：{e}")

        # Budget allocation for the video path. Measured on controlled clips: the light
        # per-frame path (L1 + L1b forensics + global CLIP on EVERY frame) alone costs
        # about 4 s of the 5 s budget, leaving nothing for the native-resolution tiles
        # that carry the image path's discriminative power (light path AUC 0.62 vs 0.97
        # for the image path on the same stills). So: batched global CLIP on every frame
        # (cheap), and the expensive L1/L1b + tiled-CLIP evidence only on the K most
        # suspicious frames, ranked by their global-CLIP score.
        from l1_forensics import feature_vector as _fv
        from l1b_domain import extract_l1b_features as _l1b, feature_vector_b as _fvb
        _t_frames = time.perf_counter()
        per_frame: list[tuple] = []          # (idx, ts, scores, frame)
        for k, (idx, fr) in enumerate(frames):
            ts = idx / fps if fps else 0.0
            s: list[ModelScore] = []
            if clip_batch is not None:
                cf = clip_batch[k]
                s.append(ModelScore("clip_external", self._pack_logit_clip_head(cf)))
                if self.clip_domain_pack is not None:
                    s.append(ModelScore("clip_domain", self._pack_logit(self.clip_domain_pack, cf)))
            if self.npr is not None:
                s.append(ModelScore("npr", self.npr.logit(fr)))
            if not s:
                rep.uncertainty.append("无可用L2模型，视频无法评分")
                continue
            per_frame.append((idx, ts, s, fr))
        latency["frames_light"] = (time.perf_counter() - _t_frames) * 1000

        # expensive evidence on the K most suspicious frames only
        k_tiles = max(0, int(getattr(self.cfg, "video_tile_frames", 2)))
        if k_tiles and per_frame:
            from l2_clip_tiles import extract_tiled_features
            order = sorted(range(len(per_frame)),
                           key=lambda i: self.fusion.fuse(per_frame[i][2]).fused_score,
                           reverse=True)[:k_tiles]
            t = time.perf_counter()
            done = 0
            for i in order:
                if (time.perf_counter() - t0) >= self.cfg.time_budget_s - 0.5:
                    break
                idx, ts, s, fr = per_frame[i]
                try:
                    ff = extract_l1_features(ParsedInput(ok=True, bgr=fr, raw_bytes=0,
                                                         orig_width=fr.shape[1],
                                                         orig_height=fr.shape[0]))
                    xf = np.concatenate([_fv(ff), _fvb(_l1b(ParsedInput(
                        ok=True, bgr=fr, raw_bytes=0, orig_width=fr.shape[1],
                        orig_height=fr.shape[0])))])
                    s2 = list(s)
                    if self.forensic_pack is not None:
                        s2.append(ModelScore("forensic", self._pack_logit(self.forensic_pack, xf)))
                    if self.combined_pack is not None and self.clip is not None:
                        cf = clip_batch[i] if clip_batch is not None else self.clip(fr)
                        tf = extract_tiled_features(self.clip, fr)
                        s2.append(ModelScore("combined",
                                             self._pack_logit(self.combined_pack,
                                                              np.concatenate([xf, cf, tf]))))
                    per_frame[i] = (idx, ts, s2, fr)
                    done += 1
                except Exception as e:
                    rep.uncertainty.append(f"视频深度证据不可用：{e}")
            latency["video_deep_frames"] = (time.perf_counter() - t) * 1000
            rep.versions["video_deep_frames"] = str(done)

        for idx, ts, s, fr in per_frame:
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
        rep.localization = {"available": True, "kind": "temporal segments", "reason": None,
                            "heatmap_png_b64": None,
                            "suspicious_segments": vid.suspicious_ranges(frame_scores, self.fusion.thr_high),
                            "note": "视频通道复用图像模型（批量全局CLIP+取证），分块CLIP仅用于图像入口；"
                                    "该通道不提供空间热力图"}
        # no spatial attention map on the video channel: state it explicitly instead of
        # leaving the field blank (manual ch.9.2)
        rep.explanation = self._unavailable_explanation(
            "视频通道（video-stage1）不产出空间关注区域解释；可疑片段见定位分节",
            method="不适用：视频通道采用批量全局CLIP+取证打分，不产出块级特征",
            disclaimer="关注区域解释 ≠ 伪造区域定位；视频通道无可视化关注区域。",
            gradcam={"available": False, "status": "不可用",
                     "reason": "video-stage1 通道对批量帧只做全局 CLIP + 取证打分，"
                               "未生成块级特征，无可归因特征图",
                     "substitute": "不适用（视频通道无可视化关注区域）"})
        try:
            rep.input_hash = sha256_file(path)
        except Exception as e:
            rep.uncertainty.append(f"视频输入哈希不可用：{e}")
        latency["total"] = (time.perf_counter() - t0) * 1000
        rep.latency_ms = {k: round(v, 1) for k, v in latency.items()}
        rep.versions = self._traceability(rep, extra={
            "pipeline": "open-sz-01 v0.2", "channel": "video-stage1",
            "frames": str(len(frame_scores)),
            "note": "合成烟测片用于验证通道；命题样本不含视频，视频指标非基准结果"})
        return rep.finalize()
