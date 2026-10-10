"""Video channel (manual ch.8.5) - Stage 1 minimal path.

video -> uniformly spaced frames -> per-frame image detection ->
aggregate (mean / median / P90 / std) -> video-level score + suspicious segments.

Design rules:
  - sequential uniform sampling (predictable latency, reproducible)
  - complex temporal features (flicker / optical flow) only if Stage 1 shows gains
  - degradable: if frame extraction fails, fall back to first/middle/last frame
  - frame budget: the Stage-1 design used 16 frames, but the measured end-to-end
    cost exceeded the 5s budget on the reference machine, so the default is 10.
    Manual ch.16.3 lists "限制抽帧" as the documented latency mitigation; the
    count stays configurable per call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_FRAMES = 10


@dataclass
class VideoFrameScore:
    frame_idx: int
    timestamp_s: float
    score: float


@dataclass
class VideoResult:
    n_frames_total: int
    fps: float
    duration_s: float
    frame_scores: list = field(default_factory=list)   # list[VideoFrameScore]
    agg_mean: float = 0.0
    agg_median: float = 0.0
    agg_p90: float = 0.0
    agg_std: float = 0.0
    suspicious_segments: list = field(default_factory=list)  # [(t0, t1)]
    degraded: bool = False


def extract_frames(video_path: str, n: int = N_FRAMES) -> tuple[list, float, int]:
    """Uniform sequential sampling. Returns (frames_bgr, fps, total_frames)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    if total <= 0:
        # unknown length: read sequentially and sample afterwards
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        total = len(frames)
        if total == 0:
            raise RuntimeError("no frames decoded")
        idx = np.linspace(0, total - 1, min(n, total)).astype(int)
        return [frames[i] for i in idx], fps, total
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append((int(i), fr))
    cap.release()
    if len(frames) < 3:  # degrade: first / middle / last
        frames = []
        for i in {0, total // 2, total - 1}:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            cap.release()
            if ok:
                frames.append((int(i), fr))
    return frames, fps, total


def aggregate(scores: list[float]) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "p90": float(np.percentile(s, 90)),
        "std": float(s.std()),
    }


def suspicious_ranges(frame_scores: list[VideoFrameScore], thr: float, min_run: int = 2):
    """Group consecutive above-threshold frames into time segments."""
    segs, run = [], []
    for fs in frame_scores:
        if fs.score >= thr:
            run.append(fs)
        else:
            if len(run) >= min_run:
                segs.append((round(run[0].timestamp_s, 2), round(run[-1].timestamp_s, 2)))
            run = []
    if len(run) >= min_run:
        segs.append((round(run[0].timestamp_s, 2), round(run[-1].timestamp_s, 2)))
    return segs
