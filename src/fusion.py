"""Decision layer: score calibration, evidence fusion, three-tier decision.

Pipeline (manual ch.9):
  per-model raw scores -> temperature-scaling calibration -> weighted logit
  fusion -> final calibration -> three-tier decision (normal / review / suspect).

Thresholds are derived ONLY from a pure-real calibration set (target FPR) and
verified on an independent real holdout - never on the final test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelScore:
    name: str
    raw: float          # raw model output (logit or error)
    direction: int = 1  # +1: higher = more AI; -1: higher = more real


@dataclass
class DecisionResult:
    verdict: str               # "normal" | "review" | "suspect"
    confidence: float          # calibrated probability of "AI-generated/tampered"
    fused_score: float
    model_scores: dict = field(default_factory=dict)
    disagreement: float = 0.0  # std of per-model standardized scores
    reasons: list = field(default_factory=list)


class TemperatureScaler:
    """Single-parameter calibration on logits (Guo et al., ICML 2017)."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200):
        from scipy.optimize import minimize

        def nll(t):
            t = np.clip(t[0], 1e-2, 100.0)
            z = logits / t
            # numerically stable BCE with logits
            loss = np.mean(np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z))))
            return loss

        res = minimize(nll, x0=[1.0], method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-4})
        self.temperature = float(np.clip(res.x[0], 1e-2, 100.0))
        return self

    def transform_proba(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-z))


class FusionEngine:
    """Weighted logit fusion with three-tier thresholds."""

    def __init__(self, weights: dict[str, float] | None = None,
                 thr_low: float = 0.35, thr_high: float = 0.80,
                 disagree_review: float = 1.25):
        self.weights = weights or {}
        self.thr_low = thr_low      # below -> normal
        self.thr_high = thr_high    # above -> suspect
        self.disagree_review = disagree_review
        self.scalers: dict[str, TemperatureScaler] = {}

    # ---------- calibration ----------
    def fit_calibrators(self, scores: dict[str, np.ndarray], labels: np.ndarray):
        for name, s in scores.items():
            sc = TemperatureScaler().fit(np.asarray(s, dtype=np.float64), labels)
            self.scalers[name] = sc
        return self

    def calibrate(self, name: str, raw: float) -> float:
        sc = self.scalers.get(name)
        if sc is None:
            return float(1.0 / (1.0 + np.exp(-raw)))
        return float(sc.transform_proba(np.array([raw]))[0])

    # ---------- thresholding on pure-real set ----------
    def set_thresholds_from_real(self, real_fused_scores: np.ndarray,
                                 target_fpr_low: float = 0.10,
                                 target_fpr_high: float = 0.01):
        """Empirical quantiles on a pure-real calibration set."""
        s = np.sort(np.asarray(real_fused_scores))
        n = len(s)
        self.thr_low = float(s[min(n - 1, int(np.floor((1 - target_fpr_low) * n)))])
        self.thr_high = float(s[min(n - 1, int(np.floor((1 - target_fpr_high) * n)))])
        return self

    # ---------- fusion ----------
    def fuse(self, scores: list[ModelScore]) -> DecisionResult:
        calibrated = {}
        for s in scores:
            p = self.calibrate(s.name, s.raw * s.direction)
            calibrated[s.name] = p
        if not calibrated:
            return DecisionResult(verdict="review", confidence=0.5, fused_score=0.0,
                                  reasons=["no model evidence available"])
        # Only members listed in the frozen weight table participate. A member that
        # was ablated out (or is not part of the frozen version) must NOT silently
        # enter the decision, so an unconfigured name has weight 0.
        names = [n for n in calibrated if self.weights.get(n, 0.0) > 0]
        if names:
            w = np.array([self.weights[n] for n in names], dtype=np.float64)
        else:
            # no weight table (bare engine, e.g. unit tests): use every member equally
            names = list(calibrated)
            w = np.ones(len(names), dtype=np.float64)
        w = w / w.sum()
        ps = np.array([calibrated[n] for n in names])
        # logit-level weighted mean
        eps = 1e-6
        logits = np.log(np.clip(ps, eps, 1 - eps) / (1 - np.clip(ps, eps, 1 - eps)))
        fused_logit = float((w * logits).sum())
        fused = float(1.0 / (1.0 + np.exp(-fused_logit)))
        disagreement = float(np.std(logits))

        reasons = []
        if fused >= self.thr_high:
            verdict = "suspect"
            reasons.append(f"fused score {fused:.3f} >= high threshold {self.thr_high:.3f}")
        elif fused <= self.thr_low:
            verdict = "normal"
            reasons.append(f"fused score {fused:.3f} <= low threshold {self.thr_low:.3f}")
        else:
            verdict = "review"
            reasons.append(f"fused score {fused:.3f} in uncertainty band "
                           f"({self.thr_low:.3f}, {self.thr_high:.3f})")
        if disagreement >= self.disagree_review and verdict != "suspect":
            verdict = "review"
            reasons.append(f"model disagreement {disagreement:.2f} >= {self.disagree_review}")

        return DecisionResult(
            verdict=verdict,
            confidence=fused,
            fused_score=fused,
            model_scores={n: round(float(p), 4) for n, p in calibrated.items()},
            disagreement=round(disagreement, 4),
            reasons=reasons,
        )

    # ---------- persistence ----------
    def save(self, path: str):
        payload = {
            "weights": self.weights,
            "thr_low": self.thr_low,
            "thr_high": self.thr_high,
            "disagree_review": self.disagree_review,
            "temperatures": {k: v.temperature for k, v in self.scalers.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "FusionEngine":
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        eng = cls(weights=p["weights"], thr_low=p["thr_low"], thr_high=p["thr_high"],
                  disagree_review=p.get("disagree_review", 1.25))
        for k, t in p.get("temperatures", {}).items():
            sc = TemperatureScaler()
            sc.temperature = t
            eng.scalers[k] = sc
        return eng
