#!/usr/bin/env python3
"""D2/D3 stratification keys for split_dataset.py - and an honest NULL result.

The brief ships no generator or acquisition labels for the 119 samples, so
split_dataset.py's D2 (per-generator) and D3 (per-acquisition) stratification
currently degrade to label-only. This script:

  1. attempts a *measurable* proxy for acquisition/content from the L1b features;
  2. validates the proxy against the documented, qualitatively audited bias
     (docs/data-audit.md: the AI class is dominated by digitally rendered
     documents, the real class by phone captures of paper; scene photos are few);
  3. if the proxy contradicts that bias it is REJECTED and NOT shipped - the
     stratification keys are written as empty and marked unavailable.

Why reject instead of shipping something: a proxy that does not reproduce the
known bias would silently make a D3-stratified split meaningless while looking
complete. Recording the rejection is the honest outcome.

Outputs:
  data/meta.csv                  - seq,label,generator,acquisition,content,source
                                   (generator/acquisition/content = "unavailable"
                                    when the proxy is rejected)
  reports/meta_proxy_attempt.json - the attempt, its validation and the verdict

Usage: python repo/scripts/build_meta.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

WHITE_MIN = 0.05
FLAT_NOISE_MAX = 0.9

# documented qualitative expectations (docs/data-audit.md, section 5)
EXPECTED = {
    "scene_photos_total": "about 10 (a few property-loss/food photos); the rest are documents",
    "ai_mostly": "digitally rendered documents",
    "real_mostly": "phone captures of paper",
}


def main():
    from l1b_domain import FEATURE_NAMES_B as names

    feats = WS / "models" / "l1b_feats_119.npy"
    if not feats.exists():
        raise SystemExit(f"missing {feats}; run scripts/extract_l1b_features.py first")
    X = np.load(feats)
    i_white = names.index("b_nearWhiteFrac")
    i_noise = names.index("b_flatNoise")

    seqs, labels = [], []
    with open(WS / "data" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seqs.append(int(r["seq"]))
            labels.append(r["label"])

    rows = []
    for k, seq in enumerate(seqs):
        white, noise = float(X[k, i_white]), float(X[k, i_noise])
        content = "document" if white >= WHITE_MIN else "scene"
        acq = "rendered" if (white >= WHITE_MIN and noise < FLAT_NOISE_MAX) else "photo"
        rows.append({"seq": seq, "label": labels[k], "acquisition_proxy": acq,
                     "content_proxy": content})

    acq_x_label = {f"{a}/{l}": sum(1 for r in rows if r["acquisition_proxy"] == a and r["label"] == l)
                   for a in ("rendered", "photo") for l in ("ai", "real")}
    con_counts = dict(Counter(r["content_proxy"] for r in rows))

    # validation against the documented bias
    render_ai, render_real = acq_x_label["rendered/ai"], acq_x_label["rendered/real"]
    photo_ai, photo_real = acq_x_label["photo/ai"], acq_x_label["photo/real"]
    n_ai = sum(1 for r in rows if r["label"] == "ai")
    n_real = sum(1 for r in rows if r["label"] == "real")
    # the bias would show up as rendered being concentrated in the AI class
    render_share_ai = render_ai / max(1, n_ai)
    render_share_real = render_real / max(1, n_real)
    bias_gap = render_share_ai - render_share_real
    scene_fraction = con_counts.get("scene", 0) / len(rows)

    verdict = "rejected"
    reasons = []
    if abs(bias_gap) < 0.15:
        reasons.append(
            f"rendered share is nearly equal across classes (ai {render_share_ai:.2f} vs "
            f"real {render_share_real:.2f}; gap {bias_gap:+.2f}), so the proxy does not "
            f"reproduce the documented 'AI mostly rendered / real mostly photographed' bias")
    if scene_fraction > 0.30:
        reasons.append(
            f"the content proxy labels {con_counts.get('scene', 0)}/{len(rows)} "
            f"({scene_fraction:.0%}) as scene photos, contradicting the documented audit "
            f"({EXPECTED['scene_photos_total']})")

    attempt = {
        "script": "repo/scripts/build_meta.py",
        "purpose": "D2/D3 stratification keys for scripts/split_dataset.py",
        "proxy_rule": {"b_nearWhiteFrac >= ": WHITE_MIN, "b_flatNoise < ": FLAT_NOISE_MAX,
                       "source_features": "models/l1b_feats_119.npy",
                       "note": "rule is measurable and reproducible, but it is still a proxy"},
        "acquisition_proxy_x_label": acq_x_label,
        "content_proxy_counts": con_counts,
        "validation": {
            "expected_from_data_audit": EXPECTED,
            "rendered_share_ai": round(render_share_ai, 4),
            "rendered_share_real": round(render_share_real, 4),
            "bias_gap": round(bias_gap, 4),
            "scene_fraction": round(scene_fraction, 4),
        },
        "verdict": verdict,
        "reasons": reasons,
        "consequence": (
            "data/meta.csv keeps generator/acquisition/content as 'unavailable'; D2/D3 "
            "stratification in scripts/split_dataset.py therefore degrades to label-only "
            "with its built-in warning. Compliance audit records D2/D3 as PARTIAL: labels "
            "are not provided by the brief and the in-project proxy was validated and "
            "rejected rather than shipped."),
        "how_to_close": (
            "hand-label acquisition/content for the 119 samples (or obtain the brief's "
            "metadata) and re-run scripts/split_dataset.py --meta data/meta.csv; the "
            "18-generator cross-generator evidence in reports/cross_generator.json "
            "independently covers D2's generalisation intent."),
    }

    out_attempt = WS / "reports" / "meta_proxy_attempt.json"
    out_attempt.write_text(json.dumps(attempt, ensure_ascii=False, indent=2), encoding="utf-8")

    src_col = "proxy_rejected_unavailable" if verdict == "rejected" else "proxy_accepted"
    meta_rows = [{"seq": r["seq"], "label": r["label"],
                  "generator": "unavailable", "acquisition": "unavailable",
                  "content": "unavailable", "source": src_col} for r in rows]
    out_csv = WS / "data" / "meta.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "label", "generator", "acquisition",
                                          "content", "source"])
        w.writeheader()
        w.writerows(meta_rows)

    print(json.dumps({"verdict": verdict, "reasons": reasons,
                      "acquisition_proxy_x_label": acq_x_label,
                      "content_proxy_counts": con_counts}, ensure_ascii=False, indent=2))
    print(f"[save] {out_csv} (generator/acquisition/content = unavailable)")
    print(f"[save] {out_attempt}")


if __name__ == "__main__":
    main()
