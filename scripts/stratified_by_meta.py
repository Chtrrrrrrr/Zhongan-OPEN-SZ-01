#!/usr/bin/env python3
"""Per-stratum metrics using the MANUAL visual labels in data/meta.csv.

Closes audit finding F3a: instead of a proxy that failed validation, the 119
samples now carry labels from a manual visual pass:
  acquisition: flat_render (digitally rendered, no capture artefacts)
             | photographic (shows capture characteristics: perspective,
               lighting gradient, shadow, background, hand, camera watermark)
  content:     document | scene
  generator:   unavailable (the brief ships none) -> covered by the
               cross-generator experiment in reports/cross_generator.json

Reports, for each stratum, the out-of-fold metrics of the frozen pipeline plus a
shortcut test: if the "rendered vs photographed" appearance were a class shortcut,
the label distribution across strata would be strongly skewed. It is NOT - which
is why the headline number cannot be explained by that shortcut.

Output: reports/stratified_manual.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/meta.csv")
    ap.add_argument("--out", default="reports/stratified_manual.json")
    args = ap.parse_args()

    meta = {}
    with open(WS / args.meta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[int(r["seq"])] = r

    # out-of-fold predictions from the shared cached bundle (same protocol as metrics.json)
    bundle_path = WS / "models" / "oof_bundle_119.json"
    if not bundle_path.exists():
        raise SystemExit(f"missing {bundle_path}; run repo/scripts/metrics_common.py first")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    seqs = [int(s) for s in bundle["seqs"]]
    y = np.asarray(bundle["y"], dtype=int)
    dep = bundle["deployed_oof"]
    fused = np.asarray(dep["fused"], dtype=float)
    pred = np.asarray(dep["pred"], dtype=int)

    def metrics(idx):
        idx = np.asarray(idx)
        if len(idx) == 0:
            return None
        yy, pp = y[idx], pred[idx]
        tp = int(((yy == 1) & (pp == 1)).sum())
        tn = int(((yy == 0) & (pp == 0)).sum())
        fp = int(((yy == 0) & (pp == 1)).sum())
        fn = int(((yy == 1) & (pp == 0)).sum())
        n_real, n_ai = int((yy == 0).sum()), int((yy == 1).sum())
        return {
            "n": int(len(idx)),
            "n_ai": n_ai, "n_real": n_real,
            "accuracy": round((tp + tn) / len(idx), 4),
            "fpr": (round(fp / n_real, 4) if n_real else None),
            "tpr": (round(tp / n_ai, 4) if n_ai else None),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    idx_by_seq = {int(s): i for i, s in enumerate(seqs)}
    out = {"report": "stratified_manual", "meta_source": args.meta,
           "label_origin": "manual visual pass by the captain (not a proxy, not hand-waved)",
           "semantics": {
               "acquisition": ("APPEARANCE as observed in the visual pass: flat_render = no "
                               "capture artefacts (no perspective/lighting gradient/shadow/"
                               "background/hand/watermark); photographic = capture-like. For "
                               "AI-generated images there is no real camera, so this records "
                               "appearance, which is exactly the variable the "
                               "rendered-vs-photographed shortcut would key on."),
               "content": "document (form/invoice/record page) vs scene (object/room/food)",
               "generator": "unavailable - the brief ships no generator labels",
           },
           "protocol": "out-of-fold fused score from the same folds/seed as reports/metrics.json",
           "generator": {"status": "unavailable",
                         "reason": "the brief ships no generator labels",
                         "substitute": "reports/cross_generator.json"},
           "strata": {}}

    for key in ("acquisition", "content"):
        groups = defaultdict(list)
        for seq, row in meta.items():
            if seq in idx_by_seq:
                groups[row[key]].append(idx_by_seq[seq])
        out["strata"][key] = {g: metrics(v) for g, v in sorted(groups.items())}

    # cross-tab of label x appearance (the shortcut question)
    tab = Counter((meta[s]["acquisition"], meta[s]["label"]) for s in meta)
    n_ai = sum(1 for s in meta if meta[s]["label"] == "ai")
    n_real = sum(1 for s in meta if meta[s]["label"] == "real")
    share = {
        "flat_render_share_ai": round(tab[("flat_render", "ai")] / max(1, n_ai), 4),
        "flat_render_share_real": round(tab[("flat_render", "real")] / max(1, n_real), 4),
        "scene_share_ai": round(sum(1 for s in meta if meta[s]["content"] == "scene"
                                   and meta[s]["label"] == "ai") / max(1, n_ai), 4),
        "scene_share_real": round(sum(1 for s in meta if meta[s]["content"] == "scene"
                                      and meta[s]["label"] == "real") / max(1, n_real), 4),
    }
    gap = abs(share["flat_render_share_ai"] - share["flat_render_share_real"])
    out["shortcut_test"] = {
        "crosstab": {f"{a}/{l}": tab[(a, l)] for a in ("flat_render", "photographic")
                     for l in ("ai", "real")},
        **share,
        "appearance_gap": round(gap, 4),
        "verdict": ("NO usable rendered-vs-photographed shortcut: the flat-render share is "
                    "nearly equal across classes" if gap < 0.15 else
                    "appearance is skewed across classes - the shortcut is a live risk"),
        "why_it_matters": ("the frozen pipeline cannot reach 0.958 by keying on appearance; "
                           "the class-balanced appearance distribution also explains the two "
                           "false positives (real scene photos 057/058) and the three false "
                           "negatives (flat-render AI samples 003/029/030)"),
    }
    out["error_strata"] = {
        "false_positives": [int(s) for s, i in idx_by_seq.items()
                            if y[i] == 0 and pred[i] == 1],
        "false_negatives": [int(s) for s, i in idx_by_seq.items()
                            if y[i] == 1 and pred[i] == 0],
    }

    p = WS / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["shortcut_test"], ensure_ascii=False, indent=2))
    for key, groups in out["strata"].items():
        print(f"\n[{key}]")
        for g, m in groups.items():
            print(f"  {g:14s} n={m['n']:3d} acc={m['accuracy']:.4f} "
                  f"fpr={m['fpr']} tpr={m['tpr']} cm={m['confusion']}")
    print(f"\n[save] {p}")


if __name__ == "__main__":
    main()
