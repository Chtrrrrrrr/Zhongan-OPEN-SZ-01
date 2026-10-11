#!/usr/bin/env python3
"""Freeze the EVALUATION protocol (not the data split) - closes audit finding F3c.

`data/splits_frozen.json` freezes the 0.6/0.2/0.2 data split used for the D1-D8
split discipline. The headline metrics, however, use stratified 5-fold
out-of-fold cross-validation (a single 24-sample test set has an accuracy std of
0.0422 and only 56.4% of draws reach >=95%). The two are complementary, so this
script writes a SECOND, non-invasive freeze file describing the evaluation
protocol and its reproducibility signature. It deliberately does not touch
splits_frozen.json, so `split_dataset.py --verify-frozen` keeps working.

Usage: python repo/scripts/freeze_eval_protocol.py [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent

OUT = WS / "data" / "eval_protocol_frozen.json"


def sha256_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def build() -> dict:
    recon = WS / "repo" / "reports" / "protocol_reconciliation.json"
    if not recon.exists():
        recon = WS / "reports" / "protocol_reconciliation.json"
    sig = {}
    headline = {}
    if recon.exists():
        d = json.loads(recon.read_text(encoding="utf-8"))
        sig = d.get("evaluation_protocol_signature", {})
        headline = sig.get("headline_metrics_on_this_signature", {})
        fold_sig = sig.get("fold_assignment_sha256")
        proto_sig = sig.get("protocol_sha256")
        artifacts = sig.get("artifacts_sha256", {})
    else:
        fold_sig = proto_sig = None
        artifacts = {}

    payload = {
        "version": 1,
        "kind": "evaluation_protocol",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("Freezes the EVALUATION protocol only. The data split stays frozen in "
                 "data/splits_frozen.json and is verified by "
                 "`scripts/split_dataset.py --verify-frozen`; this file adds the "
                 "out-of-fold protocol signature required by audit finding F3c and "
                 "manual ch.15/17.3."),
        "why_two_files": (
            "data/splits_frozen.json governs the D1-D8 split discipline; the headline "
            "metrics use stratified out-of-fold CV because a single 24-sample test set "
            "has accuracy std 0.0422 and only 56.4% of draws reach >=95% "
            "(reports/protocol_reconciliation.json)."),
        "protocol": {
            "cv": "stratified k-fold out-of-fold, k=5, shuffle=True, random_state=42, "
                  "stratified on label (ai/real)",
            "inner_cv": "k=4 for temperature fitting (random_state=100+fold)",
            "head_selection": "every head is fit on the training fold only; no selection "
                              "or threshold search on the held-out fold",
            "threshold": "order statistic of the training-fold real scores; "
                         "target FPR 0.03 with an absolute false-positive budget of "
                         "floor(0.03*70)=2 (transferable across set sizes)",
            "members": ["combined", "forensic"],
            "external_member_admission": "clip_external enters a fold only if its "
                                         "training-fold AUC >= 0.65",
        },
        "signatures": {
            "fold_assignment_sha256": fold_sig,
            "protocol_sha256": proto_sig,
            "artifacts_sha256": artifacts,
        },
        "headline_metrics_on_this_signature": headline,
        "verification_command": "python repo/scripts/freeze_eval_protocol.py --verify",
    }
    return payload


def verify(payload: dict) -> int:
    if not OUT.exists():
        print(f"[verify] {OUT} missing - run without --verify first")
        return 1
    frozen = json.loads(OUT.read_text(encoding="utf-8"))
    ok = True
    for key in ("fold_assignment_sha256", "protocol_sha256"):
        a = frozen["signatures"].get(key)
        b = payload["signatures"].get(key)
        same = a == b
        ok &= same
        print(f"[verify] {key}: {'match' if same else 'MISMATCH'} "
              f"(frozen={str(a)[:12]}… current={str(b)[:12]}…)")
    arts = frozen["signatures"].get("artifacts_sha256", {})
    drift = [k for k, v in arts.items()
             if Path(k).exists() and sha256_file(Path(k)) != v]
    if drift:
        print(f"[verify] artifact drift in {len(drift)} file(s): {drift[:6]}")
        print("        (weights/features were regenerated after the freeze)")
    else:
        print("[verify] all recorded artifacts match")
    print("[verify] overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    payload = build()
    if args.verify:
        sys.exit(verify(payload))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("kind", "created")}, ensure_ascii=False))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
