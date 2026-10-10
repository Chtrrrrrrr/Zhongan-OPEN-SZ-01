#!/usr/bin/env python3
"""Smoke-test the demo HTTP service (page, health, image upload, video upload).

Usage:
  python demo/app.py --port 7862        # in one shell
  python scripts/test_demo_api.py --base http://127.0.0.1:7862
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7862")
    ap.add_argument("--ai-image", default="data/images/001.png")
    ap.add_argument("--real-image", default="data/images/057.png")
    ap.add_argument("--video", default="demo_assets/smoke_test.mp4")
    args = ap.parse_args()

    ok = True
    r = requests.get(args.base + "/", timeout=20)
    print(f"[page]   {r.status_code}  {len(r.content)} bytes")
    ok &= r.status_code == 200

    r = requests.get(args.base + "/api/health", timeout=20)
    print(f"[health] {r.json()}")
    ok &= r.status_code == 200

    for tag, rel in (("ai", args.ai_image), ("real", args.real_image)):
        p = WS / rel
        if not p.exists():
            print(f"[{tag}] skipped (missing {p})")
            continue
        r = requests.post(args.base + "/api/detect?name=" + p.name,
                          data=p.read_bytes(), timeout=300)
        j = r.json()
        if "error" in j:
            print(f"[{tag}] ERROR {j['error']}")
            ok = False
            continue
        print(f"[{tag}]   verdict={j['verdict_cn']} conf={j['confidence']:.3f} "
              f"fused={j['fused_score']:.3f} "
              f"total={j.get('latency_ms', {}).get('total')}ms "
              f"clues={len(j.get('forensic_clues', []))} "
              f"heatmap={'yes' if j.get('localization', {}).get('heatmap_png_b64') else 'no'} "
              f"models={list(j.get('model_scores', {}).keys())}")
        ok &= bool(j.get("localization", {}).get("heatmap_png_b64"))

    vp = WS / args.video
    if vp.exists():
        r = requests.post(args.base + "/api/detect_video?name=" + vp.name,
                          data=vp.read_bytes(), timeout=300)
        j = r.json()
        if "error" in j:
            print(f"[video] ERROR {j['error']}")
            ok = False
        else:
            print(f"[video] verdict={j['verdict_cn']} conf={j['confidence']:.3f} "
                  f"duration={j.get('duration_s')}s "
                  f"total={j.get('latency_ms', {}).get('total')}ms "
                  f"segments={j.get('localization', {}).get('suspicious_segments')}")
    else:
        print(f"[video] skipped (missing {vp})")

    print("\nDEMO API TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
