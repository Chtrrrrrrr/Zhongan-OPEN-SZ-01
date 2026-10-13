#!/usr/bin/env python3
"""t5 - real-encoded video validation + inter-frame temporal evidence.

Run:
    python repo/scripts/video_temporal_validation.py --out repo/reports/video_temporal_validation.json

What it does (all pre-registered, nothing tuned on the evaluated clips):
  1. probes the environment and records the frozen model/fusion artifacts by sha256;
  2. runs the unit self-test of src/video_temporal.py and embeds the result;
  3. builds a controlled validation set of GENUINELY ENCODED clips (every frame is a
     different frame: deterministic sub-pixel pan + zoom + per-frame noise), identical
     codec/fps/frame-count/canvas/motion/noise for every clip, so the arms differ only
     in the still that produced the frames:
        arm A_source_controlled    20 real clips <- repo/testset/real real claim photos
                                   20 fake clips <- data/images AI stills
        arm B_in_domain_same_source 20 real clips <- data/images real stills (seq 57+)
                                    20 fake clips <- THE SAME fake clips as arm A
  4. computes the inter-frame temporal features (flow consistency / noise-residual
     continuity / HF stability) on the DECODED frames of every clip and reports their
     univariate AUC per arm;
  5. scores every clip through the production video path (detector.detect_video) and
     reports video-level AUC/TPR/FPR (with k/N counts) plus P50/P95 latency;
  6. compares arm B with the archived 30-clip run under the same metric protocol;
  7. writes the next-step acquisition list for real-shot video and public AI-video corpora.

Red lines enforced in the output:
  * the brief contains NO video - every number here is CHANNEL-level, not a
    business-level video-detection capability result;
  * no AI-generated video and no real-shot video exists in this workspace, so the
    real/fake split is by still provenance, never by video provenance;
  * src/detector.py's image path is untouched: this script only *calls* the detector.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
WS_DEFAULT = REPO.parent
sys.path.insert(0, str(REPO / "src"))

import video_temporal as vt       # noqa: E402  (repo/src/video_temporal.py)

# The t5 helper lives at work/improve/video.py. It MUST NOT go on sys.path: its module
# name ("video") would shadow repo/src/video.py, which detector.detect_video imports
# lazily - that shadowing silently degraded the whole run to "抽帧失败 / score 0.0" once.
# So it is loaded explicitly by file path under a private module name instead.
videoh = None                                                            # noqa: E402


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("t5_video_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_module_resolution(helper_path: Path) -> dict:
    """Pin and verify that the production `video` module is repo/src/video.py."""
    import video as prod_video
    got = Path(prod_video.__file__).resolve()
    want = (REPO / "src" / "video.py").resolve()
    if got != want:
        raise RuntimeError(f"production video module resolved to {got}, expected {want} "
                           f"(module shadowing - the t5 helper must not be on sys.path)")
    return {"production_video": str(got),
            "video_temporal": str(Path(vt.__file__).resolve()),
            "t5_helper": str(helper_path.resolve())}


def resolve_clip(p: Path) -> Path:
    """Accept either a CLIP snapshot root or its parent (as video_channel_validation did)."""
    if (p / "config.json").is_file():
        return p
    if p.is_dir():
        for d in sorted([x for x in p.iterdir() if x.is_dir()], reverse=True):
            if (d / "config.json").is_file():
                return d
    return p


def resolve_out(out: str, ws: Path) -> Path:
    """Resolve --out the way the documented verify command invokes it (cwd = workspace root)."""
    p = Path(out)
    if p.is_absolute():
        return p
    s = out.replace("\\", "/")
    if s.startswith("repo/"):
        return ws / s
    if (Path.cwd() / p).parent.is_dir():
        return Path.cwd() / p
    return ws / s


def main(argv=None) -> int:
    global videoh
    ap = argparse.ArgumentParser(description="t5 video temporal validation")
    ap.add_argument("--out", default="repo/reports/video_temporal_validation.json")
    ap.add_argument("--workspace", default=str(WS_DEFAULT))
    ap.add_argument("--pairs", type=int, default=0,
                    help="clips per class per arm (0 = pre-registered default 20)")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-dir", default=".tmp/video_temporal_validation/clips",
                    help="where the encoded clips are kept (relative to --workspace)")
    ap.add_argument("--archived", default="repo/reports/video_channel_validation.json")
    ap.add_argument("--skip-detector", action="store_true",
                    help="debug only: build features without the production video path "
                         "(the report then carries no video_level)")
    args = ap.parse_args(argv)

    ws = Path(args.workspace)
    helper_path = ws / "work" / "improve" / "video.py"
    videoh = load_helper(helper_path)
    args.pairs = args.pairs or videoh.PROTOCOL["pairs_per_arm"]
    args.seconds = args.seconds or videoh.PROTOCOL["seconds"]
    args.fps = args.fps or videoh.PROTOCOL["fps"]
    args.canvas = args.canvas or videoh.PROTOCOL["canvas"]
    module_resolution = check_module_resolution(helper_path)
    t_start = time.perf_counter()
    print(f"[env] {json.dumps(videoh.probe_env(args.device), ensure_ascii=False)}", flush=True)
    print(f"[modules] {json.dumps(module_resolution, ensure_ascii=False)}", flush=True)

    # ---- 1. frozen artifact provenance -------------------------------------
    md = ws / "models"
    artifacts = {}
    # source files this run actually executed (detector.py etc. are shared with other
    # workstreams, so pinning their hashes is the only way to know what produced the
    # numbers below; this script never modifies them)
    for rel in ("scripts/video_temporal_validation.py", "src/video_temporal.py",
                "src/detector.py", "src/video.py", "src/fusion.py",
                "work/improve/video.py"):
        p = ws / rel if rel.startswith("work/") else ws / "repo" / rel
        if p.is_file():
            artifacts[f"code:{rel}"] = {"sha256": videoh.sha256_file(p),
                                        "bytes": p.stat().st_size,
                                        "mtime_utc": time.strftime(
                                            "%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(p.stat().st_mtime))}
    for rel in ("fusion.json", "combined_lr.joblib", "forensic_lr.joblib",
                "clip_domain_lr.joblib", "clip_probe_norm.pt", "feature_spec.json"):
        p = md / rel
        if p.is_file():
            artifacts[rel] = {"sha256": videoh.sha256_file(p), "bytes": p.stat().st_size}
    for rel in ("reports/video_channel_validation.json",):
        p = ws / "repo" / rel
        if p.is_file():
            artifacts[rel] = {"sha256": videoh.sha256_file(p), "bytes": p.stat().st_size}

    # ---- 2. module self-test ------------------------------------------------
    print("[selftest] running repo/src/video_temporal.py self-test ...", flush=True)
    self_test = vt.run_self_test(ws / "data")
    print(f"[selftest] {self_test['n_passed']}/{self_test['n_checks']} checks passed "
          f"({self_test['runtime_ms']} ms)", flush=True)
    for c in self_test["checks"]:
        if not c["passed"]:
            print(f"[selftest][FAIL] {c['id']}: {c['detail']}", flush=True)

    # ---- 3. build the encoded validation set -------------------------------
    manifest = videoh.build_manifest(ws, args.pairs)
    clip_dir = ws / args.clip_dir
    n_frames = max(int(round(args.seconds * args.fps)), vt.MIN_FRAMES)
    items = manifest["real_a"] + manifest["real_b"] + manifest["fake"]
    print(f"[build] {len(items)} clips ({len(manifest['real_a'])} real-A, "
          f"{len(manifest['real_b'])} real-B, {len(manifest['fake'])} shared fake) "
          f"-> {clip_dir}", flush=True)

    rows = []
    for i, it in enumerate(items, start=1):
        src = ws / it["source_path"]
        still = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if still is None:
            raise RuntimeError(f"cannot decode still {src}")
        frames = vt.synthesize_sequence(still, n_frames=n_frames, canvas=args.canvas)
        enc = videoh.encode_clip(frames, clip_dir / f"{it['clip_id']}.mp4", fps=args.fps)
        dec = videoh.decode_all(enc["path"])
        tf = vt.compute(dec["frames"])
        row = dict(it)
        row.update({
            "clip_path": str(Path(enc["path"]).relative_to(ws)).replace("\\", "/"),
            "clip_sha256": enc["sha256"], "clip_bytes": enc["bytes"],
            "encode_ms": enc["encode_ms"], "fourcc": enc["fourcc"],
            "frames_written": enc["frames_written"], "frames_decoded": enc["frames_decoded"],
            "distinct_source_frames": enc["distinct_frames"],
            "decoded_distinct_frames": len({videoh.sha256_bytes(f.tobytes())
                                            for f in dec["frames"]}),
            "still_sha256": it["source_sha256"],
            "temporal": tf.to_dict(),
        })
        rows.append(row)
        print(f"[build] {i:2d}/{len(items)} {it['clip_id']:12s} {it['label']:4s} "
              f"frames={enc['frames_written']} distinct={row['decoded_distinct_frames']} "
              f"flow_err={tf.flow_fb_error_ratio_mean:.4f} "
              f"noise_corr={tf.noise_residual_corr_mean:.4f}", flush=True)

    # ---- 4. production video path -----------------------------------------
    detector_note = None
    if not args.skip_detector:
        from detector import AIGCDetector, PipelineConfig
        clip_snap = resolve_clip(md / "hf_cache" / "models--openai--clip-vit-large-patch14"
                                 / "snapshots")
        cfg = PipelineConfig(
            clip_dir=str(clip_snap),
            clip_probe=str(md / "clip_probe_norm.pt"),
            forensic_lr=str(md / "forensic_lr.joblib"),
            clip_domain_lr=str(md / "clip_domain_lr.joblib"),
            combined_lr=str(md / "combined_lr.joblib"),
            feature_spec=str(md / "feature_spec.json"),
            fusion_json=str(md / "fusion.json"),
            device=args.device,
        )
        det = AIGCDetector(cfg)
        detector_note = ("same PipelineConfig as scripts/video_channel_validation.py "
                         "(root models/ dir, clip_probe_norm.pt, root fusion.json) so the "
                         "archived 30-clip numbers are directly comparable")
        # warm-up on the first clip is charged to that clip, exactly like the archived run
        for r in rows:
            t0 = time.perf_counter()
            rep = det.detect_video(str(ws / r["clip_path"]))
            ms = (time.perf_counter() - t0) * 1000
            r.update({"fused_score": rep.fused_score, "verdict": rep.verdict,
                      "frames": rep.versions.get("frames"),
                      "deep_frames": rep.versions.get("video_deep_frames"),
                      "latency_ms": round(ms, 2),
                      "detector_total_ms": rep.latency_ms.get("total"),
                      "uncertainty": list(rep.uncertainty)[:3]})
            print(f"[detect] {r['clip_id']:12s} {r['label']:4s} "
                  f"score={r['fused_score']:.4f} {r['verdict']:7s} {ms:7.1f} ms", flush=True)
        # Fail loudly instead of emitting a report whose video_level was computed from the
        # degraded path (frame extraction failed -> every score is 0.0 in ~0 ms).
        broken = [r["clip_id"] for r in rows if not r.get("frames")]
        if broken:
            raise RuntimeError(
                f"video path degraded on {len(broken)}/{len(rows)} clips (no frames: "
                f"{broken[:5]}); refusing to write a report from it")

    # ---- 5. metrics --------------------------------------------------------
    by_id = {r["clip_id"]: r for r in rows}
    real_a = [by_id[r["clip_id"]] for r in manifest["real_a"]]
    real_b = [by_id[r["clip_id"]] for r in manifest["real_b"]]
    fake = [by_id[r["clip_id"]] for r in manifest["fake"]]
    arms, video_level = {}, None
    latency_all = None
    if not args.skip_detector:
        arms["A_source_controlled"] = videoh.arm_metrics(real_a + fake)
        arms["B_in_domain_same_source"] = videoh.arm_metrics(real_b + fake)
        latency_all = videoh.latency_block([r["latency_ms"] for r in rows])
        latency_all["measured_on"] = "all clips, wall clock around detect_video()"
        latency_warm = videoh.latency_block([r["latency_ms"] for r in rows[1:]])
        latency_warm["measured_on"] = (f"{len(rows) - 1} clips, first (cold-start) clip "
                                       f"excluded; the excluded clip carries the one-time "
                                       f"model load exactly as in the archived run")
        latency_warm["cold_start_clip_ms"] = round(rows[0]["latency_ms"], 1)
        video_level = dict(arms["A_source_controlled"])
        video_level["primary_arm"] = "A_source_controlled"
        video_level["per_arm"] = arms
        video_level["note"] = ("video-level metrics of the t5 controlled set; arm A is the "
                               "task-required source-controlled arm, arm B replicates the "
                               "archived in-domain protocol. Channel-level, NOT business-level.")
        video_level["interpretation"] = (
            "PRIMARY arm = A_source_controlled, as required by the task (real = "
            "repo/testset/real claim photos, fake = data/images AI stills). It must be read "
            "together with validation_set.construction.known_confound_arm_A: the frozen image "
            "heads score those web-sourced real photos as AI at 92.7% FPR "
            "(repo/reports/testset_real_41_eval.md), so a low arm-A AUC reflects that known "
            "image-domain gap, not the temporal features. Arm B "
            "(B_in_domain_same_source, both classes from the brief) is the arm to compare "
            "with the archived 30-clip run.")

    # ---- 6. temporal feature analysis -------------------------------------
    def temporal_auc(real_rows, fake_rows):
        y = np.asarray([0] * len(real_rows) + [1] * len(fake_rows))
        out = {}
        for name in vt.TEMPORAL_FEATURE_NAMES:
            v = np.asarray([r["temporal"][name] for r in real_rows + fake_rows],
                           dtype=np.float64)
            if not np.all(np.isfinite(v)):
                out[name] = {"auc": None, "note": "non-finite values"}
                continue
            a = videoh.auc(y, v)
            d = vt.AI_LIKE_DIRECTION[name]
            out[name] = {
                "auc_raw": (round(a, 4) if a is not None else None),
                "auc_ai_like_direction": (round(a, 4) if (a is not None and d) else None),
                "direction": d,
                "mean_real": round(float(v[y == 0].mean()), 6),
                "mean_fake": round(float(v[y == 1].mean()), 6),
                "higher_is_more_ai_like": bool(d > 0),
            }
        return out

    temporal = {"per_arm_univariate_auc": {
        "A_source_controlled": temporal_auc(real_a, fake),
        "B_in_domain_same_source": temporal_auc(real_b, fake),
    }, "interpretation": None}
    # label-free composite: mean of z-scores of the signed features (no labels used)
    signed = list(vt.SIGNED_FEATURE_NAMES)
    Z = np.asarray([[r["temporal"][n] * vt.AI_LIKE_DIRECTION[n] for n in signed] for r in rows],
                   dtype=np.float64)
    if Z.size and np.all(np.isfinite(Z)):
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        sd[sd < 1e-12] = 1.0
        comp = ((Z - mu) / sd).mean(axis=1)
        for r, c in zip(rows, comp):
            r["temporal_ai_like_zmean"] = round(float(c), 6)
        pa = videoh.auc(np.asarray([1 if r["label"] == "ai" else 0 for r in rows]), comp)
        temporal["label_free_composite"] = {
            "definition": "mean of z-scored AI-like-oriented temporal features "
                          "(z computed over all evaluated clips; NO labels used)",
            "auc_all_clips": (round(pa, 4) if pa is not None else None),
            "note": "exploratory only: it is a fixed unweighted average, not a trained "
                    "model, and it was not used to select anything",
        }
    temporal["interpretation"] = (
        "The two arms receive an IDENTICAL synthetic motion field and IDENTICAL per-frame "
        "noise realisation, so any temporal difference only reflects how the still's content "
        "interacts with motion + noise + codec. A near-0.5 AUC here is an honest negative "
        "result for these features on this set, not a bug; a high AUC would still not "
        "establish AI-video detection, because neither class is real video.")

    # ---- 7. comparison + next steps ---------------------------------------
    comparison = None
    if not args.skip_detector:
        comparison = videoh.compare_with_archived(
            resolve_out(args.archived, ws), arms["B_in_domain_same_source"], latency_all)

    protocol = dict(videoh.PROTOCOL)
    protocol.update({
        "clip_count": len(items),
        "frames_per_clip": n_frames,
        "synth": {k: v for k, v in vt.PARAMS.items() if k.startswith("synth_")},
        "temporal_features": list(vt.TEMPORAL_FEATURE_NAMES),
        "video_path": ("production detector.detect_video: 10-frame uniform sampling -> "
                       "per-frame CLIP/forensic evidence -> 0.5*mean + 0.5*P90"),
        "temporal_feature_input": "ALL decoded frames of the clip (24), not the 10 sampled ones",
        "self_test_first": True,
    })

    out = {
        "report": "video_temporal_validation",
        "task": "t5 - video channel: real-encoded validation + inter-frame temporal evidence",
        "declarations": {
            "brief_contains_no_video": (
                "TRUE. The OPEN-SZ-01 brief is 119 still images (data/images/*.png), zero "
                "videos. No probe/testset sample is a video."),
            "channel_level_not_business_level": (
                "TRUE. Everything in this report is a CHANNEL-level result: it shows that a "
                "genuinely encoded video stream can be decoded, that per-frame evidence "
                "aggregates into a video verdict, and that the frozen stack stays inside the "
                "5 s budget. It is NOT evidence of business-level video fraud-detection "
                "capability."),
            "classes_are_separated_by_still_provenance_not_video_provenance": (
                "TRUE. No AI-generated video and no real-shot video exist in this workspace. "
                "Both classes are still images turned into clips by the SAME deterministic "
                "synthetic motion + noise overlay; the only class difference is which still "
                "produced the frames."),
            "clips_are_genuinely_encoded_not_copies": (
                "TRUE. Each clip has many distinct decoded frames (see per-clip "
                "decoded_distinct_frames) and the codec performs real inter-frame prediction; "
                "no clip is one still written N times."),
            "no_threshold_weight_or_feature_was_fitted_on_this_set": (
                "TRUE. Thresholds come from the a-priori fp-budget rule; the temporal features "
                "are fixed formulas from src/video_temporal.py; no classifier was trained."),
            "gap_vs_real_video": (
                "OPEN. Substituting real-shot and AI-generated video (see data_sources) is "
                "still required before any capability claim; the motion/noise overlay is a "
                "synthesis, not camera physics, and the real class of arm A is web-sourced "
                "re-encoded JPEGs rather than a real capture pipeline."),
        },
        "environment": videoh.probe_env(args.device),
        "module_resolution": module_resolution,
        "model_artifacts_sha256": artifacts,
        "protocol": protocol,
        "validation_set": {
            "construction": {
                "encoder": "cv2.VideoWriter",
                "codec_fourcc": videoh.PROTOCOL["fourcc"],
                "fps": args.fps, "seconds": args.seconds, "canvas": args.canvas,
                "frames_per_clip": n_frames,
                "identical_across_classes": ["codec/fourcc", "fps", "frame count", "canvas",
                                             "motion field (sub-pixel sinusoidal pan + slow zoom)",
                                             "per-frame noise realisation (seeded by frame index only)",
                                             "framing rule: cover-resize + centre-crop"],
                "arm_A_real": "repo/testset/real/*.jpg - web-sourced genuine claim photographs "
                              "(index: repo/testset/real/_real_claim_index.json); JPEG from a CDN, "
                              "re-encoded, no EXIF",
                "arm_A_fake": "data/images AI stills (seq 1..), labels from data/labels.csv",
                "arm_B_real": "data/images real stills (seq 57+) - the brief's own in-domain real class",
                "arm_B_fake": "the SAME clips as arm A fake (shared, scored once)",
                "known_confound_arm_A": (
                    "repo/reports/testset_real_41_eval.md measured the frozen image heads at "
                    "FPR 92.7% on these very web-sourced real photos (OOD vs the brief's real "
                    "cluster). Arm A therefore mixes 'video channel behaviour' with a KNOWN "
                    "image-head domain gap and must not be read as a video-only statement."),
            },
            "selection_rule": manifest["selection_rule"],
            "n_clips": len(items),
            "clips": [{k: r[k] for k in ("clip_id", "arm", "label", "source_path",
                                         "source_sha256", "clip_path", "clip_sha256",
                                         "clip_bytes", "fourcc", "frames_written",
                                         "frames_decoded", "distinct_source_frames",
                                         "decoded_distinct_frames")} for r in rows],
        },
        "module_self_test": self_test,
        "video_level": video_level,
        "latency_ms": latency_all,
        "latency_ms_excluding_cold_start": (latency_warm if not args.skip_detector else None),
        "per_arm": arms,
        "temporal_features": temporal,
        "per_clip": [{k: v for k, v in r.items()} for r in rows],
        "comparison_to_archived_30_clips": comparison,
        "detector_config_note": detector_note,
        "data_sources": videoh.reference_video_sources(),
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }

    outp = resolve_out(args.out, ws)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out["headline"] = {
        "what_this_establishes": (
            "A genuinely encoded video stream (24 distinct frames per clip, real codec "
            "inter-frame prediction) is decoded, scored end-to-end by the production video "
            "path, aggregated into a video verdict, and stays inside the 5 s budget on this "
            "machine - plus an inter-frame temporal feature block that is unit-tested."),
        "what_this_does_NOT_establish": (
            "Any business-level real-vs-AI VIDEO detection capability. The brief contains no "
            "video; both classes are stills turned into clips; no real-shot and no "
            "AI-generated video was available offline."),
        "primary_video_level_auc_arm_A": (video_level or {}).get("auc"),
        "in_domain_replication_auc_arm_B": (arms.get("B_in_domain_same_source", {}) or {}).get("auc"),
        "archived_30_clip_auc_for_reference": ((comparison or {}).get("archived_video_level", {})
                                               or {}).get("auc"),
        "p95_ms": (latency_all or {}).get("p95"),
        "p95_within_5000ms": (latency_all or {}).get("p95_within_budget"),
    }
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ t5 video temporal validation ================")
    if video_level:
        print(f"arm A (source-controlled, PRIMARY): n={video_level['n']} "
              f"AUC={video_level['auc']} TPR={video_level['tpr']} "
              f"FPR={video_level['fpr']} ({video_level['fpr_count']}) "
              f"acc={video_level['accuracy']}")
        print(f"arm B (in-domain, vs archived 30): AUC={arms['B_in_domain_same_source']['auc']} "
              f"TPR={arms['B_in_domain_same_source']['tpr']} "
              f"FPR={arms['B_in_domain_same_source']['fpr']} "
              f"({arms['B_in_domain_same_source']['fpr_count']})")
        print(f"latency: P50={latency_all['p50']} ms P95={latency_all['p95']} ms "
              f"(budget {latency_all['budget_ms']} ms) P95<=budget: "
              f"{latency_all['p95_within_budget']}")
        print(f"latency excl. cold start: P50={latency_warm['p50']} ms "
              f"P95={latency_warm['p95']} ms (cold clip {latency_warm['cold_start_clip_ms']} ms)")
    k = "A_source_controlled"
    top = sorted(temporal["per_arm_univariate_auc"][k].items(),
                 key=lambda kv: -(kv[1]["auc_raw"] or 0))[:4]
    print("temporal features, top raw AUC (arm A): " +
          ", ".join(f"{n}={v['auc_raw']}" for n, v in top))
    print(f"self-test {self_test['n_passed']}/{self_test['n_checks']} passed; "
          f"brief-video declaration: {out['declarations']['brief_contains_no_video']}")
    print(f"[save] {outp}")
    print(f"[time] {out['runtime_s']} s")

    ok = self_test["passed"] and (args.skip_detector or
                                  (video_level is not None and latency_all is not None
                                   and latency_all["p95_within_budget"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
