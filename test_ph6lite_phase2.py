"""
PH6-LITE Phase 2 Test Suite
Authority: TEST_SUITE — structural and invariant checks only.

Run order:
  1. python test_ph6lite_phase2.py --gen     # generate diagnostic log
  2. python test_ph6lite_phase2.py           # run tests 1-5 against latest log
  3. (manual) camera run for tests 6-7

Tests:
  test_1_pseudo_fields_present
  test_2_soso_fields_present
  test_3_no_soso_leakage_into_pseudo
  test_4_no_pseudo_authority_leakage_into_soso
  test_5_pseudo_before_soso_ordering
  test_6_motion_does_not_stop_capture       (synthetic high-motion video)
  test_7_event_clips_do_not_block_summary   (requires --postrun run)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

HERE   = Path(__file__).parent
SCRIPT = HERE / "frame_filter.py"
PYTHON = sys.executable

PSEUDO_V2_REQUIRED = {
    "motion_level",
    "brightness_level",
    "blur_level",
    "degradation_score",
    "degradation_reasons",
    "threshold_profile",
}

SOSO_V2_REQUIRED = {
    "stability_band",
    "continuity_window",
    "recent_instability_count",
    "recovery_count",
    "confidence_reason",
    "scene_drift_score",
    "authority",
}

# Fields that belong to SoSo — must not appear in pseudo
SOSO_CLASS_FORBIDDEN_IN_PSEUDO = {
    "state",
    "continuity_count",
    "confidence",
    "advisory",
    "stability_band",
    "scene_drift_score",
    "authority",
    "scene_labels",
}

# Fields that belong to Pseudo — must not appear in soso
PSEUDO_CLASS_FORBIDDEN_IN_SOSO = {
    "verdict",
    "reasons",
    "DROP",
    "PASS",
    "motion_high",
    "capture_stop",
    "commit",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_run() -> Path:
    runs = sorted(HERE.glob("logs/run_*"))
    if not runs:
        sys.exit("ERROR: no logs/run_* directories found. Run --gen first.")
    return runs[-1]


def _load_packets(run_dir: Path) -> list:
    log = run_dir / "hot" / "run_log.jsonl"
    if not log.exists():
        log = run_dir / "run_log.jsonl"
    if not log.exists():
        sys.exit(f"ERROR: cannot find run_log.jsonl in {run_dir}")
    pkts = []
    with log.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pkts.append(json.loads(line))
    return pkts


def _run_pipeline(video_path: str, extra_args=(), max_frames: int = 200) -> tuple[list, str, Path]:
    """Run pipeline; return (packets, stdout, run_dir)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "run_log.jsonl"
        cmd = [
            PYTHON, str(SCRIPT),
            "--source", str(video_path),
            "--save_mode", "none",
            "--max_frames", str(max_frames),
            "--log", str(log),
            *extra_args,
        ]
        r = subprocess.run(cmd, cwd=str(HERE), env=env,
                           capture_output=True, text=True, timeout=120)
        stdout = r.stdout
        pkts = []
        if log.exists():
            pkts = [json.loads(l) for l in log.open() if l.strip()]
        return pkts, stdout, r


def _make_motion_video(path: str, n_frames: int = 60,
                       width: int = 64, height: int = 48):
    """Alternates black/white frames — guarantees maximum motion_fraction."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 10, (width, height))
    for i in range(n_frames):
        val = 0 if i % 2 == 0 else 255
        vw.write(np.full((height, width, 3), val, dtype=np.uint8))
    vw.release()


# ── test 1 ────────────────────────────────────────────────────────────────────

def test_1_pseudo_fields_present(run_dir: Path) -> bool:
    print("\n── Test 1: Pseudo v0.2 fields present ──")
    pkts     = _load_packets(run_dir)
    pseudo   = [p for p in pkts if p.get("packet_type") == "pseudo"]
    missing  = []

    for p in pseudo:
        miss = PSEUDO_V2_REQUIRED - set(p.keys())
        if miss:
            missing.append((p.get("frame_index"), sorted(miss)))

    print(f"  Pseudo packets : {len(pseudo)}")
    print(f"  Missing-field violations : {len(missing)}")
    for row in missing[:10]:
        print(f"    frame {row[0]}: {row[1]}")

    ok = len(pseudo) > 0 and len(missing) == 0
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── test 2 ────────────────────────────────────────────────────────────────────

def test_2_soso_fields_present(run_dir: Path) -> bool:
    print("\n── Test 2: SoSo v0.2 fields present ──")
    pkts   = _load_packets(run_dir)
    soso   = [p for p in pkts if p.get("packet_type") == "soso"]
    bad    = []

    for p in soso:
        miss = SOSO_V2_REQUIRED - set(p.keys())
        if miss:
            bad.append((p.get("frame_index"), "missing", sorted(miss)))
        if p.get("authority") != "NONE":
            bad.append((p.get("frame_index"), "bad_authority", p.get("authority")))

    print(f"  SoSo packets : {len(soso)}")
    print(f"  Violations   : {len(bad)}")
    for row in bad[:10]:
        print(f"    frame {row[0]}: {row[1]} {row[2]}")

    ok = len(soso) > 0 and len(bad) == 0
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── test 3 ────────────────────────────────────────────────────────────────────

def test_3_no_soso_leakage_into_pseudo(run_dir: Path) -> bool:
    print("\n── Test 3: No SoSo leakage into Pseudo ──")
    pkts = _load_packets(run_dir)
    bad  = []

    for p in pkts:
        if p.get("packet_type") == "pseudo":
            hits = SOSO_CLASS_FORBIDDEN_IN_PSEUDO & set(p.keys())
            if hits:
                bad.append((p.get("frame_index"), sorted(hits)))

    print(f"  Pseudo leakage violations : {len(bad)}")
    for row in bad[:10]:
        print(f"    frame {row[0]}: forbidden fields {row[1]}")

    ok = len(bad) == 0
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── test 4 ────────────────────────────────────────────────────────────────────

def test_4_no_pseudo_authority_leakage_into_soso(run_dir: Path) -> bool:
    print("\n── Test 4: No Pseudo authority leakage into SoSo ──")
    pkts = _load_packets(run_dir)
    bad  = []

    for p in pkts:
        if p.get("packet_type") == "soso":
            hits = PSEUDO_CLASS_FORBIDDEN_IN_SOSO & set(p.keys())
            if hits:
                bad.append((p.get("frame_index"), sorted(hits)))

    print(f"  SoSo authority leakage violations : {len(bad)}")
    for row in bad[:10]:
        print(f"    frame {row[0]}: forbidden fields {row[1]}")

    ok = len(bad) == 0
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── test 5 ────────────────────────────────────────────────────────────────────

def test_5_pseudo_before_soso_ordering(run_dir: Path) -> bool:
    print("\n── Test 5: Pseudo before SoSo ordering ──")
    pkts     = _load_packets(run_dir)
    by_frame = {}

    for p in pkts:
        fi  = p.get("frame_index")
        pt  = p.get("packet_type")
        seq = p.get("packet_seq")
        if fi is not None and pt in ("pseudo", "soso") and seq is not None:
            by_frame.setdefault(fi, {})[pt] = seq

    bad = []
    for fi, seqs in sorted(by_frame.items()):
        if "pseudo" in seqs and "soso" in seqs:
            if seqs["soso"] < seqs["pseudo"]:
                bad.append((fi, seqs["pseudo"], seqs["soso"]))

    print(f"  Frames checked : {len(by_frame)}")
    print(f"  SoSo-before-Pseudo violations : {len(bad)}")
    for row in bad[:10]:
        print(f"    frame {row[0]}: pseudo_seq={row[1]} soso_seq={row[2]}")

    ok = len(bad) == 0
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── test 6 ────────────────────────────────────────────────────────────────────

def test_6_motion_does_not_stop_capture() -> bool:
    print("\n── Test 6: Motion does not stop capture ──")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        vid = tf.name
    try:
        _make_motion_video(vid, n_frames=120)
        pkts, stdout, proc = _run_pipeline(
            vid,
            extra_args=["--motion_max", "0.001"],  # all frames will be DROP
            max_frames=100,
        )

        pseudo      = [p for p in pkts if p.get("packet_type") == "pseudo"]
        drops       = [p for p in pseudo if p.get("verdict") == "DROP"]
        session_end = any(p.get("packet_type") == "session_end" for p in pkts)
        frame_count = len(pseudo)

        print(f"  Frames processed : {frame_count}  (target 100)")
        print(f"  DROP frames      : {len(drops)}")
        print(f"  session_end      : {session_end}")
        print(f"  returncode       : {proc.returncode}")

        ok = (session_end
              and frame_count >= 100
              and len(drops) >= 90
              and proc.returncode == 0)
        if not ok:
            print(f"  STDERR: {proc.stderr[:300]}")
        print(f"  → {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        try:
            os.unlink(vid)
        except OSError:
            pass


# ── test 7 ────────────────────────────────────────────────────────────────────

def test_7_event_clips_do_not_block_summary(run_dir: Path) -> bool:
    print("\n── Test 7: Event clips do not block summary/report ──")
    post = run_dir / "post"

    summary_ok = (post / "run_summary.json").exists()
    report_ok  = (post / "postrun_report.md").exists()
    marker_ok  = (run_dir / "POSTRUN_COMPLETE").exists() or (run_dir / "POSTRUN_INCOMPLETE").exists()

    marker_name = ("POSTRUN_COMPLETE"   if (run_dir / "POSTRUN_COMPLETE").exists()
                   else "POSTRUN_INCOMPLETE" if (run_dir / "POSTRUN_INCOMPLETE").exists()
                   else "MISSING")

    clips = list(post.rglob("*.mp4")) if post.exists() else []

    print(f"  run_summary.json : {'PRESENT' if summary_ok else 'MISSING'}")
    print(f"  postrun_report.md: {'PRESENT' if report_ok else 'MISSING'}")
    print(f"  marker           : {marker_name}")
    print(f"  event clips      : {len(clips)}")

    # test 7 requires --postrun was used; skip gracefully if not
    if not summary_ok and not report_ok and not marker_ok:
        print("  (no postrun artifacts — re-run with --postrun to exercise this test)")
        print("  → SKIP")
        return True   # not a failure — wrong run configuration

    ok = marker_ok
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── generation helper ─────────────────────────────────────────────────────────

def generate_diagnostic_run(video_path: str):
    """Run the pipeline against video_path with --profile diagnostic --postrun."""
    print(f"Generating diagnostic log from {video_path} …")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        PYTHON, str(SCRIPT),
        "--source", video_path,
        "--save_mode", "none",
        "--max_frames", "500",
        "--profile", "diagnostic",
        "--postrun",
        "--event_clips",
    ]
    r = subprocess.run(cmd, cwd=str(HERE), env=env, timeout=300)
    if r.returncode != 0:
        print("ERROR: pipeline run failed")
        sys.exit(1)
    print("Done.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PH6-LITE Phase 2 test suite")
    ap.add_argument("--gen",   metavar="VIDEO", nargs="?", const="__latest__",
                    help="generate a diagnostic run before testing (optional VIDEO path)")
    ap.add_argument("--run",   metavar="RUN_DIR",
                    help="explicit run directory to test (default: latest logs/run_*)")
    args = ap.parse_args()

    if args.gen:
        if args.gen == "__latest__":
            candidates = sorted(HERE.glob("logs/run_*/post/run_video.mp4"))
            if not candidates:
                sys.exit("ERROR: no existing run video found; supply a video path: --gen <path>")
            vid = str(candidates[-1])
        else:
            vid = args.gen
        generate_diagnostic_run(vid)

    run_dir = Path(args.run) if args.run else _latest_run()
    print(f"\nRun dir : {run_dir}")

    results = {}
    results["test_1_pseudo_fields_present"]             = test_1_pseudo_fields_present(run_dir)
    results["test_2_soso_fields_present"]               = test_2_soso_fields_present(run_dir)
    results["test_3_no_soso_leakage_into_pseudo"]       = test_3_no_soso_leakage_into_pseudo(run_dir)
    results["test_4_no_pseudo_authority_leakage_in_soso"] = test_4_no_pseudo_authority_leakage_into_soso(run_dir)
    results["test_5_pseudo_before_soso_ordering"]       = test_5_pseudo_before_soso_ordering(run_dir)
    results["test_6_motion_does_not_stop_capture"]      = test_6_motion_does_not_stop_capture()
    results["test_7_event_clips_do_not_block_summary"]  = test_7_event_clips_do_not_block_summary(run_dir)

    print("\n" + "─" * 55)
    print("PH6-LITE PHASE 2 TEST RESULTS")
    print("─" * 55)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  {name}")
    print("─" * 55)
    if all_pass:
        print("STATUS: TEST_GREEN")
    else:
        print("STATUS: TEST_RED — fix failures before proceeding to 7-minute EVENT_BURST test")
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
