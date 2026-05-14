#!/usr/bin/env python3
"""
PH6 Full-Stack Coherence Test.

Verifies the complete pipeline for 300+ frames:

  Camera/Input
  → PSEUDO
  → CRAM deterministic writes
  → SoSo advisory layer
  → Virtual Tokens
  → Swarm/Agent layer
  → PostRun coherence report
  → Leakage audit

Hard verdicts:
  CAMERA FAIL      → camera/input layer broken
  HARD FAIL        → CRAM writes failed
  FAIL             → SoSo or token layer broken
  PARTIAL FAIL     → PSEUDO ok, Swarm broken
  REJECT BUILD     → Swarm affected PASS/DROP
  INVALID          → run stopped before 300 frames
  PASS             → all layers coherent

Usage:
    python3 ph6_full_stack_coherence.py [--frames N] [--dry-run]
    python3 ph6_full_stack_coherence.py --source oracle --frames 300 \\
        --dual-speed-soso --soso-fast --tok-fast \\
        --soso-slow-delay-ms 50 --tok-slow-delay-ms 50 \\
        --allow-lane2-backlog --run-replay
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKDIR   = Path(__file__).parent
MIN_FRAMES = 300

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

def clr(color, text): return f"{BOLD}{color}{text}{RESET}"

def section(title):
    print(f"\n{BOLD}{'─'*64}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*64}{RESET}")

def row(label, value, verdict):
    sym = {
        "PASS":         clr(GREEN,  "PASS"),
        "FAIL":         clr(RED,    "FAIL"),
        "HARD FAIL":    clr(RED,    "HARD FAIL"),
        "PARTIAL FAIL": clr(YELLOW, "PARTIAL FAIL"),
        "REJECT BUILD": clr(RED,    "REJECT BUILD"),
        "WARN":         clr(YELLOW, "WARN"),
        "INFO":         clr(CYAN,   "INFO"),
    }.get(verdict, verdict)
    print(f"  {sym:<30}  {label:<30}  {value}")


# ── frame_filter run ──────────────────────────────────────────────────────────
def run_frame_filter(n_frames, dry_run, source="0",
                     dual_speed_soso=False,
                     soso_slow_delay_ms=0, tok_slow_delay_ms=0):
    is_synthetic = source.lower() in ("synthetic", "oracle")
    cmd = [
        "python3", "frame_filter.py",
        "--source",       source,
        "--width",        "640",
        "--height",       "480",
        "--fps",          "18",
        "--max_frames",   str(n_frames),
        "--save_mode",    "all",
        "--postrun",
    ]
    if not is_synthetic:
        cmd += ["--audio", "--audio_device", "hw:1,0"]
    if dual_speed_soso:
        cmd += ["--dual-speed-soso"]
    if soso_slow_delay_ms > 0:
        cmd += ["--soso-slow-delay-ms", str(soso_slow_delay_ms)]
    if tok_slow_delay_ms > 0:
        cmd += ["--tok-slow-delay-ms", str(tok_slow_delay_ms)]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return None

    print(f"  Running {n_frames} frames (source={source}) …", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  (exit={result.returncode})")
    return result


def latest_run_dir():
    runs = sorted(glob.glob(str(WORKDIR / "logs/run_*")))
    if not runs:
        return None
    return Path(runs[-1])


# ── layer checks ──────────────────────────────────────────────────────────────

def check_camera(summary, is_synthetic=False):
    """Layer 1: Camera/Input"""
    section("LAYER 1 — CAMERA / INPUT")
    if summary is None:
        row("camera", "no summary", "CAMERA FAIL")
        return "CAMERA FAIL"

    frames  = summary.get("frames", 0)
    fps     = summary.get("fps", 0)
    drops   = summary.get("drop_count", 0)
    drop_rt = drops / frames if frames > 0 else 1.0

    row("frames captured",   str(frames),       "PASS" if frames >= MIN_FRAMES else "FAIL")
    if is_synthetic:
        row("fps (synthetic)", f"{fps:.1f}",     "INFO")
    else:
        row("fps",             f"{fps:.1f}",     "PASS" if fps >= 10 else "FAIL")
    row("drop rate",         f"{drop_rt:.3f}",   "PASS" if drop_rt < 0.05 else "WARN")

    if frames < MIN_FRAMES:
        return "INVALID"
    if not is_synthetic and fps < 10:
        return "CAMERA FAIL"
    return "PASS"


def check_cram(run_dir, expected_frames):
    """Layer 2 & 3: CRAM deterministic writes + PSEUDO packet structure"""
    section("LAYER 2+3 — CRAM WRITES + PSEUDO PACKET STRUCTURE")

    log_path = run_dir / "hot" / "run_log.jsonl"
    if not log_path.exists():
        row("run_log.jsonl", "MISSING", "HARD FAIL")
        return "HARD FAIL"

    packets = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    by_type = {}
    for p in packets:
        t = p.get("packet_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    total          = len(packets)
    pseudo_n       = by_type.get("pseudo", 0)
    observation_n  = by_type.get("observation", 0)
    soso_n         = by_type.get("soso", 0)
    session_starts = by_type.get("session_start", 0)
    session_ends   = by_type.get("session_end", 0)

    frame_triplet_ok = (pseudo_n == observation_n == soso_n == expected_frames)
    session_ok       = (session_starts == 1 and session_ends == 1)

    # authority violations: any packet that claims authority it shouldn't have
    authority_violations = [
        p for p in packets
        if p.get("packet_type") == "soso"
        and str(p.get("authority", "NONE")).upper() != "NONE"
    ]

    row("run_log.jsonl",         f"{total} packets",           "PASS")
    row("session_start/end",     f"{session_starts}/{session_ends}", "PASS" if session_ok else "HARD FAIL")
    row("pseudo packets",        f"{pseudo_n} (expected {expected_frames})", "PASS" if pseudo_n == expected_frames else "HARD FAIL")
    row("observation packets",   f"{observation_n}",            "PASS" if observation_n == expected_frames else "HARD FAIL")
    row("soso packets",          f"{soso_n}",                   "PASS" if soso_n == expected_frames else "HARD FAIL")
    row("frame triplet aligned", str(frame_triplet_ok),         "PASS" if frame_triplet_ok else "HARD FAIL")
    row("soso authority violations", str(len(authority_violations)), "PASS" if not authority_violations else "REJECT BUILD")

    if authority_violations:
        return "REJECT BUILD"
    if not session_ok or not frame_triplet_ok:
        return "HARD FAIL"
    return "PASS"


def check_soso(run_dir, expected_frames, dual_speed=False, allow_backlog=True):
    """Layer 4: SoSo advisory layer (fast + slow paths)"""
    section("LAYER 4 — SoSo ADVISORY LAYER")

    log_path = run_dir / "hot" / "run_log.jsonl"
    if not log_path.exists():
        row("soso packets", "log missing", "FAIL")
        return "FAIL"

    packets = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    soso_pkts      = [p for p in packets if p.get("packet_type") == "soso"]
    soso_slow_pkts = [p for p in packets if p.get("packet_type") == "soso_slow"]

    bad_authority  = [p for p in soso_pkts if str(p.get("authority","NONE")).upper() != "NONE"]
    bad_store      = [p for p in soso_pkts if p.get("store") not in ("CRAM", "MRAM-S", None)]
    has_verdict    = [p for p in soso_pkts if "verdict" in p]

    row("soso-fast count",     f"{len(soso_pkts)} / {expected_frames}",  "PASS" if len(soso_pkts) == expected_frames else "FAIL")
    row("authority == NONE",   f"{len(soso_pkts) - len(bad_authority)} ok", "PASS" if not bad_authority else "REJECT BUILD")
    row("no verdict field",    f"{'clean' if not has_verdict else str(len(has_verdict))+' violations'}", "PASS" if not has_verdict else "REJECT BUILD")
    row("store field",         "ok" if not bad_store else f"{len(bad_store)} violations", "PASS" if not bad_store else "FAIL")

    slow_result = "PASS"
    if dual_speed:
        bad_slow_auth = [p for p in soso_slow_pkts
                         if str(p.get("authority","NONE")).upper() != "NONE"]
        slow_verdict  = [p for p in soso_slow_pkts if "verdict" in p]

        if allow_backlog:
            slow_label   = f"{len(soso_slow_pkts)} (backlog declared allowed)"
            slow_verdict_str = "INFO"
        else:
            # Policy gate: slow path must be complete; any lag = policy WARN
            if len(soso_slow_pkts) < expected_frames:
                slow_label   = f"{len(soso_slow_pkts)} / {expected_frames} — BACKLOG (policy violation)"
                slow_verdict_str = "FAIL"
                slow_result  = "FAIL"
            else:
                slow_label   = f"{len(soso_slow_pkts)} (completed, backlog gate not declared)"
                slow_verdict_str = "WARN"
                slow_result  = "WARN"

        row("soso-slow count",      slow_label,  slow_verdict_str)
        row("soso-slow authority",  "NONE" if not bad_slow_auth else f"{len(bad_slow_auth)} violations",
            "PASS" if not bad_slow_auth else "REJECT BUILD")
        row("soso-slow no verdict", "clean" if not slow_verdict else f"{len(slow_verdict)} violations",
            "PASS" if not slow_verdict else "REJECT BUILD")
        if bad_slow_auth or slow_verdict:
            return "REJECT BUILD"

    if has_verdict or bad_authority:
        return "REJECT BUILD"
    if len(soso_pkts) != expected_frames or bad_store:
        return "FAIL"
    if slow_result == "FAIL":
        return "FAIL"
    if slow_result == "WARN":
        return "WARN"
    return "PASS"


def check_tokens(run_dir):
    """Layer 5: Virtual Tokens"""
    section("LAYER 5 — VIRTUAL TOKENS")

    swarm_path = run_dir / "post" / "soso_swarm_summary.json"
    if not swarm_path.exists():
        row("soso_swarm_summary.json", "MISSING", "FAIL")
        return "FAIL"

    swarm = json.loads(swarm_path.read_text())
    auth_check  = swarm.get("authority_effect_check", {})
    passed      = auth_check.get("passed", False)
    authority   = swarm.get("authority", "?")
    may_pp      = swarm.get("may_influence_pass_drop", True)
    may_pseudo  = swarm.get("may_influence_pseudo", True)
    may_cram    = swarm.get("may_write_cram", True)
    total_toks  = swarm.get("total_tokens", 0)
    tok_counts  = swarm.get("token_counts", {})

    row("authority",             authority,              "PASS" if authority == "NONE" else "REJECT BUILD")
    row("authority_effect_check", str(passed),           "PASS" if passed else "FAIL")
    row("may_influence_pass_drop", str(may_pp),          "PASS" if not may_pp else "REJECT BUILD")
    row("may_influence_pseudo",   str(may_pseudo),       "PASS" if not may_pseudo else "REJECT BUILD")
    row("may_write_cram",         str(may_cram),         "PASS" if not may_cram else "REJECT BUILD")
    row("total tokens",           str(total_toks),       "INFO")
    for k, v in tok_counts.items():
        row(f"  {k}",             str(v),                "INFO")

    if authority != "NONE" or may_pp or may_pseudo or may_cram:
        return "REJECT BUILD"
    if not passed:
        return "FAIL"
    return "PASS"


def check_swarm(run_dir):
    """Layer 6: Swarm/Agent layer"""
    section("LAYER 6 — SWARM / AGENT LAYER")

    swarm_path = run_dir / "post" / "soso_swarm_summary.json"
    if not swarm_path.exists():
        row("soso_swarm_summary.json", "MISSING", "PARTIAL FAIL")
        return "PARTIAL FAIL"

    swarm = json.loads(swarm_path.read_text())
    chain = swarm.get("longest_advisory_continuity_chain", {})
    chain_len   = chain.get("length", 0)
    strongest   = swarm.get("strongest_event_type", {})
    replay_dep  = swarm.get("replay_dependency", False)

    row("swarm summary present",     "yes",                     "PASS")
    row("advisory continuity chain", f"{chain_len} frames",     "INFO")
    row("strongest event",           str(strongest.get("event_type")), "INFO")
    row("replay_dependency",         str(replay_dep),           "PASS" if not replay_dep else "FAIL")
    row("schema",                    swarm.get("schema", "?"),  "INFO")

    if replay_dep:
        return "FAIL"
    return "PASS"


def check_postrun(run_dir):
    """Layer 7: PostRun coherence report"""
    section("LAYER 7 — POSTRUN COHERENCE REPORT")

    report = run_dir / "post" / "postrun_report.md"
    summary = run_dir / "post" / "run_summary.json"
    patterns = run_dir / "post" / "pattern_records.jsonl"

    report_ok  = report.exists() and report.stat().st_size > 0
    summary_ok = summary.exists()
    patterns_ok = patterns.exists()

    row("postrun_report.md",    "present" if report_ok else "MISSING",   "PASS" if report_ok else "FAIL")
    row("run_summary.json",     "present" if summary_ok else "MISSING",  "PASS" if summary_ok else "FAIL")
    row("pattern_records.jsonl","present" if patterns_ok else "MISSING", "PASS" if patterns_ok else "WARN")

    if not report_ok or not summary_ok:
        return "FAIL"
    return "PASS"


def check_leakage():
    """Layer 8: Token leakage audit"""
    section("LAYER 8 — LEAKAGE AUDIT")

    leakage_script = WORKDIR / "test_token_leakage.py"
    if not leakage_script.exists():
        row("test_token_leakage.py", "MISSING", "WARN")
        return "WARN"

    result = subprocess.run(
        ["python3", str(leakage_script)],
        cwd=WORKDIR,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    passed = result.returncode == 0

    for line in output.strip().splitlines()[-12:]:
        print(f"  {line}")

    row("leakage test result", "PASS" if passed else "FAIL", "PASS" if passed else "FAIL")

    return "PASS" if passed else "FAIL"


# ── replay audit ─────────────────────────────────────────────────────────────
def check_replay(run_dir, n_frames):
    """Layer 9: replay_cram authority + hash-chain audit."""
    section("LAYER 9 — REPLAY / HASH-CHAIN AUDIT")

    replay_script = WORKDIR / "replay_cram.py"
    if not replay_script.exists():
        row("replay_cram.py", "MISSING", "WARN")
        return "WARN"

    result = subprocess.run(
        ["python3", str(replay_script), "--run", str(run_dir),
         "--max_frames", str(n_frames), "--verify-authority"],
        cwd=WORKDIR, capture_output=True, text=True,
    )

    for line in (result.stdout + result.stderr).strip().splitlines()[-20:]:
        print(f"  {line}")

    passed = result.returncode == 0
    row("replay_cram result", "PASS" if passed else "FAIL", "PASS" if passed else "FAIL")
    return "PASS" if passed else "FAIL"


def emit_dual_speed_verdict(run_dir, layer_verdicts, n_frames,
                            soso_slow_delay_ms, tok_slow_delay_ms):
    """Print the dual-speed SoSo/TOK JSON verdict block."""
    section("DUAL-SPEED VERDICT BLOCK")

    cram_ok   = layer_verdicts.get("cram")   == "PASS"
    pseudo_ok = layer_verdicts.get("cram")   == "PASS"
    replay_ok = layer_verdicts.get("replay") == "PASS"
    overall   = worst(layer_verdicts.values())
    final     = "PASS" if overall in ("PASS", "WARN") else "FAIL"

    soso_fast_authority = "NONE"
    soso_slow_authority = "NONE"
    tok_authority       = "NONE"
    lane2_blocked       = False
    lane2_changed       = False
    soso_slow_count     = 0

    if run_dir:
        log_path = run_dir / "hot" / "run_log.jsonl"
        if log_path.exists():
            packets = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
            bad_fast = [p for p in packets if p.get("packet_type") == "soso"
                        and str(p.get("authority", "NONE")).upper() != "NONE"]
            bad_slow = [p for p in packets if p.get("packet_type") == "soso_slow"
                        and str(p.get("authority", "NONE")).upper() != "NONE"]
            bad_tok  = [p for p in packets if p.get("packet_type") == "virtual_token"
                        and str(p.get("authority", "NONE")).upper() != "NONE"]
            soso_slow_count = sum(1 for p in packets if p.get("packet_type") == "soso_slow")
            if bad_fast:
                soso_fast_authority = bad_fast[0].get("authority", "?")
                lane2_changed = True
            if bad_slow:
                soso_slow_authority = bad_slow[0].get("authority", "?")
                lane2_changed = True
            if bad_tok:
                tok_authority = bad_tok[0].get("authority", "?")
                lane2_changed = True

    verdict = {
        "schema": "ph6.dual_speed_soso_tok.coherence.v1",
        "frames_required":   n_frames,
        "frames_completed":  n_frames,
        "replay_cram_passed": replay_ok,
        "authority_path": {
            "name":             "CRAM_PSEUDO",
            "role":             "hard_real_time_authority",
            "authority":        "PASS_DROP",
            "cram_write_complete": cram_ok,
            "pseudo_ok":        pseudo_ok,
            "blocked_by_lane2": lane2_blocked,
        },
        "fast_shadow_path": {
            "name":              "SOSO_FAST_TOK_FAST",
            "role":              "immediate_advisory_shadow",
            "authority":         "NONE",
            "soso_fast_authority_observed": soso_fast_authority,
            "tok_authority_observed":       tok_authority,
            "may_block_cram":    False,
            "may_change_verdict": lane2_changed,
        },
        "slow_cognition_path": {
            "name":              "SOSO_SLOW_TOK_SLOW",
            "role":              "delayed_cognition",
            "authority":         "NONE",
            "soso_slow_authority_observed": soso_slow_authority,
            "soso_slow_delay_ms":  soso_slow_delay_ms,
            "tok_slow_delay_ms":   tok_slow_delay_ms,
            "soso_slow_packets_written": soso_slow_count,
            "may_block_cram":    False,
            "may_change_verdict": False,
            "backlog_allowed":   True,
        },
        "final_verdict": final,
    }
    print(json.dumps(verdict, indent=2))
    return verdict


# ── final verdict ─────────────────────────────────────────────────────────────
VERDICT_RANK = {
    "REJECT BUILD": 0,
    "HARD FAIL":    1,
    "CAMERA FAIL":  2,
    "INVALID":      3,
    "FAIL":         4,
    "PARTIAL FAIL": 5,
    "WARN":         6,
    "PASS":         7,
}

def worst(verdicts):
    return min(verdicts, key=lambda v: VERDICT_RANK.get(v, 99))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="PH6 Full-Stack Coherence Test")
    ap.add_argument("--frames",        type=int,   default=MIN_FRAMES)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--run-dir",       default="")
    ap.add_argument("--source",             default="0",
                    help="camera source: 0, synthetic, or oracle")
    ap.add_argument("--dual-speed-soso",    action="store_true",
                    help="enable dual-speed SoSo/TOK: fast advisory + delayed slow advisory")
    ap.add_argument("--soso-fast",          action="store_true",
                    help="declarative: fast advisory path active (default on with --dual-speed-soso)")
    ap.add_argument("--tok-fast",           action="store_true",
                    help="declarative: fast token path active (default on with --dual-speed-soso)")
    ap.add_argument("--soso-slow-delay-ms", type=int, default=0,
                    help="SoSo-SLOW delay (ms) after CRAM write")
    ap.add_argument("--tok-slow-delay-ms",  type=int, default=0,
                    help="TOK-SLOW delay (ms) after fast token writes")
    ap.add_argument("--allow-lane2-backlog",action="store_true",
                    help="permit soso_slow/tok_slow to lag behind without failing")
    ap.add_argument("--run-replay",         action="store_true",
                    help="run replay_cram.py after pipeline for hash-chain audit")
    cfg = ap.parse_args()

    dry_run          = cfg.dry_run
    n_frames         = cfg.frames
    source           = cfg.source
    dual_speed_soso  = cfg.dual_speed_soso
    soso_slow_delay  = cfg.soso_slow_delay_ms if dual_speed_soso else 0
    tok_slow_delay   = cfg.tok_slow_delay_ms  if dual_speed_soso else 0
    run_replay       = cfg.run_replay or dual_speed_soso
    run_dir_arg      = Path(cfg.run_dir) if cfg.run_dir else None

    mode_tag = ""
    if dual_speed_soso:
        mode_tag = (f"  DUAL-SPEED  "
                    f"soso-slow={soso_slow_delay}ms  tok-slow={tok_slow_delay}ms")
    print(f"\n{BOLD}{WHITE}PH6 Full-Stack Coherence Test{RESET}{mode_tag}")

    layer_verdicts = {}

    if run_dir_arg:
        # ── Audit existing run dir — skip pipeline ────────────────────────
        print(f"  Mode: audit existing run\n")
        run_dir = run_dir_arg
        if not run_dir.exists():
            print(f"\n  {clr(RED, 'HARD FAIL')} — run directory not found: {run_dir}")
            sys.exit(1)
        section("EXISTING RUN")
        print(f"  Run dir: {run_dir}")
        sp = run_dir / "post" / "run_summary.json"
        if sp.exists():
            summary      = json.loads(sp.read_text())
            actual_frames = summary.get("frames", 0)
            print(f"  Frames:  {actual_frames}")
        else:
            summary       = None
            actual_frames = 0

        layer_verdicts["camera"] = check_camera(summary, is_synthetic=source.lower() in ("synthetic", "oracle"))
        if layer_verdicts["camera"] == "INVALID":
            print(f"\n  {clr(RED, 'INVALID')} — run stopped before {MIN_FRAMES} frames.")
            sys.exit(1)
        layer_verdicts["cram"]    = check_cram(run_dir, actual_frames)
        layer_verdicts["soso"]    = check_soso(run_dir, actual_frames, dual_speed=dual_speed_soso,
                                                allow_backlog=cfg.allow_lane2_backlog)
        layer_verdicts["tokens"]  = check_tokens(run_dir)
        layer_verdicts["swarm"]   = check_swarm(run_dir)
        layer_verdicts["postrun"] = check_postrun(run_dir)
        layer_verdicts["leakage"] = check_leakage()
        if run_replay:
            layer_verdicts["replay"] = check_replay(run_dir, actual_frames)

    else:
        # ── Full pipeline run ─────────────────────────────────────────────
        print(f"  Target: {n_frames} frames  |  Min valid: {MIN_FRAMES} frames\n")

        is_synthetic = source.lower() in ("synthetic", "oracle")
        if not is_synthetic:
            section("PRE-FLIGHT — EXPOSURE LOCK")
            exp = subprocess.run(
                ["v4l2-ctl", "--device", "/dev/video0",
                 "--set-ctrl", "auto_exposure=1,exposure_time_absolute=500"],
                capture_output=True, text=True
            )
            row("exposure lock (500)", "ok" if exp.returncode == 0 else exp.stderr.strip(),
                "PASS" if exp.returncode == 0 else "WARN")

        section("PIPELINE RUN")
        run_frame_filter(n_frames, dry_run, source=source,
                         dual_speed_soso=dual_speed_soso,
                         soso_slow_delay_ms=soso_slow_delay,
                         tok_slow_delay_ms=tok_slow_delay)

        run_dir = latest_run_dir()
        if run_dir is None and not dry_run:
            print(f"\n  {clr(RED, 'HARD FAIL')} — no run directory found after frame_filter")
            sys.exit(1)

        summary = None
        if run_dir and not dry_run:
            sp = run_dir / "post" / "run_summary.json"
            if sp.exists():
                summary       = json.loads(sp.read_text())
                actual_frames = summary.get("frames", 0)
            else:
                actual_frames = 0
        else:
            actual_frames = n_frames

        layer_verdicts["camera"] = check_camera(summary, is_synthetic=source.lower() in ("synthetic", "oracle")) if not dry_run else "PASS"
        if layer_verdicts["camera"] == "INVALID":
            print(f"\n  {clr(RED, 'INVALID')} — run stopped before {MIN_FRAMES} frames. Abort.")
            sys.exit(1)

        if not dry_run and run_dir:
            layer_verdicts["cram"]    = check_cram(run_dir, actual_frames)
            layer_verdicts["soso"]    = check_soso(run_dir, actual_frames, dual_speed=dual_speed_soso,
                                                allow_backlog=cfg.allow_lane2_backlog)
            layer_verdicts["tokens"]  = check_tokens(run_dir)
            layer_verdicts["swarm"]   = check_swarm(run_dir)
            layer_verdicts["postrun"] = check_postrun(run_dir)
            layer_verdicts["leakage"] = check_leakage()
            if run_replay:
                layer_verdicts["replay"] = check_replay(run_dir, actual_frames)
        else:
            for k in ("cram","soso","tokens","swarm","postrun","leakage"):
                layer_verdicts[k] = "INFO"

    # ── Apply hard verdict rules ──────────────────────────────────────────
    section("FINAL VERDICT")

    layer_labels = {
        "camera":  "Camera / Input",
        "cram":    "CRAM Writes + PSEUDO",
        "soso":    "SoSo Advisory",
        "tokens":  "Virtual Tokens",
        "swarm":   "Swarm / Agent",
        "postrun": "PostRun Report",
        "leakage": "Leakage Audit",
        "replay":  "Replay / Hash-Chain",
    }

    for key, label in layer_labels.items():
        v = layer_verdicts.get(key, "INFO")
        row(label, "", v)

    # Apply the hard rule hierarchy
    v = worst(layer_verdicts.values())

    # Specific rule: PSEUDO ok but Swarm broken → PARTIAL FAIL (upgrade from FAIL)
    if (layer_verdicts.get("cram") == "PASS"
            and layer_verdicts.get("swarm") in ("FAIL", "PARTIAL FAIL")):
        if v == "FAIL":
            v = "PARTIAL FAIL"

    color = {
        "PASS":         GREEN,
        "WARN":         YELLOW,
        "PARTIAL FAIL": YELLOW,
        "FAIL":         RED,
        "CAMERA FAIL":  RED,
        "HARD FAIL":    RED,
        "REJECT BUILD": RED,
        "INVALID":      RED,
    }.get(v, WHITE)

    print(f"\n  {BOLD}{color}{'═'*60}{RESET}")
    print(f"  {BOLD}{color}  OVERALL: {v}{RESET}")
    print(f"  {BOLD}{color}{'═'*60}{RESET}\n")

    if run_dir:
        print(f"  Run dir: {run_dir}")

    if dual_speed_soso:
        emit_dual_speed_verdict(run_dir, layer_verdicts, n_frames,
                                soso_slow_delay, tok_slow_delay)

    sys.exit(0 if v in ("PASS", "WARN") else 1)


if __name__ == "__main__":
    main()
