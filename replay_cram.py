#!/usr/bin/env python3
"""
PH6 CRAM Replay — authority and coherence audit on an existing run.

Reads run_log.jsonl from a completed run directory and verifies:
  - Packet structure integrity (session_start, triplets, session_end)
  - PSEUDO authority boundaries (no authority leak into SoSo)
  - SoSo advisory-only (authority=NONE, no verdict field)
  - Virtual token boundaries (from soso_swarm_summary.json)
  - Hash chain continuity (prev_hash / packet_hash if present)
  - Minimum frame count

Usage:
    python3 replay_cram.py --run logs/run_<stamp> [--max_frames N] [--verify-authority]

Exit codes:
    0  all checks pass (PASS or WARN)
    1  one or more checks failed
"""

import argparse
import json
import sys
from pathlib import Path

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
    colors = {
        "PASS":         GREEN,
        "FAIL":         RED,
        "HARD FAIL":    RED,
        "REJECT BUILD": RED,
        "WARN":         YELLOW,
        "INFO":         CYAN,
    }
    c = colors.get(verdict, WHITE)
    print(f"  {clr(c, f'{verdict:<14}')}  {label:<32}  {value}")


# ── load packets ──────────────────────────────────────────────────────────────
def load_packets(run_dir):
    log = run_dir / "hot" / "run_log.jsonl"
    if not log.exists():
        return None, f"run_log.jsonl missing: {log}"
    packets = []
    for i, line in enumerate(log.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            packets.append(json.loads(line))
        except json.JSONDecodeError as e:
            return None, f"JSON parse error at line {i+1}: {e}"
    return packets, None


# ── checks ───────────────────────────────────────────────────────────────────

def check_packet_structure(packets, min_frames):
    section("PACKET STRUCTURE")
    by_type = {}
    for p in packets:
        t = p.get("packet_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    total        = len(packets)
    pseudo_n     = by_type.get("pseudo", 0)
    obs_n        = by_type.get("observation", 0)
    soso_n       = by_type.get("soso", 0)
    sess_start   = by_type.get("session_start", 0)
    sess_end     = by_type.get("session_end", 0)
    triplet_ok   = (pseudo_n == obs_n == soso_n)
    session_ok   = (sess_start == 1 and sess_end == 1)
    frames_ok    = pseudo_n >= min_frames

    row("total packets",       str(total),            "INFO")
    row("session_start",       str(sess_start),       "PASS" if sess_start == 1 else "HARD FAIL")
    row("session_end",         str(sess_end),         "PASS" if sess_end == 1 else "HARD FAIL")
    row("pseudo packets",      str(pseudo_n),         "PASS" if frames_ok else "FAIL")
    row("observation packets", str(obs_n),            "PASS" if obs_n == pseudo_n else "HARD FAIL")
    row("soso packets",        str(soso_n),           "PASS" if soso_n == pseudo_n else "HARD FAIL")
    row("triplet aligned",     str(triplet_ok),       "PASS" if triplet_ok else "HARD FAIL")
    row("min frames met",      f"{pseudo_n} >= {min_frames}", "PASS" if frames_ok else "FAIL")

    if not session_ok or not triplet_ok:
        return "HARD FAIL", pseudo_n
    if not frames_ok:
        return "FAIL", pseudo_n
    return "PASS", pseudo_n


def check_authority(packets):
    section("AUTHORITY BOUNDARIES")

    soso_pkts  = [p for p in packets if p.get("packet_type") == "soso"]
    bad_auth   = [p for p in soso_pkts if str(p.get("authority", "NONE")).upper() != "NONE"]
    has_verdict= [p for p in soso_pkts if "verdict" in p]
    bad_passdrop = [p for p in soso_pkts if p.get("may_influence_pass_drop") is True]

    # check no soso/token source claims PSEUDO authority
    pseudo_pkts = [p for p in packets if p.get("packet_type") == "pseudo"]
    bad_pseudo  = [p for p in pseudo_pkts
                   if p.get("authority") not in (None, "PSEUDO", "NONE")]

    row("soso authority==NONE",    f"{len(soso_pkts) - len(bad_auth)}/{len(soso_pkts)}", "PASS" if not bad_auth else "REJECT BUILD")
    row("soso has no verdict",     "clean" if not has_verdict else f"{len(has_verdict)} violations", "PASS" if not has_verdict else "REJECT BUILD")
    row("soso may_influence=False",f"{len(bad_passdrop)} violations", "PASS" if not bad_passdrop else "REJECT BUILD")
    row("pseudo authority clean",  "clean" if not bad_pseudo else f"{len(bad_pseudo)} bad", "PASS" if not bad_pseudo else "FAIL")

    if bad_auth or has_verdict or bad_passdrop:
        return "REJECT BUILD"
    return "PASS"


def check_hash_chain(packets):
    section("HASH CHAIN CONTINUITY")

    chained = [p for p in packets if "packet_hash" in p and "prev_hash" in p]
    if not chained:
        row("hash chain", "not present in this run", "INFO")
        return "PASS"

    broken = 0
    prev = None
    for p in chained:
        if prev is not None and p.get("prev_hash") != prev.get("packet_hash"):
            broken += 1
        prev = p

    row("chained packets",  str(len(chained)), "INFO")
    row("broken links",     str(broken),       "PASS" if broken == 0 else "FAIL")
    return "PASS" if broken == 0 else "FAIL"


def check_swarm_summary(run_dir):
    section("SWARM / TOKEN AUTHORITY (soso_swarm_summary.json)")

    path = run_dir / "post" / "soso_swarm_summary.json"
    if not path.exists():
        row("soso_swarm_summary.json", "MISSING", "WARN")
        return "WARN"

    s = json.loads(path.read_text())
    authority  = s.get("authority", "?")
    may_pp     = s.get("may_influence_pass_drop", True)
    may_pseudo = s.get("may_influence_pseudo", True)
    may_cram   = s.get("may_write_cram", True)
    passed     = s.get("authority_effect_check", {}).get("passed", False)

    row("authority",               authority,    "PASS" if authority == "NONE" else "REJECT BUILD")
    row("authority_effect passed", str(passed),  "PASS" if passed else "FAIL")
    row("may_influence_pass_drop", str(may_pp),  "PASS" if not may_pp else "REJECT BUILD")
    row("may_influence_pseudo",    str(may_pseudo), "PASS" if not may_pseudo else "REJECT BUILD")
    row("may_write_cram",          str(may_cram), "PASS" if not may_cram else "REJECT BUILD")

    if authority != "NONE" or may_pp or may_pseudo or may_cram:
        return "REJECT BUILD"
    return "PASS" if passed else "FAIL"


def check_postrun_files(run_dir):
    section("POSTRUN FILES")

    files = {
        "run_summary.json":       run_dir / "post" / "run_summary.json",
        "postrun_report.md":      run_dir / "post" / "postrun_report.md",
        "soso_swarm_summary.json":run_dir / "post" / "soso_swarm_summary.json",
        "pattern_records.jsonl":  run_dir / "post" / "pattern_records.jsonl",
        "spike_events.jsonl":     run_dir / "hot"  / "spike_events.jsonl",
    }

    all_ok = True
    for name, path in files.items():
        exists = path.exists() and path.stat().st_size > 0
        row(name, "present" if exists else "MISSING", "PASS" if exists else "WARN")
        if not exists and name in ("run_summary.json", "postrun_report.md"):
            all_ok = False

    return "PASS" if all_ok else "WARN"


# ── verdict aggregation ───────────────────────────────────────────────────────
RANK = {
    "REJECT BUILD": 0,
    "HARD FAIL":    1,
    "FAIL":         2,
    "WARN":         3,
    "PASS":         4,
    "INFO":         5,
}

def worst(verdicts):
    return min(verdicts, key=lambda v: RANK.get(v, 99))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PH6 CRAM Replay — authority audit")
    ap.add_argument("--run",              required=True, help="path to run directory")
    ap.add_argument("--max_frames",       type=int, default=300, help="minimum expected frames")
    ap.add_argument("--verify-authority", action="store_true",  help="enforce authority checks (default on)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"{clr(RED, 'ERROR:')} run directory not found: {run_dir}")
        sys.exit(1)

    print(f"\n{BOLD}{WHITE}PH6 CRAM Replay — Coherence Audit{RESET}")
    print(f"  Run dir:    {run_dir}")
    print(f"  Min frames: {args.max_frames}\n")

    # Load
    section("LOADING run_log.jsonl")
    packets, err = load_packets(run_dir)
    if packets is None:
        print(f"  {clr(RED, 'HARD FAIL:')} {err}")
        sys.exit(1)
    print(f"  Loaded {len(packets)} packets")

    # Checks
    verdicts = []

    v_struct, frame_count = check_packet_structure(packets, args.max_frames)
    verdicts.append(v_struct)

    verdicts.append(check_authority(packets))
    verdicts.append(check_hash_chain(packets))
    verdicts.append(check_swarm_summary(run_dir))
    verdicts.append(check_postrun_files(run_dir))

    # Final
    overall = worst(verdicts)
    color = {
        "PASS":         GREEN,
        "WARN":         YELLOW,
        "FAIL":         RED,
        "HARD FAIL":    RED,
        "REJECT BUILD": RED,
    }.get(overall, WHITE)

    section("FINAL VERDICT")
    print(f"\n  {BOLD}{color}{'═'*60}{RESET}")
    print(f"  {BOLD}{color}  OVERALL: {overall}  ({frame_count} frames audited){RESET}")
    print(f"  {BOLD}{color}{'═'*60}{RESET}\n")

    sys.exit(0 if overall in ("PASS", "WARN", "INFO") else 1)


if __name__ == "__main__":
    main()
