#!/usr/bin/env python3
"""
PH6 Synthetic Oracle 300 — known-answer pipeline test.

Feeds PH6 a deterministic 300-frame sequence through the REAL pipeline
(PSEUDO evaluator, SoSo, tokens, swarm, CRAM writer, PostRun generator)
and verifies actual outputs against pre-computed expected values.

Oracle windows:
  frames   1– 60   quiet stable         → PASS, no spikes
  frames  61– 90   pre-motion           → approaching threshold
  frames  91–120   strong motion        → SPIKE_MOTION + DROP
  frames 121–150   overlight            → SPIKE_OVERLIGHT + DROP
  frames 151–180   sustained overlight  → SPIKE_OVERLIGHT continues
  frames 181–210   blur / degraded      → SPIKE_BLUR + DROP
  frames 211–240   recovery             → PASS recovering
  frames 241–300   quiet stable         → PASS, token decay

Usage:
    python3 ph6_synthetic_oracle_300.py [--verbose]

Exit codes:
    0  PASS
    1  FAIL or worse
"""

import json
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = Path(__file__).parent

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

def clr(color, text): return f"{BOLD}{color}{text}{RESET}"
def section(t): print(f"\n{BOLD}{'─'*64}{RESET}\n{BOLD}  {t}{RESET}\n{BOLD}{'─'*64}{RESET}")

# ── Oracle specification ───────────────────────────────────────────────────────
# Each entry: (start, end, window_name, required_spikes, forbidden_spikes, required_verdict)
# required_spikes: spike types that MUST appear at least once in [start, end]
# forbidden_spikes: spike types that must NOT appear in [start, end]
# required_verdict: "PASS" or "DROP" — majority verdict in window
ORACLE = [
    (1,   60,  "QUIET (stable)",          [],                  ["SPIKE_MOTION", "SPIKE_SOUND"], "PASS"),
    (61,  90,  "PRE-MOTION (rising)",     [],                  [],                              "PASS"),
    (91,  120, "STRONG MOTION",           ["SPIKE_MOTION"],    [],                              "DROP"),
    (121, 150, "OVERLIGHT",               ["SPIKE_OVERLIGHT"], [],                              "DROP"),
    (151, 180, "SUSTAINED OVERLIGHT",     ["SPIKE_OVERLIGHT"], [],                              "DROP"),
    (181, 210, "BLUR / DEGRADED",         ["SPIKE_BLUR"],      [],                              "DROP"),
    (211, 240, "RECOVERY",                [],                  [],                              "PASS"),
    (241, 300, "QUIET STABLE (decay)",    [],                  ["SPIKE_SOUND"],                 "PASS"),
]

# Global oracle checks (whole run)
ORACLE_GLOBAL = {
    "frames":            300,
    "authority_effect":  False,   # may_influence_pass_drop must be False
    "postrun_complete":  True,
    "cram_written":      True,    # run_log.jsonl must exist and have packets
    "tokens_created":    True,    # RT + VDT count > 0
}


# ── run frame_filter ──────────────────────────────────────────────────────────
def run_pipeline():
    cmd = [
        "python3", "frame_filter.py",
        "--source",     "oracle",
        "--width",      "640",
        "--height",     "480",
        "--fps",        "18",
        "--max_frames", "300",
        "--save_mode",  "all",
        "--postrun",
    ]
    print(f"  Running: {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  Pipeline done in {elapsed:.1f}s  exit={result.returncode}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[-500:]}")
    return result


def latest_run_dir():
    import glob
    runs = sorted(glob.glob(str(WORKDIR / "logs/run_*")))
    return Path(runs[-1]) if runs else None


# ── load artifacts ────────────────────────────────────────────────────────────
def load_artifacts(run_dir):
    log_path    = run_dir / "hot" / "run_log.jsonl"
    spike_path  = run_dir / "hot" / "spike_events.jsonl"
    swarm_path  = run_dir / "post" / "soso_swarm_summary.json"
    summary_path= run_dir / "post" / "run_summary.json"

    packets = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()] if log_path.exists() else []
    spikes  = [json.loads(l) for l in spike_path.read_text().splitlines() if l.strip()] if spike_path.exists() else []
    swarm   = json.loads(swarm_path.read_text()) if swarm_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    # Index pseudo packets by frame_index for fast lookup
    pseudo_by_frame = {p["frame_index"]: p
                       for p in packets if p.get("packet_type") == "pseudo"}

    # Index spike events by frame_index
    spikes_by_frame = {}
    for sp in spikes:
        fi = sp.get("frame_index", 0)
        spikes_by_frame.setdefault(fi, []).append(sp)

    return pseudo_by_frame, spikes_by_frame, swarm, summary, packets


# ── window audit ──────────────────────────────────────────────────────────────
def audit_window(start, end, name, required, forbidden, req_verdict,
                 pseudo_by_frame, spikes_by_frame, verbose):

    found_spikes = set()
    verdicts     = []
    frames_seen  = 0

    for fi in range(start, end + 1):
        pseudo = pseudo_by_frame.get(fi)
        if pseudo:
            verdicts.append(pseudo.get("verdict", "?"))
            frames_seen += 1
        for sp in spikes_by_frame.get(fi, []):
            found_spikes.update(sp.get("spikes", []))

    # majority verdict
    if verdicts:
        pass_count = sum(1 for v in verdicts if v == "PASS")
        drop_count = len(verdicts) - pass_count
        majority = "PASS" if pass_count >= drop_count else "DROP"
    else:
        majority = "?"

    results = []
    window_pass = True

    # check required spikes
    for spike in required:
        ok = spike in found_spikes
        results.append((f"requires {spike}", "present" if ok else "MISSING", ok))
        if not ok:
            window_pass = False

    # check forbidden spikes
    for spike in forbidden:
        bad = spike in found_spikes
        results.append((f"forbids {spike}", "ABSENT" if not bad else "PRESENT (violation)", not bad))
        if bad:
            window_pass = False

    # check verdict
    verdict_ok = (req_verdict == "any" or majority == req_verdict)
    results.append((f"majority verdict", majority, verdict_ok))
    if not verdict_ok:
        window_pass = False

    # frame coverage
    expected_frames = end - start + 1
    cov_ok = frames_seen == expected_frames
    results.append(("frame coverage", f"{frames_seen}/{expected_frames}", cov_ok))
    if not cov_ok:
        window_pass = False

    # print
    wcolor = GREEN if window_pass else RED
    status = "PASS" if window_pass else "FAIL"
    print(f"  {clr(wcolor, f'{status:<6}')}  frames {start:>3}–{end:<3}  {name}")
    if verbose or not window_pass:
        for label, value, ok in results:
            sym = clr(GREEN, "  ✓") if ok else clr(RED, "  ✗")
            print(f"  {sym}  {label:<30}  {value}")
        if found_spikes:
            print(f"       spikes seen: {sorted(found_spikes)}")

    return window_pass


# ── global oracle checks ──────────────────────────────────────────────────────
def audit_global(swarm, summary, packets, run_dir):
    section("GLOBAL ORACLE CHECKS")
    results = {}
    verbose = "--verbose" in sys.argv

    # frames
    actual_frames = summary.get("frames", 0)
    ok = actual_frames == ORACLE_GLOBAL["frames"]
    print(f"  {'PASS' if ok else 'FAIL'}  frames captured: {actual_frames} (expected {ORACLE_GLOBAL['frames']})")
    results["frames"] = ok

    # CRAM written
    log_path = run_dir / "hot" / "run_log.jsonl"
    cram_ok = log_path.exists() and len(packets) > 0
    print(f"  {'PASS' if cram_ok else 'FAIL'}  CRAM packets: {len(packets)}")
    results["cram_written"] = cram_ok

    # tokens created
    tok = swarm.get("token_counts", {})
    rt  = tok.get("RT", 0)
    vdt = tok.get("VDT", 0)
    tok_ok = (rt + vdt) > 0
    print(f"  {'PASS' if tok_ok else 'FAIL'}  tokens: RT={rt} VDT={vdt} VLT={tok.get('VLT',0)}")
    results["tokens_created"] = tok_ok

    # authority
    auth_ok = not swarm.get("may_influence_pass_drop", True)
    auth_effect_ok = swarm.get("authority_effect_check", {}).get("passed", False)
    print(f"  {'PASS' if auth_ok else 'REJECT'}  authority effect: may_influence_pass_drop={swarm.get('may_influence_pass_drop')}")
    print(f"  {'PASS' if auth_effect_ok else 'FAIL'}  authority_effect_check: {auth_effect_ok}")
    results["authority_effect"] = auth_ok and auth_effect_ok

    # postrun complete
    report = run_dir / "post" / "postrun_report.md"
    postrun_ok = report.exists() and report.stat().st_size > 0
    print(f"  {'PASS' if postrun_ok else 'FAIL'}  postrun_report.md: {'present' if postrun_ok else 'MISSING'}")
    results["postrun_complete"] = postrun_ok

    return all(results.values()), results


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    verbose = "--verbose" in sys.argv

    print(f"\n{BOLD}{WHITE}PH6 Synthetic Oracle 300{RESET}")
    print(f"  Deterministic 300-frame known-answer test")
    print(f"  Source: oracle (OracleSyntheticCapture)")
    print(f"  Real pipeline: PSEUDO → SoSo → Tokens → Swarm → CRAM → PostRun\n")

    section("PIPELINE RUN")
    ff = run_pipeline()
    if ff.returncode != 0:
        print(f"\n  {clr(RED, 'HARD FAIL')} — frame_filter exited {ff.returncode}")
        sys.exit(1)

    run_dir = latest_run_dir()
    if run_dir is None:
        print(f"  {clr(RED, 'HARD FAIL')} — no run directory found")
        sys.exit(1)
    print(f"  Run dir: {run_dir}")

    pseudo_by_frame, spikes_by_frame, swarm, summary, packets = load_artifacts(run_dir)

    section("WINDOW-BY-WINDOW ORACLE AUDIT")
    window_results = []
    for start, end, name, required, forbidden, req_verdict in ORACLE:
        ok = audit_window(start, end, name, required, forbidden, req_verdict,
                          pseudo_by_frame, spikes_by_frame, verbose)
        window_results.append(ok)

    global_ok, _ = audit_global(swarm, summary, packets, run_dir)

    # ── final verdict ─────────────────────────────────────────────────────────
    all_windows_pass = all(window_results)
    overall_pass     = all_windows_pass and global_ok

    section("FINAL VERDICT")
    for (start, end, name, *_), ok in zip(ORACLE, window_results):
        color = GREEN if ok else RED
        print(f"  {clr(color, 'PASS' if ok else 'FAIL')}  frames {start:>3}–{end:<3}  {name}")

    color  = GREEN if global_ok else RED
    gcolor = GREEN if overall_pass else RED

    print(f"\n  {'─'*60}")
    print(f"  {clr(color, 'PASS' if global_ok else 'FAIL')}  global oracle checks")
    print()
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"  {BOLD}{gcolor}{'═'*60}{RESET}")
    print(f"  {BOLD}{gcolor}  PH6_SYNTHETIC_ORACLE_300: {verdict}{RESET}")
    print(f"  {BOLD}{gcolor}{'═'*60}{RESET}\n")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
