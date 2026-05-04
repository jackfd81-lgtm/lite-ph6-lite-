#!/usr/bin/env python3
"""
CLAW Manual-Gated Calibration Controller.

  ENTER = start recording (10 seconds, then auto-stop)

  After phase result:
    ENTER         = next phase
    R + ENTER     = repeat this phase
"""

import glob
import json
import os
import subprocess
import sys
import termios
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PHASE_SECONDS = 10
PHASE_FRAMES  = int(PHASE_SECONDS * 18)   # 180

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BELL   = "\a"

def clr(color, text):
    return f"{BOLD}{color}{text}{RESET}"

def drain_stdin():
    """Discard any characters buffered in stdin before we start listening for STOP."""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass

def banner(color, title, instruction):
    w = 64
    print(f"\n{BOLD}{color}{'═'*w}{RESET}")
    print(f"{BOLD}{color}  {title}{RESET}")
    print(f"  {instruction}")
    print(f"{BOLD}{color}{'═'*w}{RESET}{BELL}", flush=True)

# ── Phase definitions ─────────────────────────────────────────────────────────
PHASES = [
    {
        "name":    "quiet",
        "color":   GREEN,
        "title":   "PHASE 1 — QUIET",
        "instruction": "Stay completely still. No movement. No sound. No light change.",
        "expected_dominant": None,
        "min_spikes": 0,
        "max_spikes": 5,
    },
    {
        "name":    "camera",
        "color":   CYAN,
        "title":   "PHASE 2 — MOTION",
        "instruction": "Wave your hand or move your body in front of the camera.",
        "expected_dominant": "SPIKE_MOTION",
        "min_spikes": 3,
        "max_spikes": 9999,
    },
    {
        "name":    "flashlight",
        "color":   YELLOW,
        "title":   "PHASE 3 — FLASHLIGHT",
        "instruction": "Turn a flashlight (or phone torch) ON, hold it, then turn it OFF.",
        "expected_dominant": "SPIKE_OVERLIGHT",
        "min_spikes": 1,
        "max_spikes": 9999,
    },
    {
        "name":    "music",
        "color":   RED,
        "title":   "PHASE 4 — AUDIO",
        "instruction": "Play music loudly or clap near the microphone.",
        "expected_dominant": "SPIKE_SOUND",
        "min_spikes": 1,
        "max_spikes": 9999,
    },
    {
        "name":    "combined",
        "color":   WHITE,
        "title":   "PHASE 5 — COMBINED",
        "instruction": "Move + flashlight on/off + audio — all at the same time.",
        "expected_dominant": None,
        "min_spikes": 3,
        "max_spikes": 9999,
    },
]

# ── frame_filter command ───────────────────────────────────────────────────────
BASE_FF = [
    "python3", "frame_filter.py",
    "--source",       "0",
    "--width",        "640",
    "--height",       "480",
    "--fps",          "18",
    "--audio",
    "--audio_device", "hw:1,0",
    "--max_frames",   str(PHASE_FRAMES),
    "--save_mode",    "all",
    "--postrun",
]

# ── record one phase ──────────────────────────────────────────────────────────
def record_phase(phase, dry_run):
    name = phase["name"]

    try:
        input(f"\n  {BOLD}Press ENTER to start — recording {PHASE_SECONDS}s automatically …{RESET}  ")
    except (EOFError, KeyboardInterrupt):
        print(f"\n{YELLOW}Aborted.{RESET}")
        sys.exit(0)

    drain_stdin()

    if dry_run:
        print(f"\n  {clr(GREEN, '● RECORDING')}  (dry-run, simulating {PHASE_SECONDS}s …)\n")
        time.sleep(2)
        print(f"  {clr(CYAN, '■ DONE')}  (dry-run)\n")
        return None

    cmd = BASE_FF + ["--phases", f"{name}:1-{PHASE_FRAMES}"]

    t_start = time.time()
    print(f"\n  {clr(GREEN, '● RECORDING')}  {PHASE_SECONDS}s  ({PHASE_FRAMES} frames)\n", flush=True)

    proc = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)

    elapsed = time.time() - t_start
    print(f"  {clr(CYAN, '■ DONE')}  {elapsed:.1f}s")

    summaries = sorted(glob.glob(os.path.join(WORKDIR, "logs/run_*/post/run_summary.json")))
    if not summaries:
        print(f"  {clr(RED, 'ERROR:')} no run_summary.json found")
        return None

    with open(summaries[-1]) as f:
        summary = json.load(f)

    summary["_elapsed"] = elapsed
    return summary

# ── evaluate ──────────────────────────────────────────────────────────────────
def evaluate(phase, summary):
    if summary is None:
        print(f"  {clr(RED, 'FAIL')} — no data\n")
        return "FAIL"

    spikes   = summary.get("spike_counters", {})
    total    = sum(spikes.values())
    dominant = max(spikes, key=spikes.get) if total > 0 else None
    fps_val  = summary.get("fps", 0)

    exp_dom  = phase["expected_dominant"]
    count_ok = phase["min_spikes"] <= total <= phase["max_spikes"]
    dom_ok   = (exp_dom is None) or (dominant == exp_dom)
    fps_ok   = fps_val >= 8
    passed   = count_ok and dom_ok and fps_ok

    w = 60
    print(f"\n  {BOLD}{'─'*w}{RESET}")
    print(f"  {'Spike type':<26}  {'count':>6}")
    print(f"  {'─'*26}  {'─'*6}")
    any_spike = False
    for k, v in sorted(spikes.items()):
        if v > 0:
            marker = "  ◄" if k == dominant else ""
            print(f"  {k:<26}  {v:>6}{marker}")
            any_spike = True
    if not any_spike:
        print(f"  {'(no spikes)'}")
    print(f"  {'─'*26}  {'─'*6}")
    print(f"  {'TOTAL':<26}  {total:>6}\n")

    def row(label, value, ok):
        sym = clr(GREEN, "✓") if ok else clr(RED, "✗")
        print(f"  {sym}  {label:<26}  {value}")

    row("dominant spike",  str(dominant),        dom_ok)
    row("expected",        str(exp_dom) if exp_dom else "none (quiet)", dom_ok)
    row("spike count",     str(total),            count_ok)
    row("fps",             f"{fps_val:.1f}",      fps_ok)
    row("duration",        f"{summary['_elapsed']:.1f}s", True)

    verdict = "PASS" if passed else "FAIL"
    vcolor  = GREEN if passed else RED
    print(f"\n  {'─'*w}")
    print(f"  RESULT:  {clr(vcolor, verdict)}\n")
    return verdict

# ── run one phase with repeat option ─────────────────────────────────────────
def run_phase_loop(phase, dry_run):
    while True:
        banner(phase["color"], phase["title"], phase["instruction"])

        summary = record_phase(phase, dry_run)
        verdict = evaluate(phase, summary)

        try:
            choice = input(f"  Press ENTER for next phase,  or R + ENTER to repeat:  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice == "r":
            continue
        return verdict == "PASS"

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    dry_run = "--dry-run" in sys.argv

    print(f"\n{clr(CYAN, 'CLAW Manual-Gated Calibration')}")
    print(f"  Press ENTER to start each phase ({PHASE_SECONDS}s, auto-stop)")
    print(f"  R + ENTER after result = repeat phase\n")

    results = {}

    for phase in PHASES:
        passed = run_phase_loop(phase, dry_run)
        results[phase["name"]] = passed

    w = 64
    print(f"\n{BOLD}{'═'*w}{RESET}")
    print(f"{BOLD}  CALIBRATION SUMMARY{RESET}")
    print(f"{BOLD}{'═'*w}{RESET}\n")

    all_pass = True
    for phase in PHASES:
        r = results.get(phase["name"])
        if r is None:
            status = clr(YELLOW, "SKIPPED")
            all_pass = False
        elif r:
            status = clr(GREEN, "PASS")
        else:
            status = clr(RED, "FAIL")
            all_pass = False
        print(f"  {phase['title']:<32}  {status}")

    print()
    if all_pass:
        print(f"  {clr(GREEN, 'ALL PHASES PASS — calibration complete.')}")
    else:
        print(f"  {clr(YELLOW, 'One or more phases did not pass.')}")
    print()


if __name__ == "__main__":
    main()
