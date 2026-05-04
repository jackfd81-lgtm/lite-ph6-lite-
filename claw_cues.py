#!/usr/bin/env python3
"""
CLAW calibration cue driver.
Runs a countdown, announces each phase, then launches frame_filter
as a subprocess. Phase cues are timed to match 18 FPS × 90 frames = 5s/phase.

Usage:
    python3 claw_cues.py [--dry-run]

--dry-run  Print cues only, do not launch frame_filter.
"""

import subprocess
import sys
import time
import shutil

# ── ANSI helpers ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BELL   = "\a"

def cue(color, label, detail=""):
    line = f"{BOLD}{color}{'═'*60}{RESET}"
    print(line)
    print(f"{BOLD}{color}  {label}{RESET}")
    if detail:
        print(f"  {detail}")
    print(line)
    print(BELL, end="", flush=True)

def tick(seconds, color=WHITE):
    for i in range(seconds, 0, -1):
        print(f"  {color}{BOLD}{i}…{RESET}", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20, end="\r", flush=True)

# ── frame_filter command ───────────────────────────────────────────────────────
FF_CMD = [
    "python3", "frame_filter.py",
    "--source",       "0",
    "--width",        "640",
    "--height",       "480",
    "--fps",          "18",
    "--audio",
    "--audio_device", "hw:1,0",
    "--phases",       "quiet:1-90,camera:91-180,flashlight:181-270,music:271-360,combined:361-450",
    "--max_frames",   "450",
    "--save_mode",    "all",
    "--postrun",
]

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    dry_run = "--dry-run" in sys.argv

    print(f"\n{BOLD}{CYAN}CLAW Calibration Cue System — 18 FPS / 450 frames / 5 phases{RESET}\n")

    if not dry_run:
        ff_path = shutil.which("python3")
        print(f"frame_filter command: {' '.join(FF_CMD)}\n")

    # Countdown — frame_filter launches at "1"
    print(f"{BOLD}Starting in:{RESET}")
    for n in [3, 2]:
        print(f"  {BOLD}{YELLOW}{n}…{RESET}", flush=True)
        time.sleep(1)
    print(f"  {BOLD}{RED}1 — GO{RESET}", flush=True)
    time.sleep(0.5)

    # Launch frame_filter
    proc = None
    if not dry_run:
        proc = subprocess.Popen(FF_CMD, cwd="/home/jack/frame_filter")

    t_start = time.time()

    # ── Phase 1: QUIET (0–5s) ──────────────────────────────────────────────
    cue(GREEN, "QUIET PHASE  [frames 1–90]", "Stay still. No sound. No movement.")
    tick(5, GREEN)

    # ── Phase 2: CAMERA (5–10s) ───────────────────────────────────────────
    cue(CYAN, "CAMERA PHASE  [frames 91–180]", "MOVE in front of the camera now.")
    tick(5, CYAN)

    # ── Phase 3: FLASHLIGHT (10–15s) ──────────────────────────────────────
    cue(YELLOW, "FLASHLIGHT PHASE  [frames 181–270]", "Turn light ON then OFF. Hold briefly.")
    tick(5, YELLOW)

    # ── Phase 4: MUSIC (15–20s) ───────────────────────────────────────────
    cue(RED, "MUSIC PHASE  [frames 271–360]", "START playing audio / music now.")
    tick(5, RED)

    # ── Phase 5: COMBINED (20–25s) ────────────────────────────────────────
    cue(WHITE, "COMBINED PHASE  [frames 361–450]", "MOVE + LIGHT + AUDIO — all together.")
    tick(5, WHITE)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    cue(GREEN, "CALIBRATION COMPLETE", f"Elapsed: {elapsed:.1f}s. Stop all actions.")

    if proc is not None:
        proc.wait()
        print(f"\n{BOLD}frame_filter exited with code {proc.returncode}{RESET}")

        # Print postrun report if it exists
        import glob, os
        runs = sorted(glob.glob("/home/jack/frame_filter/logs/run_*/post/postrun_report.md"))
        if runs:
            latest = runs[-1]
            print(f"\n{BOLD}── Postrun report: {latest} ──{RESET}\n")
            with open(latest) as f:
                print(f.read())


if __name__ == "__main__":
    main()
