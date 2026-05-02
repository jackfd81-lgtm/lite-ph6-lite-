#!/usr/bin/env python3
"""
PH6-Lite Pi 5 Coherence Check

Purpose:
- Verify core PH6-Lite files exist.
- Verify Python modules import cleanly.
- Verify CRAMWriter modes behave.
- Verify store/durable annotation.
- Verify virtual tokens remain MRAM-S / authority NONE.
- Run existing test suite.
- Check camera visibility.
- Check disk/mount basics.
- Check recent JSONL logs for malformed lines.

This script is diagnostic only.
It does not modify PH6 authority logic.
"""

import os
import sys
import json
import time
import glob
import shutil
import tempfile
import subprocess
from pathlib import Path


ROOT = Path.cwd()

REQUIRED_FILES = [
    "frame_filter.py",
    "cram_writer.py",
    "virtual_tokens.py",
    "test_token_leakage.py",
    "test_ph6lite_phase2.py",
]

OPTIONAL_FILES = [
    "ph6lite/backend.py",
    "ph6lite/advisory_client.py",
    "ph6lite/schema_validator.py",
]

CRAM_MODES = ["forensic", "balanced", "burst"]


results = {
    "pass": 0,
    "fail": 0,
    "warn": 0,
    "checks": [],
}


def report(status, name, detail=""):
    status = status.upper()
    if status == "PASS":
        results["pass"] += 1
    elif status == "FAIL":
        results["fail"] += 1
    else:
        results["warn"] += 1

    results["checks"].append({
        "status": status,
        "name": name,
        "detail": detail,
    })

    marker = {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "WARN": "WARN",
    }.get(status, status)

    if detail:
        print(f"{marker:4} | {name} — {detail}")
    else:
        print(f"{marker:4} | {name}")


def run_cmd(name, cmd, timeout=30, allow_fail=False):
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if p.returncode == 0:
            report("PASS", name)
            return True, p.stdout.strip(), p.stderr.strip()
        else:
            detail = p.stderr.strip() or p.stdout.strip() or f"returncode={p.returncode}"
            if allow_fail:
                report("WARN", name, detail[:300])
                return False, p.stdout.strip(), p.stderr.strip()
            report("FAIL", name, detail[:300])
            return False, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        if allow_fail:
            report("WARN", name, f"timeout after {timeout}s")
            return False, "", "timeout"
        report("FAIL", name, f"timeout after {timeout}s")
        return False, "", "timeout"
    except Exception as e:
        if allow_fail:
            report("WARN", name, repr(e))
            return False, "", repr(e)
        report("FAIL", name, repr(e))
        return False, "", repr(e)


def check_python():
    report("PASS", "python executable", sys.executable)
    report("PASS", "python version", sys.version.split()[0])


def check_project_files():
    for f in REQUIRED_FILES:
        if (ROOT / f).exists():
            report("PASS", f"required file exists: {f}")
        else:
            report("FAIL", f"required file missing: {f}")

    for f in OPTIONAL_FILES:
        if (ROOT / f).exists():
            report("PASS", f"optional file exists: {f}")
        else:
            report("WARN", f"optional file missing: {f}")


def check_imports():
    modules = [
        "cram_writer",
        "virtual_tokens",
    ]

    for mod in modules:
        try:
            __import__(mod)
            report("PASS", f"import {mod}")
        except Exception as e:
            report("FAIL", f"import {mod}", repr(e))


def check_cram_writer_modes():
    try:
        from cram_writer import CRAMWriter
    except Exception as e:
        report("FAIL", "CRAMWriter import", repr(e))
        return

    for mode in CRAM_MODES:
        try:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / f"test_{mode}.jsonl"

                writer = CRAMWriter(
                    out,
                    mode=mode,
                    flush_every=1,
                    buffer_size=8,
                )

                pkt = {
                    "packet_type": "session_start",
                    "ts_utc": "2026-05-02T00:00:00Z",
                    "packet_seq": 1,
                    "session_id": "coherence_check",
                    "message": "coherence test packet",
                }

                writer.write(pkt)
                if hasattr(writer, "flush"):
                    writer.flush()
                if hasattr(writer, "close"):
                    writer.close()

                if not out.exists():
                    report("FAIL", f"CRAMWriter mode {mode}", "output file missing")
                    continue

                lines = out.read_text(encoding="utf-8").strip().splitlines()
                if not lines:
                    report("FAIL", f"CRAMWriter mode {mode}", "no JSONL lines written")
                    continue

                obj = json.loads(lines[-1])

                store_ok = obj.get("store") == "CRAM"
                durable_ok = obj.get("durable") is True

                if store_ok and durable_ok:
                    report("PASS", f"CRAMWriter mode {mode}", "store=CRAM durable=true")
                else:
                    report(
                        "FAIL",
                        f"CRAMWriter mode {mode}",
                        f"store={obj.get('store')} durable={obj.get('durable')}",
                    )

        except Exception as e:
            report("FAIL", f"CRAMWriter mode {mode}", repr(e))


def check_virtual_token_quarantine():
    try:
        from virtual_tokens import VirtualTokenTracker
    except Exception as e:
        report("FAIL", "VirtualTokenTracker import", repr(e))
        return

    try:
        tracker = VirtualTokenTracker(
            link_window_frames=30,
            promote_min_frames=3,
            promote_min_strength=2,
            close_after_frames=60,
        )

        pkts = []
        for i in range(5):
            emitted = tracker.observe_event(
                frame_index=i,
                event_kind="motion",
                ts="2026-05-02T00:00:00Z",
            )
            pkts.extend(emitted or [])

        if not pkts:
            report("WARN", "virtual token emission", "no packets emitted")
            return

        bad = []
        for p in pkts:
            if p.get("authority") != "NONE":
                bad.append(("authority", p))
            if p.get("store") != "MRAM-S":
                bad.append(("store", p))
            if "verdict" in p:
                bad.append(("verdict_leak", p))
            if p.get("durable") is True:
                bad.append(("durability_leak", p))

        if bad:
            report("FAIL", "virtual token quarantine", f"{len(bad)} bad token fields")
        else:
            report("PASS", "virtual token quarantine", "authority=NONE store=MRAM-S no verdict/durable leak")

    except Exception as e:
        report("FAIL", "virtual token quarantine", repr(e))


def check_existing_tests():
    tests = [
        "python3 test_token_leakage.py",
        "python3 test_ph6lite_phase2.py --gen",
    ]

    for t in tests:
        run_cmd(f"existing test: {t}", t, timeout=90)


def check_camera():
    video_devices = sorted(glob.glob("/dev/video*"))
    if video_devices:
        report("PASS", "camera device nodes", ", ".join(video_devices))
    else:
        report("WARN", "camera device nodes", "no /dev/video* found")

    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()

        if ok and frame is not None:
            h, w = frame.shape[:2]
            report("PASS", "OpenCV camera read", f"{w}x{h}")
        else:
            report("WARN", "OpenCV camera read", "camera exists but frame read failed")
    except ImportError:
        report("WARN", "OpenCV camera read", "cv2 not installed")
    except Exception as e:
        report("WARN", "OpenCV camera read", repr(e))


def check_disk_and_mounts():
    total, used, free = shutil.disk_usage(str(ROOT))
    free_gb = free / (1024 ** 3)

    if free_gb >= 1.0:
        report("PASS", "disk free space", f"{free_gb:.2f} GB free")
    else:
        report("WARN", "disk free space", f"{free_gb:.2f} GB free")

    ok, out, err = run_cmd(
        "mount info",
        "findmnt -T . -o TARGET,SOURCE,FSTYPE,OPTIONS",
        timeout=10,
        allow_fail=True,
    )
    if out:
        print("\n--- mount info ---")
        print(out)
        print("------------------\n")

    ok, out, err = run_cmd(
        "block devices",
        "lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS",
        timeout=10,
        allow_fail=True,
    )
    if out:
        print("\n--- block devices ---")
        print(out)
        print("---------------------\n")


def check_recent_jsonl_logs():
    candidates = []
    for pattern in [
        "logs/**/*.jsonl",
        "logs/**/*.json",
        "*.jsonl",
    ]:
        candidates.extend(glob.glob(pattern, recursive=True))

    candidates = sorted(
        set(candidates),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )[:20]

    if not candidates:
        report("WARN", "recent JSONL logs", "no JSONL/JSON logs found")
        return

    malformed = 0
    checked_lines = 0

    for path in candidates:
        p = Path(path)
        if p.suffix == ".json":
            try:
                json.loads(p.read_text(encoding="utf-8"))
                report("PASS", f"JSON readable: {path}")
            except Exception as e:
                report("WARN", f"JSON malformed: {path}", repr(e))
            continue

        try:
            with p.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if idx > 1000:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    checked_lines += 1
                    try:
                        json.loads(line)
                    except Exception:
                        malformed += 1
                        report("WARN", f"JSONL malformed line: {path}:{idx}")
                        break
        except Exception as e:
            report("WARN", f"JSONL unreadable: {path}", repr(e))

    if malformed == 0:
        report("PASS", "recent JSONL logs", f"checked {checked_lines} lines")


def write_report():
    report_path = ROOT / "ph6lite_coherence_report.json"
    payload = {
        "timestamp_unix": time.time(),
        "root": str(ROOT),
        "summary": {
            "pass": results["pass"],
            "fail": results["fail"],
            "warn": results["warn"],
        },
        "checks": results["checks"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nReport written: {report_path}")


def main():
    print("\nPH6-Lite Pi 5 Coherence Check")
    print("=" * 42)
    print(f"Root: {ROOT}")
    print("=" * 42)

    check_python()
    check_project_files()
    check_imports()
    check_cram_writer_modes()
    check_virtual_token_quarantine()
    check_existing_tests()
    check_camera()
    check_disk_and_mounts()
    check_recent_jsonl_logs()
    write_report()

    print("\nFinal Summary")
    print("=" * 42)
    print(f"PASS: {results['pass']}")
    print(f"WARN: {results['warn']}")
    print(f"FAIL: {results['fail']}")

    if results["fail"] == 0:
        print("\nFINAL VERDICT: PASS — Pi 5 PH6-Lite stack is coherent.")
        return 0

    print("\nFINAL VERDICT: FAIL — fix failed checks before running evidence tests.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
