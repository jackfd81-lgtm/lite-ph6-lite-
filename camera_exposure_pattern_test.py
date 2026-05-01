import cv2
import time
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("logs") / ("exposure_pattern_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG = OUT_DIR / "exposure_pattern_log.jsonl"

DEVICE = "/dev/video0"
SOURCE = 0
WIDTH = 640
HEIGHT = 480
FPS = 30

# Exposure values are camera-dependent.
# Many UVC webcams use exposure_absolute values like 5-10000.
PATTERN = [
    {
        "name": "baseline_auto_10s",
        "mode": "auto",
        "duration_sec": 10,
        "wait_after_sec": 5,
    },
    {
        "name": "manual_short_10s",
        "mode": "manual",
        "exposure": 50,
        "duration_sec": 10,
        "wait_after_sec": 30,
    },
    {
        "name": "manual_medium_10s",
        "mode": "manual",
        "exposure": 300,
        "duration_sec": 10,
        "wait_after_sec": 30,
    },
    {
        "name": "manual_long_10s",
        "mode": "manual",
        "exposure": 1000,
        "duration_sec": 10,
        "wait_after_sec": 30,
    },
    {
        "name": "manual_very_long_10s",
        "mode": "manual",
        "exposure": 3000,
        "duration_sec": 10,
        "wait_after_sec": 30,
    },
    {
        "name": "return_auto_10s",
        "mode": "auto",
        "duration_sec": 10,
        "wait_after_sec": 0,
    },
]


def run_cmd(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "error": str(e),
        }


def set_auto_exposure():
    # Different UVC drivers use different control names.
    results = []
    results.append(run_cmd(f"v4l2-ctl -d {DEVICE} -c exposure_auto=3"))
    results.append(run_cmd(f"v4l2-ctl -d {DEVICE} -c auto_exposure=3"))
    return results


def set_manual_exposure(value):
    results = []
    results.append(run_cmd(f"v4l2-ctl -d {DEVICE} -c exposure_auto=1"))
    results.append(run_cmd(f"v4l2-ctl -d {DEVICE} -c auto_exposure=1"))
    results.append(run_cmd(f"v4l2-ctl -d {DEVICE} -c exposure_absolute={value}"))
    results.append(run_cmd(f"v4l2-ctl -d {DEVICE} -c exposure_time_absolute={value}"))
    return results


def write_log(obj):
    with open(LOG, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

write_log({
    "event": "session_start",
    "out_dir": str(OUT_DIR),
    "width": WIDTH,
    "height": HEIGHT,
    "fps_requested": FPS,
    "fps_reported": cap.get(cv2.CAP_PROP_FPS),
    "fourcc_reported": int(cap.get(cv2.CAP_PROP_FOURCC)),
})

frame_seq = 0

for step_index, step in enumerate(PATTERN, start=1):
    print(f"\n=== STEP {step_index}: {step['name']} ===")

    if step["mode"] == "auto":
        ctrl_results = set_auto_exposure()
    else:
        ctrl_results = set_manual_exposure(step["exposure"])

    write_log({
        "event": "pattern_step_start",
        "step_index": step_index,
        "step": step,
        "control_results": ctrl_results,
        "timestamp": time.time(),
    })

    step_start = time.perf_counter()
    frames = 0
    brightness_values = []

    while time.perf_counter() - step_start < step["duration_sec"]:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()

        if not ok:
            write_log({
                "event": "frame_read_failed",
                "step_index": step_index,
                "frame_seq": frame_seq,
                "timestamp": time.time(),
            })
            continue

        frame_seq += 1
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        brightness_values.append(mean_brightness)

        if frame_seq % 30 == 0:
            out_path = OUT_DIR / f"step{step_index:02d}_frame{frame_seq:06d}.jpg"
            cv2.imwrite(str(out_path), frame)

        write_log({
            "event": "frame",
            "step_index": step_index,
            "step_name": step["name"],
            "frame_seq": frame_seq,
            "read_ms": round((t1 - t0) * 1000, 3),
            "mean_brightness": round(mean_brightness, 3),
            "timestamp": time.time(),
        })

    elapsed = time.perf_counter() - step_start

    summary = {
        "event": "pattern_step_end",
        "step_index": step_index,
        "step_name": step["name"],
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "fps": round(frames / elapsed, 3) if elapsed > 0 else 0,
        "brightness_min": round(min(brightness_values), 3) if brightness_values else None,
        "brightness_max": round(max(brightness_values), 3) if brightness_values else None,
        "brightness_avg": round(sum(brightness_values) / len(brightness_values), 3) if brightness_values else None,
        "timestamp": time.time(),
    }

    print(summary)
    write_log(summary)

    wait_after = step.get("wait_after_sec", 0)
    if wait_after > 0:
        print(f"Waiting {wait_after}s before next pattern...")
        write_log({
            "event": "wait_start",
            "after_step": step_index,
            "wait_sec": wait_after,
            "timestamp": time.time(),
        })
        time.sleep(wait_after)
        write_log({
            "event": "wait_end",
            "after_step": step_index,
            "timestamp": time.time(),
        })

cap.release()

write_log({
    "event": "session_end",
    "total_frames": frame_seq,
    "timestamp": time.time(),
})

print("\nExposure pattern test complete.")
print(f"Output dir: {OUT_DIR}")
print(f"Log:        {LOG}")
