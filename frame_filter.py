import os
import cv2
import json
import hashlib
import argparse
from datetime import datetime, timezone
import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_append(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def mean_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), gray


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def motion_fraction(gray, prev_gray, threshold=20):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    return float(np.count_nonzero(diff > threshold) / diff.size)


def validate_packet(packet):
    if not isinstance(packet, dict):
        raise ValueError("packet must be dict")

    if "packet_type" not in packet:
        raise ValueError("missing packet_type")

    if "ts_utc" not in packet:
        raise ValueError("missing ts_utc")

    if "packet_seq" not in packet:
        raise ValueError("missing packet_seq")

    ptype = packet["packet_type"]

    allowed = {
        "session_start",
        "config",
        "pseudo",
        "observation",
        "soso",
        "session_end",
    }
    if ptype not in allowed:
        raise ValueError(f"invalid packet_type: {ptype}")

    if not isinstance(packet["packet_seq"], int):
        raise ValueError("packet_seq must be int")

    if ptype in {"pseudo", "observation", "soso"}:
        if "frame_index" not in packet:
            raise ValueError(f"{ptype} missing frame_index")
        if not isinstance(packet["frame_index"], int):
            raise ValueError(f"{ptype} frame_index must be int")

    if ptype == "pseudo":
        required = [
            "mean_brightness",
            "laplacian_var",
            "motion_fraction",
            "verdict",
            "reasons",
        ]
        for k in required:
            if k not in packet:
                raise ValueError(f"pseudo missing {k}")
        if packet["verdict"] not in {"PASS", "DROP"}:
            raise ValueError("pseudo verdict must be PASS or DROP")
        if not isinstance(packet["reasons"], list):
            raise ValueError("pseudo reasons must be list")

    if ptype == "observation":
        required = ["image_path", "image_sha256", "width", "height"]
        for k in required:
            if k not in packet:
                raise ValueError(f"observation missing {k}")

    if ptype == "soso":
        required = ["state", "continuity_count", "confidence", "advisory"]
        for k in required:
            if k not in packet:
                raise ValueError(f"soso missing {k}")

    if ptype == "config":
        required = [
            "session_id",
            "source",
            "save_mode",
            "bright_min",
            "bright_max",
            "lap_min",
            "motion_max",
            "max_frames",
        ]
        for k in required:
            if k not in packet:
                raise ValueError(f"config missing {k}")

    if ptype == "session_start":
        required = ["session_id", "message"]
        for k in required:
            if k not in packet:
                raise ValueError(f"session_start missing {k}")

    if ptype == "session_end":
        required = ["session_id", "frames_processed", "message"]
        for k in required:
            if k not in packet:
                raise ValueError(f"session_end missing {k}")


class PacketWriter:
    def __init__(self, log_path):
        self.log_path = log_path
        self.packet_seq = 0

    def write(self, packet):
        self.packet_seq += 1
        packet["packet_seq"] = self.packet_seq
        validate_packet(packet)
        jsonl_append(self.log_path, packet)


def evaluate_frame(frame, prev_gray, bright_min, bright_max, lap_min, motion_max):
    mb, gray = mean_brightness(frame)
    lv = laplacian_var(gray)
    mf = motion_fraction(gray, prev_gray)

    reasons = []
    if mb < bright_min:
        reasons.append("brightness_low")
    if mb > bright_max:
        reasons.append("brightness_high")
    if lv < lap_min:
        reasons.append("blur_low_detail")
    if mf > motion_max:
        reasons.append("motion_high")

    verdict = "PASS" if len(reasons) == 0 else "DROP"
    return gray, mb, lv, mf, verdict, reasons


def save_frame(frame, frames_dir, frame_index):
    os.makedirs(frames_dir, exist_ok=True)
    path = os.path.join(frames_dir, f"frame_{frame_index:06d}.jpg")
    ok = cv2.imwrite(path, frame)
    if not ok:
        raise RuntimeError(f"Failed to save frame: {path}")
    return path


def make_session_id():
    seed = utc_now().encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for camera, or path to video file")
    ap.add_argument("--camera", type=int, default=0, help="camera index if --source 0")
    ap.add_argument("--log", default="logs/packets.jsonl")
    ap.add_argument("--frames_dir", default="frames")
    ap.add_argument("--save_mode", choices=["pass_only", "all", "none"], default="pass_only")
    ap.add_argument("--bright_min", type=float, default=40.0)
    ap.add_argument("--bright_max", type=float, default=220.0)
    ap.add_argument("--lap_min", type=float, default=40.0)
    ap.add_argument("--motion_max", type=float, default=0.15)
    ap.add_argument("--max_frames", type=int, default=300, help="0 means unlimited")
    args = ap.parse_args()

    source = args.camera if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    session_id = make_session_id()
    writer = PacketWriter(args.log)

    writer.write({
        "packet_type": "session_start",
        "ts_utc": utc_now(),
        "session_id": session_id,
        "message": "session started"
    })

    writer.write({
        "packet_type": "config",
        "ts_utc": utc_now(),
        "session_id": session_id,
        "source": str(source),
        "save_mode": args.save_mode,
        "bright_min": args.bright_min,
        "bright_max": args.bright_max,
        "lap_min": args.lap_min,
        "motion_max": args.motion_max,
        "max_frames": args.max_frames
    })

    prev_gray = None
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1

            gray, mb, lv, mf, verdict, reasons = evaluate_frame(
                frame, prev_gray, args.bright_min, args.bright_max, args.lap_min, args.motion_max
            )

            ts = utc_now()
            saved_path = None
            saved_hash = None

            if args.save_mode == "all" or (args.save_mode == "pass_only" and verdict == "PASS"):
                saved_path = save_frame(frame, args.frames_dir, frame_index)
                saved_hash = sha256_file(saved_path)

            pseudo = {
                "packet_type": "pseudo",
                "ts_utc": ts,
                "frame_index": frame_index,
                "mean_brightness": mb,
                "laplacian_var": lv,
                "motion_fraction": mf,
                "verdict": verdict,
                "reasons": reasons,
            }

            obs = {
                "packet_type": "observation",
                "ts_utc": ts,
                "frame_index": frame_index,
                "image_path": saved_path,
                "image_sha256": saved_hash,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }

            if verdict == "PASS":
                state = "stable"
                advisory = "continue"
                confidence = 0.9
                continuity_count = 1
            else:
                state = "degraded"
                advisory = "review_thresholds"
                confidence = 0.5
                continuity_count = 0

            soso = {
                "packet_type": "soso",
                "ts_utc": ts,
                "frame_index": frame_index,
                "state": state,
                "continuity_count": continuity_count,
                "confidence": confidence,
                "advisory": advisory,
            }

            writer.write(pseudo)
            writer.write(obs)
            writer.write(soso)

            prev_gray = gray

            if args.max_frames and frame_index >= args.max_frames:
                break

    finally:
        cap.release()
        writer.write({
            "packet_type": "session_end",
            "ts_utc": utc_now(),
            "session_id": session_id,
            "frames_processed": frame_index,
            "message": "session ended"
        })

    print(f"Done. Frames processed: {frame_index}")
    print(f"Session ID: {session_id}")
    print(f"Log file: {args.log}")


if __name__ == "__main__":
    main()
