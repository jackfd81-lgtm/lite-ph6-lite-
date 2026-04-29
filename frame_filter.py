import os
import cv2
import json
import hashlib
import argparse
import subprocess
import threading
import collections
from datetime import datetime, timezone
import numpy as np
from cram_writer import CRAMWriter
from ph6lite.advisory_client import ask as llm_ask


class AudioCapture:
    """Captures audio from mic via arecord in a background thread."""

    def __init__(self, device="hw:1,0", rate=16000):
        self.device = device
        self.rate = rate
        self._buf = collections.deque(maxlen=rate * 4)
        self._lock = threading.Lock()
        self._proc = None
        self._thread = None
        self._running = False

    def start(self):
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE",
             "-r", str(self.rate), "-c", "1", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        chunk = self.rate // 10 * 2  # 100ms of S16_LE bytes
        while self._running:
            data = self._proc.stdout.read(chunk)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16)
            with self._lock:
                self._buf.extend(samples.tolist())

    def get_metrics(self, duration_ms=200):
        n = int(self.rate * duration_ms / 1000)
        with self._lock:
            samples = list(self._buf)[-n:] if len(self._buf) >= n else list(self._buf)
        if not samples:
            return 0.0, 0.0, False
        arr = np.array(samples, dtype=np.float32) / 32768.0
        rms = float(round(float(np.sqrt(np.mean(arr ** 2))), 6))
        peak = float(round(float(np.max(np.abs(arr))), 6))
        return rms, peak, peak >= 0.999

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_append(path, obj):
    # Legacy direct-write kept for standalone use outside CRAMWriter
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
        "audio",
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

    if ptype == "audio":
        for k in ("rms_level", "peak_level", "clipping", "sample_rate", "duration_ms"):
            if k not in packet:
                raise ValueError(f"audio missing {k}")

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
    """Thin wrapper — delegates to CRAMWriter for compliant write path."""
    def __init__(self, log_path):
        self._cram = CRAMWriter(log_path, buffer_size=64, flush_every=1)

    @property
    def packet_seq(self):
        return self._cram.packet_seq

    def write(self, packet):
        self._cram.write(packet)

    def close(self):
        self._cram.close()


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
    ap.add_argument("--probe_id", default="", help="probe identifier stamped into every packet")
    ap.add_argument("--llm", action="store_true", help="enable LLM advisory via Ollama (local or Jetson)")
    ap.add_argument("--width", type=int, default=0, help="capture width (0=camera default)")
    ap.add_argument("--height", type=int, default=0, help="capture height (0=camera default)")
    ap.add_argument("--fps", type=int, default=0, help="capture fps (0=camera default)")
    ap.add_argument("--signal_frames", default="", help="comma-separated frame indices to print COVER/UNCOVER cues")
    ap.add_argument("--audio", action="store_true", help="capture audio from camera mic and emit audio packets")
    ap.add_argument("--audio_device", default="hw:1,0", help="ALSA device for mic (default hw:1,0)")
    args = ap.parse_args()

    is_url = isinstance(args.source, str) and args.source.startswith("http")
    source = args.source if is_url else (args.camera if args.source == "0" else args.source)
    if is_url:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    audio_cap = None
    if args.audio:
        audio_cap = AudioCapture(device=args.audio_device)
        audio_cap.start()

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
        "probe_id": args.probe_id,
        "save_mode": args.save_mode,
        "bright_min": args.bright_min,
        "bright_max": args.bright_max,
        "lap_min": args.lap_min,
        "motion_max": args.motion_max,
        "max_frames": args.max_frames
    })

    prev_gray = None
    frame_index = 0
    continuity_count = 0
    confidence = 0.5

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
                "probe_id": args.probe_id,
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
                "probe_id": args.probe_id,
                "image_path": saved_path,
                "image_sha256": saved_hash,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }

            if verdict == "PASS":
                continuity_count += 1
                confidence = min(0.99, confidence + 0.05)
                if continuity_count >= 10:
                    state = "stable"
                else:
                    state = "recovering" if continuity_count > 1 else "warming_up"
                advisory = "continue"
            else:
                continuity_count = 0
                confidence = max(0.1, confidence - 0.3)
                state = "degraded"
                advisory = "review_thresholds"

            llm_backend = None
            if args.llm and verdict == "DROP":
                prompt = (
                    f"Frame {frame_index} dropped. "
                    f"brightness={mb:.1f} sharpness={lv:.1f} motion={mf:.3f} "
                    f"reasons={reasons}. "
                    "One-sentence advisory for the operator."
                )
                result = llm_ask("reason", prompt)
                advisory = result.get("output", advisory).strip() or advisory
                llm_backend = result.get("backend")

            soso = {
                "packet_type": "soso",
                "ts_utc": ts,
                "frame_index": frame_index,
                "probe_id": args.probe_id,
                "state": state,
                "continuity_count": continuity_count,
                "confidence": confidence,
                "advisory": advisory,
            }
            if llm_backend:
                soso["llm_backend"] = llm_backend

            writer.write(pseudo)
            writer.write(obs)
            writer.write(soso)

            if audio_cap:
                rms, peak, clipping = audio_cap.get_metrics(duration_ms=200)
                writer.write({
                    "packet_type": "audio",
                    "ts_utc": ts,
                    "frame_index": frame_index,
                    "probe_id": args.probe_id,
                    "rms_level": rms,
                    "peak_level": peak,
                    "clipping": clipping,
                    "sample_rate": audio_cap.rate,
                    "duration_ms": 200,
                })

            if args.signal_frames:
                signals = [int(x) for x in args.signal_frames.split(",")]
                if frame_index == signals[0]:
                    print(f">>> COVER LENS NOW (frame {frame_index})", flush=True)
                elif len(signals) > 1 and frame_index == signals[1]:
                    print(f">>> UNCOVER LENS NOW (frame {frame_index})", flush=True)

            print(f"f={frame_index:3d}  {verdict}  {state:<12}  conf={confidence:.2f}  cont={continuity_count}", flush=True)

            prev_gray = gray

            if args.max_frames and frame_index >= args.max_frames:
                break

    finally:
        cap.release()
        if audio_cap:
            audio_cap.stop()
        writer.write({
            "packet_type": "session_end",
            "ts_utc": utc_now(),
            "session_id": session_id,
            "frames_processed": frame_index,
            "message": "session ended"
        })

    writer.close()
    print(f"Done. Frames processed: {frame_index}")
    print(f"Session ID: {session_id}")
    print(f"Log file: {args.log}")


if __name__ == "__main__":
    main()
