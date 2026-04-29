import os
import cv2
import json
import wave
import hashlib
import argparse
import subprocess
import threading
import collections
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from cram_writer import CRAMWriter
from ph6lite.advisory_client import ask as llm_ask


class AudioCapture:
    """Captures audio from mic via arecord in a background thread."""

    def __init__(self, device="hw:1,0", rate=16000, wav_path=None):
        self.device   = device
        self.rate     = rate
        self.wav_path = wav_path
        self._buf     = collections.deque(maxlen=rate * 4)
        self._lock    = threading.Lock()
        self._wav_lock = threading.Lock()
        self._proc    = None
        self._thread  = None
        self._running = False
        self._wav     = None

    def start(self):
        if self.wav_path:
            self._wav = wave.open(self.wav_path, "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self.rate)
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
            if self._wav:
                with self._wav_lock:
                    self._wav.writeframes(data)
            samples = np.frombuffer(data, dtype=np.int16)
            with self._lock:
                self._buf.extend(samples.tolist())

    def get_metrics(self, duration_ms=200):
        n = int(self.rate * duration_ms / 1000)
        with self._lock:
            samples = list(self._buf)[-n:] if len(self._buf) >= n else list(self._buf)
        if not samples:
            return 0.0, 0.0, False
        arr  = np.array(samples, dtype=np.float32) / 32768.0
        rms  = float(round(float(np.sqrt(np.mean(arr ** 2))), 6))
        peak = float(round(float(np.max(np.abs(arr))), 6))
        return rms, peak, peak >= 0.999

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
        if self._wav:
            with self._wav_lock:
                self._wav.close()
            self._wav = None


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


# ── spike detection ───────────────────────────────────────────────────────────

_SPIKE_CHARS = {
    "SPIKE_MOTION":     "M",
    "SPIKE_SOUND":      "S",
    "SPIKE_OVERLIGHT":  "H",
    "SPIKE_UNDERLIGHT": "D",
    "SPIKE_BLUR":       "B",
    "SPIKE_COMBINED":   "X",
}

_TREND_BLOCKS = " ▁▂▃▄▅▆▇█"

_CLIP_PRE  = 150   # ~5s pre-spike at 30fps
_CLIP_POST = 150   # ~5s post-spike


def detect_spikes(frame_no, ts, metrics, baseline):
    spikes     = []
    brightness = metrics["mean_brightness"]
    motion     = metrics["motion_fraction"]
    blur       = metrics["laplacian_var"]
    rms        = metrics.get("audio_rms", 0.0)
    peak       = metrics.get("audio_peak", 0.0)

    if brightness > 220:
        spikes.append("SPIKE_OVERLIGHT")
    if brightness < 35:
        spikes.append("SPIKE_UNDERLIGHT")
    if motion > max(0.15, baseline["motion_avg"] * 3):
        spikes.append("SPIKE_MOTION")
    if rms > max(0.20, baseline["audio_rms_avg"] * 4) or peak > 0.95:
        spikes.append("SPIKE_SOUND")
    if blur < 80:
        spikes.append("SPIKE_BLUR")
    if len(spikes) >= 2:
        spikes.append("SPIKE_COMBINED")

    if not spikes:
        return None

    score = (
        3 * ("SPIKE_MOTION"     in spikes) +
        3 * ("SPIKE_SOUND"      in spikes) +
        2 * ("SPIKE_OVERLIGHT"  in spikes) +
        2 * ("SPIKE_UNDERLIGHT" in spikes) +
        2 * ("SPIKE_BLUR"       in spikes)
    )

    if score >= 7:
        severity = "HIGH"
    elif score >= 5:
        severity = "ALERT"
    elif score >= 3:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "packet_type": "spike_event",
        "ts_utc":      ts,
        "frame_index": frame_no,
        "spikes":      spikes,
        "severity":    severity,
        "spike_score": score,
        "metrics":     metrics,
    }


def detect_presoak(frame_no, ts, metrics, prev_metrics, baseline):
    """Detect fast-rising sensor values approaching spike thresholds."""
    if prev_metrics is None:
        return None

    warnings = []

    rms      = metrics.get("audio_rms", 0.0)
    prev_rms = prev_metrics.get("audio_rms", 0.0)
    rms_thr  = max(0.20, baseline["audio_rms_avg"] * 4)
    if prev_rms > 0 and rms < rms_thr and rms > prev_rms * 1.75:
        warnings.append("PRE_SPIKE_SOUND")

    mf      = metrics["motion_fraction"]
    prev_mf = prev_metrics["motion_fraction"]
    mf_thr  = max(0.15, baseline["motion_avg"] * 3)
    if prev_mf > 0 and mf < mf_thr and mf > prev_mf * 1.75:
        warnings.append("PRE_SPIKE_MOTION")

    brightness      = metrics["mean_brightness"]
    prev_brightness = prev_metrics["mean_brightness"]
    if abs(brightness - prev_brightness) > 20 and 35 < brightness < 220:
        warnings.append("PRE_SPIKE_LIGHT")

    if not warnings:
        return None

    return {
        "packet_type":   "warning_event",
        "ts_utc":        ts,
        "frame_index":   frame_no,
        "warnings":      warnings,
        "presoak_score": len(warnings),
        "metrics":       metrics,
    }


def update_baseline(baseline, metrics, alpha=0.02):
    baseline["motion_avg"]     = baseline["motion_avg"]     * (1 - alpha) + metrics["motion_fraction"]     * alpha
    baseline["audio_rms_avg"]  = baseline["audio_rms_avg"]  * (1 - alpha) + metrics.get("audio_rms", 0.0) * alpha
    baseline["brightness_avg"] = baseline["brightness_avg"] * (1 - alpha) + metrics["mean_brightness"]     * alpha
    return baseline


_CAP_MODES   = ("QUIET", "ACTIVE", "ELEVATED_CAPTURE", "EVENT_BURST")
_CAP_COOLDOWN = 150  # frames of quiet before stepping down one level
_CAP_WINDOW   = 60   # rolling window for density counters

_CAP_CONFIG = {
    "QUIET":            {"label": "MODE 0", "pre_s":  0, "post_s":  0, "quiet_terminal": False},
    "ACTIVE":           {"label": "MODE 1", "pre_s":  3, "post_s":  3, "quiet_terminal": False},
    "ELEVATED_CAPTURE": {"label": "MODE 2", "pre_s":  5, "post_s":  5, "quiet_terminal": False},
    "EVENT_BURST":      {"label": "MODE 3", "pre_s": 10, "post_s": 10, "quiet_terminal": True},
}


def apply_cap_mode(mode, fps=20):
    """Return behavioral config dict for the current CAP mode."""
    cfg = _CAP_CONFIG[mode]
    return {
        "write_every_packet":   True,
        "save_event_buffer":    mode != "QUIET",
        "prebuffer_frames":     int(cfg["pre_s"] * fps),
        "postbuffer_frames":    int(cfg["post_s"] * fps),
        "render_clips_live":    False,
        "terminal_graphs":      False,
        "quiet_terminal":       cfg["quiet_terminal"],
    }


class CAPState:
    """CRAM Adaptive Capture Posture — deterministic mode state machine."""

    def __init__(self):
        self.mode  = "QUIET"
        self.since = 1
        self._spikes   = collections.deque()   # (frame_no, score)
        self._combined = collections.deque()   # frame_no of SPIKE_COMBINED events
        self._drops    = collections.deque()   # frame_no of DROPs

    def _purge(self, frame_no):
        cutoff = frame_no - _CAP_WINDOW
        while self._spikes   and self._spikes[0][0]   < cutoff: self._spikes.popleft()
        while self._combined and self._combined[0]    < cutoff: self._combined.popleft()
        while self._drops    and self._drops[0]       < cutoff: self._drops.popleft()

    def update(self, frame_no, spike, presoak, is_drop=False):
        """Return a cap_mode_transition packet if mode changed, else None."""
        if spike:
            self._spikes.append((frame_no, spike.get("spike_score", 1)))
            if "SPIKE_COMBINED" in spike.get("spikes", []):
                self._combined.append(frame_no)
        if is_drop:
            self._drops.append(frame_no)

        self._purge(frame_no)
        spike_count_60   = len(self._spikes)
        combined_count_60 = len(self._combined)
        drop_count_60    = len(self._drops)
        avg_score = (sum(s for _, s in self._spikes) / spike_count_60
                     if spike_count_60 else 0.0)

        if drop_count_60 >= 3 or combined_count_60 >= 2:
            stim_mode = "EVENT_BURST"
            trigger   = f"drops={drop_count_60},combined={combined_count_60}"
        elif combined_count_60 >= 1 or avg_score >= 5:
            stim_mode = "ELEVATED_CAPTURE"
            trigger   = f"combined={combined_count_60},avg_score={avg_score:.1f}"
        elif spike_count_60 >= 3 or avg_score >= 2:
            stim_mode = "ACTIVE"
            trigger   = f"spikes={spike_count_60},avg_score={avg_score:.1f}"
        else:
            stim_mode = "QUIET"
            trigger   = None

        current_idx = _CAP_MODES.index(self.mode)
        stim_idx    = _CAP_MODES.index(stim_mode)
        target      = self.mode
        trans_trigger = trigger

        if stim_idx > current_idx:
            target = stim_mode
        elif stim_idx < current_idx and (frame_no - self.since) >= _CAP_COOLDOWN:
            target = _CAP_MODES[current_idx - 1]
            trans_trigger = f"cooldown_{_CAP_COOLDOWN}f"

        if target == self.mode:
            return None

        from_mode  = self.mode
        self.mode  = target
        self.since = frame_no
        return {
            "packet_type":    "cap_mode_transition",
            "ts_utc":         utc_now(),
            "frame_index":    frame_no,
            "from_mode":      from_mode,
            "to_mode":        target,
            "trigger":        trans_trigger,
            "spike_count_60": spike_count_60,
            "combined_60":    combined_count_60,
            "drop_count_60":  drop_count_60,
            "avg_score":      round(avg_score, 2),
        }


def render_trend(values, width=12):
    if not values:
        return " " * width
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    return "".join(_TREND_BLOCKS[int((v - lo) / span * 8)] for v in list(values)[-width:])


def write_event_clip(clip, clips_dir, fps, frame_size):
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    spike    = clip["spike"]
    type_str = "_".join(s.replace("SPIKE_", "") for s in spike["spikes"] if s != "SPIKE_COMBINED")
    fi       = spike["frame_index"]
    clip_path = str(clips_dir / f"spike_{type_str}_f{fi:06d}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    vw       = cv2.VideoWriter(clip_path, fourcc, fps, frame_size)
    for jpg_bytes in (clip.get("pre", []) + clip.get("post", [])):
        if jpg_bytes is None:
            continue
        arr   = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            vw.write(frame)
    vw.release()
    return clip_path


_CLUSTER_GAP = 30   # frames within this gap merge into one cluster


def _cluster_spikes(spikes, window=_CLUSTER_GAP):
    if not spikes:
        return []
    sorted_s = sorted(spikes, key=lambda x: x["frame_index"])
    clusters, current = [], [sorted_s[0]]
    for s in sorted_s[1:]:
        if s["frame_index"] - current[-1]["frame_index"] <= window:
            current.append(s)
        else:
            clusters.append(current)
            current = [s]
    clusters.append(current)
    return clusters


def _classify_cluster(spike_types):
    has_sound  = "SPIKE_SOUND"      in spike_types
    has_motion = "SPIKE_MOTION"     in spike_types
    has_blur   = "SPIKE_BLUR"       in spike_types
    has_over   = "SPIKE_OVERLIGHT"  in spike_types
    has_under  = "SPIKE_UNDERLIGHT" in spike_types
    if has_sound and has_motion:
        return "significant physical event (impact, door, person entering/leaving)"
    if has_sound:
        return "external audio event (voice, notification, tap, or mic bump)"
    if has_motion:
        return "silent movement (object repositioned, camera adjustment)"
    if has_blur:
        return "proximity event or camera movement"
    if has_over:
        return "sudden light increase (lamp on, window exposure)"
    if has_under:
        return "sudden light decrease (lamp off, obstruction)"
    return "unclassified environmental anomaly"


def _frames_to_ts(fi, fps):
    if fps <= 0:
        return "??:??"
    secs = fi / fps
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def generate_postrun_report(run_dir, session_id, start_time, end_time,
                             frame_index, pass_count, drop_count,
                             spike_log_path, baseline, log_path=None):
    duration_s = (end_time - start_time).total_seconds()
    fps = round(frame_index / duration_s, 1) if duration_s > 0 else 0

    spikes = []
    if os.path.exists(str(spike_log_path)):
        with open(str(spike_log_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        spikes.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    spike_type_counts = collections.Counter()
    severity_counts   = collections.Counter()
    for s in spikes:
        for st in s.get("spikes", []):
            spike_type_counts[st] += 1
        severity_counts[s.get("severity", "UNKNOWN")] += 1

    clusters    = _cluster_spikes(spikes)
    sev_order   = {"LOW": 0, "MEDIUM": 1, "ALERT": 2, "HIGH": 3}
    predictions = []
    for cluster in clusters:
        all_types = set()
        for ev in cluster:
            all_types.update(ev.get("spikes", []))
        max_sev = max((ev.get("severity", "LOW") for ev in cluster),
                      key=lambda x: sev_order.get(x, 0))
        predictions.append({
            "frame_range":   f"{cluster[0]['frame_index']}–{cluster[-1]['frame_index']}",
            "spike_types":   sorted(all_types - {"SPIKE_COMBINED"}),
            "interpretation": _classify_cluster(all_types),
            "severity":      max_sev,
            "event_count":   len(cluster),
        })

    # derive brightness from run log if no external baseline provided
    brightness_avg = None
    if baseline:
        brightness_avg = baseline.get("brightness_avg")
    if brightness_avg is None and log_path and os.path.exists(str(log_path)):
        try:
            log_pkts = [json.loads(l) for l in open(str(log_path)) if l.strip()]
            bvals = [p["mean_brightness"] for p in log_pkts
                     if p.get("packet_type") == "pseudo" and "mean_brightness" in p]
            if bvals:
                brightness_avg = sum(bvals) / len(bvals)
        except Exception:
            pass

    scene_labels = []
    if brightness_avg is not None and brightness_avg > 100:
        scene_labels.append("well-lit environment")
    else:
        scene_labels.append("low-light environment")
    scene_labels.append("movement detected — possible person or object activity"
                        if spike_type_counts.get("SPIKE_MOTION", 0) > 5
                        else "mostly static scene")
    scene_labels.append("acoustic activity present"
                        if spike_type_counts.get("SPIKE_SOUND", 0) > 3
                        else "quiet acoustic environment")
    if spike_type_counts.get("SPIKE_BLUR", 0) > 10:
        scene_labels.append("camera stability concern (blur events detected)")

    summary = {
        "session_id":        session_id,
        "start_time":        start_time.isoformat(),
        "end_time":          end_time.isoformat(),
        "duration_s":        round(duration_s, 1),
        "frames":            frame_index,
        "fps":               fps,
        "pass_count":        pass_count,
        "drop_count":        drop_count,
        "spike_total":       len(spikes),
        "spike_type_counts": dict(spike_type_counts),
        "severity_counts":   dict(severity_counts),
        "event_clusters":    len(clusters),
        "scene_labels":      scene_labels,
    }
    summary_path = run_dir / "run_summary.json"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # DROP frame analysis + CAP transitions from log
    drop_clusters = []
    drop_reason_counts = collections.Counter()
    cap_transitions = []
    if log_path and os.path.exists(str(log_path)):
        try:
            log_packets = [json.loads(l) for l in open(str(log_path)) if l.strip()]
            drop_frames = sorted(p["frame_index"] for p in log_packets
                                 if p.get("packet_type") == "pseudo" and p.get("verdict") == "DROP")
            for p in log_packets:
                if p.get("packet_type") == "pseudo" and p.get("verdict") == "DROP":
                    for r in p.get("reasons", []):
                        drop_reason_counts[r] += 1
                elif p.get("packet_type") == "cap_mode_transition":
                    cap_transitions.append(p)
            if drop_frames:
                cur = [drop_frames[0]]
                for f in drop_frames[1:]:
                    if f - cur[-1] <= 5:
                        cur.append(f)
                    else:
                        drop_clusters.append(cur)
                        cur = [f]
                drop_clusters.append(cur)
        except Exception:
            pass

    # Event timeline
    timeline = [("00:00", "session start", "")]
    for dc in drop_clusters:
        ts = _frames_to_ts(dc[0], fps)
        timeline.append((ts, f"DROP cluster  f{dc[0]}–{dc[-1]}", f"{len(dc)} frame{'s' if len(dc)>1 else ''}  motion_high"))
    for ct in cap_transitions:
        ts = _frames_to_ts(ct["frame_index"], fps)
        timeline.append((ts, f"CAP {ct['from_mode']} → {ct['to_mode']}", f"trigger={ct['trigger']}"))
    for i, cluster in enumerate(clusters, 1):
        ts = _frames_to_ts(cluster[0]["frame_index"], fps)
        all_types = set()
        for ev in cluster:
            all_types.update(ev.get("spikes", []))
        types_str = "+".join(sorted(t.replace("SPIKE_", "") for t in all_types if t != "SPIKE_COMBINED"))
        label = _classify_cluster(all_types)
        timeline.append((ts, f"event cluster {i}  [{types_str}]", label))
    timeline.sort(key=lambda x: x[0])
    timeline.append((_frames_to_ts(frame_index, fps), "session end", ""))

    lines = [
        "# PH6-Lite Post-Run Report",
        "> Authority: ADVISORY ONLY — PH6-LITE-POSTRUN-v0.2",
        "",
        "## Session Summary",
        "| Field | Value |",
        "|---|---|",
        f"| Session ID | `{session_id}` |",
        f"| Start | {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        f"| End | {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')} |",
        f"| Duration | {duration_s:.1f}s |",
        f"| Frames | {frame_index} |",
        f"| FPS | {fps} |",
        f"| PASS | {pass_count} |",
        f"| DROP | {drop_count} |",
        "",
        "## Spike Analysis",
        f"Total spike events: **{len(spikes)}** across **{len(clusters)}** clusters",
        "",
        "| Type | Count |",
        "|---|---|",
    ]
    for k, v in sorted(spike_type_counts.items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "| Severity | Count |",
        "|---|---|",
    ]
    for k in ["LOW", "MEDIUM", "ALERT", "HIGH"]:
        if k in severity_counts:
            lines.append(f"| {k} | {severity_counts[k]} |")
    lines += ["", "## Event Timeline", "```"]
    for ts, event, detail in timeline:
        detail_str = f"  — {detail}" if detail else ""
        lines.append(f"{ts}  {event}{detail_str}")
    lines += ["```", ""]

    if cap_transitions:
        peak_mode = max(cap_transitions, key=lambda p: _CAP_MODES.index(p["to_mode"]))
        lines += ["## CAP Posture History", "| Frame | Transition | Trigger |", "|---|---|---|"]
        for ct in cap_transitions:
            lines.append(f"| {ct['frame_index']} | {ct['from_mode']} → {ct['to_mode']} | {ct['trigger']} |")
        lines += ["", f"> Peak posture: **{peak_mode['to_mode']}** ({_CAP_CONFIG[peak_mode['to_mode']]['label']})", ""]

    if drop_reason_counts:
        lines += ["## DROP Frame Analysis", f"Total: {drop_count} frames dropped", ""]
        lines += ["| Reason | Count |", "|---|---|"]
        for k, v in drop_reason_counts.most_common():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += ["## Prediction / Interpretation", "> Advisory only. Not Lane-1 truth.", ""]
    if not predictions:
        lines.append("No significant events detected. Environment was stable throughout the session.")
    else:
        for i, pred in enumerate(predictions, 1):
            lines += [
                f"### Event {i} — frames {pred['frame_range']} [{pred['severity']}]",
                f"- **Detected:** {', '.join(pred['spike_types'])}",
                f"- **Interpretation:** {pred['interpretation']}",
                "- **Confidence:** medium",
                "",
            ]
    lines += [
        "## Scene Observation (Advisory)",
        "> `packet_type: scene_observation_advisory` | `authority: NONE` | `store: MRAM-S`",
        "",
    ]
    for label in scene_labels:
        lines.append(f"- {label}")
    lines += ["", "---", "*Generated by PH6-LITE-POSTRUN-v0.2. Advisory only.*", ""]

    report_path = run_dir / "postrun_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return summary_path, report_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",       default="0",         help="0 for camera, or path to video file")
    ap.add_argument("--camera",       type=int, default=0, help="camera index if --source 0")
    ap.add_argument("--log",          default="",          help="override log path (default: auto in run_dir)")
    ap.add_argument("--frames_dir",   default="",          help="override frames dir (default: auto in run_dir)")
    ap.add_argument("--save_mode",    choices=["pass_only", "all", "none"], default="pass_only")
    ap.add_argument("--bright_min",   type=float, default=40.0)
    ap.add_argument("--bright_max",   type=float, default=220.0)
    ap.add_argument("--lap_min",      type=float, default=40.0)
    ap.add_argument("--motion_max",   type=float, default=0.15)
    ap.add_argument("--max_frames",   type=int,   default=300, help="0 means unlimited")
    ap.add_argument("--probe_id",     default="",  help="probe identifier stamped into every packet")
    ap.add_argument("--llm",          action="store_true", help="enable LLM advisory via Ollama")
    ap.add_argument("--width",        type=int, default=0)
    ap.add_argument("--height",       type=int, default=0)
    ap.add_argument("--fps",          type=int, default=0)
    ap.add_argument("--signal_frames",default="")
    ap.add_argument("--audio",        action="store_true", help="capture mic audio")
    ap.add_argument("--audio_device", default="hw:1,0")
    ap.add_argument("--spike_log",    default="",          help="override spike log path")
    ap.add_argument("--save_video",   action="store_true", help="record full session video")
    ap.add_argument("--event_clips",  action="store_true", help="save video clips around spike events")
    ap.add_argument("--postrun",      action="store_true", help="generate post-run report after session")
    args = ap.parse_args()

    # ── run directory — hot/ for live writes, post/ for post-processing ────────
    run_ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(f"logs/run_{run_ts}")
    hot_dir  = run_dir / "hot"
    post_dir = run_dir / "post"
    hot_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    log_path       = Path(args.log)       if args.log       else hot_dir / "run_log.jsonl"
    spike_log_path = Path(args.spike_log) if args.spike_log else hot_dir / "spike_events.jsonl"
    frames_dir     = args.frames_dir      if args.frames_dir else str(hot_dir / "frames")

    # ── camera ─────────────────────────────────────────────────────────────────
    is_url = isinstance(args.source, str) and args.source.startswith("http")
    source = args.source if is_url else (args.camera if args.source == "0" else args.source)
    cap    = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if is_url else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    if args.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:    cap.set(cv2.CAP_PROP_FPS,          args.fps)

    cap_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    cap_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))          or 30
    frame_size = (cap_w, cap_h)

    # ── audio — written live to hot/ ───────────────────────────────────────────
    wav_path  = str(hot_dir / "run_audio.wav") if args.audio else None
    audio_cap = None
    if args.audio:
        audio_cap = AudioCapture(device=args.audio_device, wav_path=wav_path)
        audio_cap.start()

    # no live VideoWriter — video encoding deferred to post phase

    # ── event clip buffer (deque size set dynamically by CAP) ─────────────────
    clips_dir     = post_dir / "event_clips"
    frame_buffer  = collections.deque(maxlen=_CLIP_PRE)
    pending_clips = []   # collected live, rendered in post phase

    # ── session ────────────────────────────────────────────────────────────────
    session_id = make_session_id()
    writer     = PacketWriter(str(log_path))
    start_time = datetime.now(timezone.utc)

    writer.write({"packet_type": "session_start", "ts_utc": utc_now(),
                  "session_id": session_id, "message": "session started"})
    writer.write({"packet_type": "config", "ts_utc": utc_now(),
                  "session_id": session_id, "source": str(source),
                  "probe_id": args.probe_id, "save_mode": args.save_mode,
                  "bright_min": args.bright_min, "bright_max": args.bright_max,
                  "lap_min": args.lap_min, "motion_max": args.motion_max,
                  "max_frames": args.max_frames})

    prev_gray        = None
    frame_index      = 0
    continuity_count = 0
    confidence       = 0.5
    pass_count       = 0
    drop_count       = 0
    prev_metrics     = None

    baseline         = {"motion_avg": 0.01, "audio_rms_avg": 0.006, "brightness_avg": 100.0}
    spike_counts     = collections.Counter()
    spike_strip_window = collections.deque(maxlen=16)
    motion_history   = collections.deque(maxlen=60)
    audio_history    = collections.deque(maxlen=60)
    bright_history   = collections.deque(maxlen=60)
    cap_state        = CAPState()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1

            # JPEG-compress into clip buffer (low RAM cost)
            if args.event_clips:
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_buffer.append(jpg.tobytes())

            if video_writer:
                video_writer.write(frame)

            gray, mb, lv, mf, verdict, reasons = evaluate_frame(
                frame, prev_gray, args.bright_min, args.bright_max, args.lap_min, args.motion_max)

            ts          = utc_now()
            saved_path  = None
            saved_hash  = None

            if args.save_mode == "all" or (args.save_mode == "pass_only" and verdict == "PASS"):
                saved_path = save_frame(frame, frames_dir, frame_index)
                saved_hash = sha256_file(saved_path)

            pseudo = {"packet_type": "pseudo", "ts_utc": ts, "frame_index": frame_index,
                      "probe_id": args.probe_id, "mean_brightness": mb, "laplacian_var": lv,
                      "motion_fraction": mf, "verdict": verdict, "reasons": reasons}
            obs    = {"packet_type": "observation", "ts_utc": ts, "frame_index": frame_index,
                      "probe_id": args.probe_id, "image_path": saved_path,
                      "image_sha256": saved_hash, "width": int(frame.shape[1]), "height": int(frame.shape[0])}

            if verdict == "PASS":
                continuity_count += 1
                confidence = min(0.99, confidence + 0.05)
                state      = "stable" if continuity_count >= 10 else ("recovering" if continuity_count > 1 else "warming_up")
                advisory   = "continue"
                pass_count += 1
            else:
                continuity_count = 0
                confidence = max(0.1, confidence - 0.3)
                state      = "degraded"
                advisory   = "review_thresholds"
                drop_count += 1

            llm_backend = None
            if args.llm and verdict == "DROP":
                prompt = (f"Frame {frame_index} dropped. brightness={mb:.1f} sharpness={lv:.1f} "
                          f"motion={mf:.3f} reasons={reasons}. One-sentence advisory for the operator.")
                result = llm_ask("reason", prompt)
                advisory    = result.get("output", advisory).strip() or advisory
                llm_backend = result.get("backend")

            soso = {"packet_type": "soso", "ts_utc": ts, "frame_index": frame_index,
                    "probe_id": args.probe_id, "state": state, "continuity_count": continuity_count,
                    "confidence": confidence, "advisory": advisory}
            if llm_backend:
                soso["llm_backend"] = llm_backend

            writer.write(pseudo)
            writer.write(obs)
            writer.write(soso)

            rms, peak, clipping = 0.0, 0.0, False
            if audio_cap:
                rms, peak, clipping = audio_cap.get_metrics(duration_ms=200)
                writer.write({"packet_type": "audio", "ts_utc": ts, "frame_index": frame_index,
                               "probe_id": args.probe_id, "rms_level": rms, "peak_level": peak,
                               "clipping": clipping, "sample_rate": audio_cap.rate, "duration_ms": 200})

            metrics = {"mean_brightness": mb, "laplacian_var": lv, "motion_fraction": mf,
                       "audio_rms": rms, "audio_peak": peak}

            # pre-spike warning
            presoak = detect_presoak(frame_index, ts, metrics, prev_metrics, baseline)
            if presoak:
                writer.write(presoak)
                print(f"       ~~ WARN {presoak['warnings']}  score={presoak['presoak_score']}", flush=True)

            # spike detection
            spike = detect_spikes(frame_index, ts, metrics, baseline)
            update_baseline(baseline, metrics)
            prev_metrics = metrics
            motion_history.append(mf)
            audio_history.append(rms)
            bright_history.append(mb)

            # spike logging
            spike_char = "."
            traffic    = "GREEN"
            if spike:
                writer.write(spike)
                with open(spike_log_path, "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(spike, separators=(",", ":")) + "\n")
                for s in spike["spikes"]:
                    spike_counts[s] += 1
                spike_char = "".join(_SPIKE_CHARS[s] for s in spike["spikes"] if s in _SPIKE_CHARS)
                traffic    = "RED" if spike["severity"] in ("HIGH", "ALERT") else "YELLOW"
            spike_strip_window.append(spike_char)
            strip = "".join(spike_strip_window).rjust(16, ".")

            # CAP mode update — pass is_drop so density counts include DROP frames
            cap_pkt = cap_state.update(frame_index, spike, presoak, is_drop=(verdict == "DROP"))
            if cap_pkt:
                writer.write(cap_pkt)
                cfg     = _CAP_CONFIG[cap_pkt["to_mode"]]
                cap_beh = apply_cap_mode(cap_pkt["to_mode"], fps=cap_fps)
                # resize pre-event buffer to match new CAP posture
                new_pre = max(cap_beh["prebuffer_frames"], _CLIP_PRE)
                if frame_buffer.maxlen != new_pre:
                    frame_buffer = collections.deque(frame_buffer, maxlen=new_pre)
                print(f"  ◆ CAP {cap_pkt['from_mode']} → {cap_pkt['to_mode']}  "
                      f"({cfg['label']})  trigger={cap_pkt['trigger']}", flush=True)

            cap_beh = apply_cap_mode(cap_state.mode, fps=cap_fps)

            # accumulate post-event frames for pending clips; start new clip on spike
            if args.event_clips:
                new_pending = []
                for clip in pending_clips:
                    clip["post"].append(frame_buffer[-1] if frame_buffer else None)
                    clip["remaining"] -= 1
                    if clip["remaining"] > 0:
                        new_pending.append(clip)
                    else:
                        new_pending.append(clip)   # keep until post phase
                        clip["done"] = True
                pending_clips = new_pending
                if spike and len(pending_clips) < 5:
                    post_frames = max(cap_beh["postbuffer_frames"], _CLIP_POST)
                    pending_clips.append({"spike": spike, "pre": list(frame_buffer),
                                          "post": [], "remaining": post_frames})

            if args.signal_frames:
                signals = [int(x) for x in args.signal_frames.split(",")]
                if frame_index == signals[0]:
                    print(f">>> COVER LENS NOW (frame {frame_index})", flush=True)
                elif len(signals) > 1 and frame_index == signals[1]:
                    print(f">>> UNCOVER LENS NOW (frame {frame_index})", flush=True)

            if not cap_beh["quiet_terminal"]:
                print(f"f={frame_index:4d}  {verdict}  {state:<12}  conf={confidence:.2f}  [{strip}]  {traffic}", flush=True)
                if spike:
                    print(f"       >> SPIKE {spike['spikes']}  score={spike['spike_score']}  {spike['severity']}"
                          f"  motion={mf:.3f}  rms={rms:.3f}  bright={mb:.1f}", flush=True)
            elif frame_index % 50 == 0:
                print(f"f={frame_index:4d}  CAP:EVENT_BURST  [{strip}]  {traffic}  (quiet terminal)", flush=True)

            prev_gray = gray
            if args.max_frames and frame_index >= args.max_frames:
                break

    finally:
        cap.release()
        if audio_cap:
            audio_cap.stop()

        end_time = datetime.now(timezone.utc)

        # scene observation advisory
        scene_labels_live = []
        if baseline["brightness_avg"] > 100:
            scene_labels_live.append("well-lit environment")
        else:
            scene_labels_live.append("low-light environment")
        if spike_counts.get("SPIKE_MOTION", 0) > 5:
            scene_labels_live.append("movement detected")
        if spike_counts.get("SPIKE_SOUND", 0) > 3:
            scene_labels_live.append("acoustic activity present")

        writer.write({"packet_type": "scene_observation_advisory", "ts_utc": utc_now(),
                      "frame_index": frame_index, "authority": "NONE", "store": "MRAM-S",
                      "scene_labels": scene_labels_live,
                      "baseline_brightness": round(baseline["brightness_avg"], 1),
                      "baseline_motion": round(baseline["motion_avg"], 4),
                      "baseline_audio_rms": round(baseline["audio_rms_avg"], 4)})

        writer.write({"packet_type": "session_end", "ts_utc": utc_now(),
                      "session_id": session_id, "frames_processed": frame_index,
                      "message": "session ended"})

    writer.close()

    duration_s = (end_time - start_time).total_seconds()
    fps_actual = round(frame_index / duration_s, 1) if duration_s > 0 else 0
    total_spikes = sum(v for k, v in spike_counts.items() if k != "SPIKE_COMBINED")

    print(f"\nDone. Frames: {frame_index}  FPS: {fps_actual}  Duration: {duration_s:.1f}s")
    print(f"Run dir:    {run_dir}")
    print(f"  hot/:    {hot_dir}")
    print(f"  post/:   {post_dir}")
    print(f"Log:        {log_path}")
    print(f"Spike log:  {spike_log_path}")
    if wav_path:
        print(f"Audio WAV:  {wav_path}")

    print(f"\n── Spike Counters ─────────────────")
    print(f"  Motion spikes:      {spike_counts['SPIKE_MOTION']}")
    print(f"  Sound spikes:       {spike_counts['SPIKE_SOUND']}")
    print(f"  Overlight spikes:   {spike_counts['SPIKE_OVERLIGHT']}")
    print(f"  Underlight spikes:  {spike_counts['SPIKE_UNDERLIGHT']}")
    print(f"  Blur spikes:        {spike_counts['SPIKE_BLUR']}")
    print(f"  Combined events:    {spike_counts['SPIKE_COMBINED']}")
    print(f"  Total events:       {total_spikes}")

    print(f"\n── Sensor Trends (last {len(motion_history)} frames) ──")
    print(f"  motion:  {render_trend(motion_history)}")
    print(f"  audio:   {render_trend(audio_history)}")
    print(f"  bright:  {render_trend(bright_history)}")

    # ── post phase: render clips, encode video, generate report ───────────────
    print(f"\n── Post phase… ──")

    if args.event_clips and pending_clips:
        print(f"  Rendering {len(pending_clips)} event clip(s)…")
        for clip in pending_clips:
            write_event_clip(clip, clips_dir, cap_fps, frame_size)

    if args.save_video:
        frames_path = Path(frames_dir)
        if frames_path.exists():
            frame_files = sorted(frames_path.glob("*.jpg"))
            if frame_files:
                video_out = str(post_dir / "run_video.mp4")
                print(f"  Encoding video from {len(frame_files)} frames → {video_out}")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(video_out, fourcc, cap_fps, frame_size)
                for ff in frame_files:
                    img = cv2.imread(str(ff))
                    if img is not None:
                        vw.write(img)
                vw.release()
                print(f"  Video: {video_out}")

    if args.postrun:
        print(f"  Generating report…")
        summary_path, report_path = generate_postrun_report(
            post_dir, session_id, start_time, end_time,
            frame_index, pass_count, drop_count, spike_log_path, baseline,
            log_path=log_path)
        print(f"  Summary: {summary_path}")
        print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
