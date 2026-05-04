"""
cram_writer.py — PH6/CRAM compliant write engine  (v0.4)

Authority note
--------------
This writer marks storage durability only (store, durable fields on packets).
PH6 authority — PASS/DROP verdicts — belongs to PSEUDO. This writer does not
decide it and must not be extended to do so.

Write path (all modes):
    RAM (ring buffer, FIFO, no eviction) →
    [annotate store/durable] →
    hash-chain →
    stage in buffer →
    flush to JSONL on threshold

Mode semantics
--------------
balanced:
    Batched JSONL append + fdatasync (fsync fallback). Default for Pi 5.

forensic:
    JSONL append + fsync + parent directory fsync, every packet.
    This is Forensic JSONL Mode — it improves durability significantly
    but is NOT full tmp→rename segment atomicity (Forensic Segment Mode).
    A partial write at the final line is detectable as malformed JSON.

burst:
    Batched JSONL append with relaxed sync. Intended for scenario/lab
    throughput testing only. Not authoritative until CRAM commit completes.

Full tmp→rename atomic segment commit (write(tmp)→fsync→rename→fsync(dir))
is implemented by SegmentCRAMWriter (also in this module).

Classes
-------
CRAMWriter          — durable JSONL batching layer (v0.3 semantics)
SegmentCRAMWriter   — full forensic segment atomicity (v0.4)
"""

import os
import json
import hashlib
import threading
import collections
from pathlib import Path

_GENESIS_HASH = "0" * 128  # 64-byte blake2b = 128 hex chars

# ── mode registry ─────────────────────────────────────────────────────────────

CRAM_MODES = {
    "forensic": {
        "flush_every": 1,
        "sync": "fsync",
        "dir_sync": True,
        "stage_store": "RAM_STAGE",
        "commit_store": "CRAM",
    },
    "balanced": {
        "flush_every": 10,
        "sync": "fdatasync",
        "dir_sync": False,
        "stage_store": "RAM_STAGE",
        "commit_store": "CRAM",
    },
    "burst": {
        "flush_every": 50,
        "sync": "none",
        "dir_sync": False,
        "stage_store": "RAM_STAGE",
        "commit_store": "CRAM",
    },
}


def sync_fd(fd: int, strategy: str) -> None:
    """Sync an open file descriptor using the named strategy."""
    if strategy == "none":
        return
    if strategy == "fdatasync" and hasattr(os, "fdatasync"):
        os.fdatasync(fd)
    else:
        os.fsync(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ── writer ────────────────────────────────────────────────────────────────────

class CRAMWriter:
    """
    Compliant CRAM write engine.

    Buffer is staging only — zero authority over what gets written.
    Backpressure blocks the caller rather than dropping packets.
    The writer annotates each packet with store/durable before hashing;
    it does not touch verdict fields or authority assignments.

    Parameters
    ----------
    log_path     : final JSONL path
    buffer_size  : max packets held in RAM before forced flush (backpressure)
    flush_every  : override mode's default flush cadence (None = use mode)
    mode         : "forensic" | "balanced" | "burst"  (default: "balanced")
    """

    def __init__(self, log_path, buffer_size=64, flush_every=None, mode="balanced"):
        if mode not in CRAM_MODES:
            raise ValueError(f"mode must be one of {list(CRAM_MODES)}, got {mode!r}")

        cfg = CRAM_MODES[mode]
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self._sync_strategy = cfg["sync"]
        self._dir_sync = cfg["dir_sync"]
        self._commit_store = cfg["commit_store"]

        self.flush_every = flush_every if flush_every is not None else cfg["flush_every"]
        self.buffer_size = max(self.flush_every, buffer_size)

        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._packet_seq = 0
        self._pending = 0
        self._prev_hash = _GENESIS_HASH
        self._fh = open(self.log_path, "a", encoding="utf-8")

    # ── public ───────────────────────────────────────────────────────────────

    def write(self, packet):
        """
        Validate, annotate, hash-chain, stage, and flush when threshold is met.
        Blocks if buffer is full (backpressure — never drops).
        Returns the assigned packet_seq.
        """
        with self._lock:
            self._packet_seq += 1
            packet["packet_seq"] = self._packet_seq
            _validate_packet(packet)

            # Storage durability annotation — writer's only authority is marking
            # where data lives and whether it has been committed to durable storage.
            # Packets with an existing store field (e.g. virtual_token: MRAM-S)
            # are not overridden.
            if "store" not in packet:
                packet["store"] = self._commit_store
                packet["durable"] = True

            # Hash chain: hash canonical JSON (with prev_hash, without packet_hash)
            packet["prev_hash"] = self._prev_hash
            canonical = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            packet_hash = hashlib.blake2b(canonical.encode()).hexdigest()
            packet["packet_hash"] = packet_hash
            self._prev_hash = packet_hash

            # Backpressure: block until buffer has room
            while len(self._buffer) >= self.buffer_size:
                self._flush_locked()

            self._buffer.append(
                json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
            )
            self._pending += 1

            if self._pending >= self.flush_every:
                self._flush_locked()

        return packet["packet_seq"]

    def flush(self):
        """Force flush all buffered packets to disk."""
        with self._lock:
            self._flush_locked()

    def close(self):
        """Flush and close."""
        self.flush()
        try:
            self._fh.close()
        except OSError:
            pass

    @property
    def packet_seq(self):
        with self._lock:
            return self._packet_seq

    # ── internal ─────────────────────────────────────────────────────────────

    def _flush_locked(self):
        """
        Append buffered lines, sync according to mode, optionally sync the dir.
        Caller must hold self._lock.
        """
        if not self._buffer:
            return

        lines = list(self._buffer)
        self._buffer.clear()
        self._pending = 0

        blob = "\n".join(lines) + "\n"
        self._fh.write(blob)
        self._fh.flush()

        sync_fd(self._fh.fileno(), self._sync_strategy)

        if self._dir_sync:
            _fsync_dir(self.log_path.parent)


# ── packet validator ──────────────────────────────────────────────────────────

def _validate_packet(packet):
    if not isinstance(packet, dict):
        raise ValueError("packet must be dict")
    for field in ("packet_type", "ts_utc", "packet_seq"):
        if field not in packet:
            raise ValueError(f"missing {field}")

    ptype = packet["packet_type"]
    allowed = {"session_start", "config", "pseudo", "observation", "soso", "audio",
               "spike_event", "warning_event", "scene_observation_advisory", "session_end",
               "cap_mode_transition", "virtual_token"}
    if ptype not in allowed:
        raise ValueError(f"invalid packet_type: {ptype}")

    if not isinstance(packet["packet_seq"], int):
        raise ValueError("packet_seq must be int")

    if ptype in {"pseudo", "observation", "soso"}:
        if not isinstance(packet.get("frame_index"), int):
            raise ValueError(f"{ptype} frame_index must be int")

    if ptype == "pseudo":
        for k in ("mean_brightness", "laplacian_var", "motion_fraction", "verdict", "reasons"):
            if k not in packet:
                raise ValueError(f"pseudo missing {k}")
        if packet["verdict"] not in {"PASS", "DROP"}:
            raise ValueError("verdict must be PASS or DROP")
        if not isinstance(packet["reasons"], list):
            raise ValueError("reasons must be list")

    if ptype == "observation":
        for k in ("image_path", "image_sha256", "width", "height"):
            if k not in packet:
                raise ValueError(f"observation missing {k}")

    if ptype == "soso":
        for k in ("state", "continuity_count", "confidence", "advisory"):
            if k not in packet:
                raise ValueError(f"soso missing {k}")

    if ptype == "audio":
        for k in ("rms_level", "peak_level", "clipping", "sample_rate", "duration_ms"):
            if k not in packet:
                raise ValueError(f"audio missing {k}")

    if ptype == "config":
        for k in ("session_id", "source", "save_mode", "bright_min",
                  "bright_max", "lap_min", "motion_max", "max_frames"):
            if k not in packet:
                raise ValueError(f"config missing {k}")

    if ptype == "session_start":
        for k in ("session_id", "message"):
            if k not in packet:
                raise ValueError(f"session_start missing {k}")

    if ptype == "spike_event":
        for k in ("frame_index", "spikes", "severity", "spike_score", "metrics"):
            if k not in packet:
                raise ValueError(f"spike_event missing {k}")
        if not isinstance(packet["spikes"], list):
            raise ValueError("spike_event spikes must be list")

    if ptype == "warning_event":
        for k in ("frame_index", "warnings", "presoak_score", "metrics"):
            if k not in packet:
                raise ValueError(f"warning_event missing {k}")
        if not isinstance(packet["warnings"], list):
            raise ValueError("warning_event warnings must be list")

    if ptype == "scene_observation_advisory":
        for k in ("frame_index", "authority", "store"):
            if k not in packet:
                raise ValueError(f"scene_observation_advisory missing {k}")

    if ptype == "session_end":
        for k in ("session_id", "frames_processed", "message"):
            if k not in packet:
                raise ValueError(f"session_end missing {k}")

    if ptype == "cap_mode_transition":
        for k in ("frame_index", "from_mode", "to_mode", "trigger"):
            if k not in packet:
                raise ValueError(f"cap_mode_transition missing {k}")

    if ptype == "virtual_token":
        for k in ("frame_index", "token_type", "token_id", "authority", "store"):
            if k not in packet:
                raise ValueError(f"virtual_token missing {k}")
        if packet["authority"] != "NONE":
            raise ValueError("virtual_token authority must be NONE")
        if packet["store"] != "MRAM-S":
            raise ValueError("virtual_token store must be MRAM-S")


# ── segment writer ────────────────────────────────────────────────────────────

class SegmentCRAMWriter:
    """
    Full Forensic Segment Mode write engine.  (v0.4)

    Each segment is an independent JSONL file committed atomically:
        write(tmp) → fsync(tmp) → rename(tmp → final) → fsync(dir)

    A .tmp file left on disk indicates a crash mid-write and is not a
    valid segment — readers must ignore or quarantine it.

    Hash chain spans segments: the prev_hash of segment N's first packet
    equals the packet_hash of segment N-1's last packet (genesis for seg 0).

    Segment files: <log_dir>/seg_NNNNNN.jsonl  (six-digit zero-padded)

    Parameters
    ----------
    log_dir      : directory where segment files land
    segment_size : packets per segment before an atomic commit fires
    buffer_size  : max packets in RAM before backpressure blocks the caller
    """

    def __init__(self, log_dir, segment_size=10, buffer_size=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.segment_size = max(1, segment_size)
        self.buffer_size = max(self.segment_size, buffer_size or self.segment_size * 4)

        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._packet_seq = 0
        self._segment_id = 0
        self._pending = 0
        self._prev_hash = _GENESIS_HASH

    # ── public ───────────────────────────────────────────────────────────────

    def write(self, packet):
        """
        Annotate, hash-chain, stage, and commit when segment_size is reached.
        Blocks if buffer is full (backpressure — never drops).
        Returns the assigned packet_seq.
        """
        with self._lock:
            self._packet_seq += 1
            packet["packet_seq"] = self._packet_seq
            _validate_packet(packet)

            if "store" not in packet:
                packet["store"] = "CRAM"
                packet["durable"] = True

            packet["prev_hash"] = self._prev_hash
            canonical = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            packet_hash = hashlib.blake2b(canonical.encode()).hexdigest()
            packet["packet_hash"] = packet_hash
            self._prev_hash = packet_hash

            while len(self._buffer) >= self.buffer_size:
                self._commit_locked()

            self._buffer.append(
                json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
            )
            self._pending += 1

            if self._pending >= self.segment_size:
                self._commit_locked()

        return packet["packet_seq"]

    def flush(self):
        """Commit any buffered packets as a partial segment."""
        with self._lock:
            if self._pending:
                self._commit_locked()

    def close(self):
        """Flush and finalize."""
        self.flush()

    @property
    def packet_seq(self):
        with self._lock:
            return self._packet_seq

    @property
    def segment_id(self):
        """Most recently committed segment number (0 = none committed yet)."""
        with self._lock:
            return self._segment_id

    # ── internal ─────────────────────────────────────────────────────────────

    def _seg_path(self, seg_id):
        return self.log_dir / f"seg_{seg_id:06d}.jsonl"

    def _commit_locked(self):
        """
        Drain buffer to a new segment via tmp→fsync→rename→fsync(dir).
        Caller must hold self._lock.
        """
        if not self._buffer:
            return

        self._segment_id += 1
        final = self._seg_path(self._segment_id)
        tmp = final.with_suffix(".jsonl.tmp")

        blob = ("\n".join(self._buffer) + "\n").encode("utf-8")
        self._buffer.clear()
        self._pending = 0

        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, blob)
            os.fsync(fd)
        finally:
            os.close(fd)

        os.rename(str(tmp), str(final))
        _fsync_dir(self.log_dir)
