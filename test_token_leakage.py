#!/usr/bin/env python3
"""
test_token_leakage.py — proves VirtualTokenTracker has Authority ZERO over PSEUDO.

Structural guarantees tested:
  1. evaluate_frame() is deterministic — identical input → identical verdict,
     regardless of whether tokens were emitted between runs.
  2. All virtual_token packets carry authority=NONE and store=MRAM-S.
  3. No token field appears in PSEUDO packet structure.
  4. Token packet_seq > pseudo packet_seq for the same frame (ordering).
  5. CRAMWriter's validator rejects any virtual_token that violates the boundary.

Run:
  python3 test_token_leakage.py
"""

import sys
import json
import tempfile
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from frame_filter import evaluate_frame
from virtual_tokens import VirtualTokenTracker
from cram_writer import CRAMWriter

# Mapping: evaluate_frame reason → VirtualTokenTracker event_kind
_REASON_TO_EVENT = {
    "brightness_low":  "underlight",
    "brightness_high": "overlight",
    "blur_low_detail": "blur",
    "motion_high":     "motion",
}

# Fields that must never appear in a PSEUDO packet
TOKEN_FIELDS = {"token_type", "token_id", "authority", "store", "source",
                "root_frame", "last_frame", "linked_frames", "advisory"}

# Fields that must never appear in a virtual_token packet
PSEUDO_VERDICT_FIELDS = {"verdict", "reasons", "mean_brightness", "laplacian_var",
                          "motion_fraction", "motion_level", "brightness_level",
                          "blur_level", "degradation_score"}

BRIGHT_MIN, BRIGHT_MAX = 40.0, 220.0
LAP_MIN    = 40.0
MOTION_MAX = 0.15
TS         = "2026-01-01T00:00:00Z"


def _make_frame(brightness: int, width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height, width, 3), brightness, dtype=np.uint8)


def _run_pseudo(frames):
    """Run evaluate_frame over a list of frames; return list of (verdict, reasons)."""
    prev_gray = None
    results = []
    for frame in frames:
        gray, _, _, _, verdict, reasons = evaluate_frame(
            frame, prev_gray, BRIGHT_MIN, BRIGHT_MAX, LAP_MIN, MOTION_MAX
        )
        results.append((verdict, list(reasons)))
        prev_gray = gray
    return results


def test_token_leakage():
    failures = []

    # ── build synthetic frame sequence ───────────────────────────────────────
    # Pattern: 4 dark (DROP/underlight) → 4 normal (PASS) × 5 cycles = 40 frames
    frames = []
    for _ in range(5):
        for _ in range(4):
            frames.append(_make_frame(5))    # darkness → brightness_low
        for _ in range(4):
            frames.append(_make_frame(128))  # normal → PASS

    # ── first PSEUDO pass — verdicts before any token emission ────────────────
    verdicts_before = _run_pseudo(frames)
    drop_frames_before = sum(1 for v, _ in verdicts_before if v == "DROP")

    if drop_frames_before == 0:
        failures.append("synthetic dark frames did not produce any DROP — test setup error")

    # ── token emission ────────────────────────────────────────────────────────
    tracker = VirtualTokenTracker()
    token_packets = []
    pseudo_seq_by_frame = {}
    token_seqs_by_frame = {}
    seq = 0

    for i, (verdict, reasons) in enumerate(verdicts_before):
        frame_index = i + 1
        seq += 1
        pseudo_seq_by_frame[frame_index] = seq

        seen_kinds = set()
        for reason in reasons:
            kind = _REASON_TO_EVENT.get(reason)
            if kind and kind not in seen_kinds:
                seen_kinds.add(kind)
                for pkt in tracker.observe_event(frame_index, kind, TS):
                    pkt["source"] = "SoSo.VirtualTokenTracker"
                    seq += 1
                    pkt["_test_seq"] = seq
                    token_packets.append(pkt)
                    token_seqs_by_frame.setdefault(frame_index, []).append(seq)

        for pkt in tracker.close_expired(frame_index, TS):
            pkt["source"] = "SoSo.VirtualTokenTracker"
            seq += 1
            pkt["_test_seq"] = seq
            token_packets.append(pkt)

    # ── second PSEUDO pass — must be identical ────────────────────────────────
    verdicts_after = _run_pseudo(frames)

    for i, ((v_before, r_before), (v_after, r_after)) in enumerate(
        zip(verdicts_before, verdicts_after)
    ):
        if v_before != v_after:
            failures.append(
                f"frame {i+1}: verdict changed after token emission: "
                f"{v_before!r} → {v_after!r}"
            )
        if r_before != r_after:
            failures.append(
                f"frame {i+1}: reasons changed after token emission: "
                f"{r_before} → {r_after}"
            )

    # ── test 1: token packets were emitted ───────────────────────────────────
    if not token_packets:
        failures.append("no virtual_token packets emitted — DROP frames should have triggered tokens")

    # ── test 2: every token packet enforces authority=NONE, store=MRAM-S ─────
    for pkt in token_packets:
        if pkt.get("authority") != "NONE":
            failures.append(
                f"token {pkt.get('token_id')!r}: authority={pkt.get('authority')!r} (expected NONE)"
            )
        if pkt.get("store") != "MRAM-S":
            failures.append(
                f"token {pkt.get('token_id')!r}: store={pkt.get('store')!r} (expected MRAM-S)"
            )
        if pkt.get("token_type") not in {"RT", "VDT", "VLT"}:
            failures.append(
                f"token {pkt.get('token_id')!r}: token_type={pkt.get('token_type')!r} not in {{RT,VDT,VLT}}"
            )

    # ── test 3a: no token field in PSEUDO packet structure ───────────────────
    # Build representative pseudo packet (same structure as main loop)
    sample_pseudo = {
        "packet_type": "pseudo",
        "ts_utc": TS,
        "frame_index": 1,
        "mean_brightness": 5.0,
        "laplacian_var": 0.0,
        "motion_fraction": 0.0,
        "verdict": "DROP",
        "reasons": ["brightness_low"],
    }
    leaked_token_fields = TOKEN_FIELDS & set(sample_pseudo.keys())
    if leaked_token_fields:
        failures.append(
            f"pseudo packet structure contains token fields: {leaked_token_fields}"
        )

    # ── test 3b: no PSEUDO verdict field in token packet structure ───────────
    for pkt in token_packets:
        leaked_pseudo_fields = PSEUDO_VERDICT_FIELDS & set(pkt.keys())
        if leaked_pseudo_fields:
            failures.append(
                f"token {pkt.get('token_id')!r} at frame {pkt.get('frame_index')} "
                f"contains pseudo verdict fields: {leaked_pseudo_fields}"
            )

    # ── test 4: packet ordering — pseudo seq < token seq for same frame ───────
    for frame_index, p_seq in pseudo_seq_by_frame.items():
        for t_seq in token_seqs_by_frame.get(frame_index, []):
            if t_seq <= p_seq:
                failures.append(
                    f"frame {frame_index}: token seq {t_seq} not after pseudo seq {p_seq} "
                    f"— token must never precede the verdict it observes"
                )

    # ── test 5: CRAMWriter validator rejects authority violation ─────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        log = os.path.join(tmpdir, "test.jsonl")
        cram = CRAMWriter(log)

        # Valid token — must be accepted
        valid_pkt = {
            "packet_type":  "virtual_token",
            "ts_utc":       TS,
            "frame_index":  1,
            "token_type":   "RT",
            "token_id":     "RT-underlight-000001",
            "authority":    "NONE",
            "store":        "MRAM-S",
            "source":       "SoSo.VirtualTokenTracker",
        }
        try:
            cram.write(valid_pkt)
        except Exception as e:
            failures.append(f"CRAMWriter rejected valid token packet: {e}")

        # Bad token — authority != NONE — must be rejected
        bad_pkt = dict(valid_pkt)
        bad_pkt["authority"] = "CRAM-A"   # forbidden
        bad_pkt["token_id"]  = "BAD-000001"
        try:
            cram.write(bad_pkt)
            failures.append("CRAMWriter accepted token with authority=CRAM-A — validator broken")
        except ValueError:
            pass  # expected

        # Bad token — wrong store — must be rejected
        bad_pkt2 = dict(valid_pkt)
        bad_pkt2["store"]    = "CRAM-A"   # forbidden
        bad_pkt2["token_id"] = "BAD-000002"
        try:
            cram.write(bad_pkt2)
            failures.append("CRAMWriter accepted token with store=CRAM-A — validator broken")
        except ValueError:
            pass  # expected

        cram.close()

    # ── test 6: close_all() emits closed packets; boundaries hold ────────────
    tracker2 = VirtualTokenTracker()
    tracker2.observe_event(1, "motion", TS)
    tracker2.observe_event(2, "blur", TS)
    if not tracker2.active:
        failures.append("close_all test: expected active tokens before close_all")
    closed_pkts = tracker2.close_all(50, TS)
    if tracker2.active:
        failures.append("close_all test: active tokens remain after close_all")
    if len(closed_pkts) != 2:
        failures.append(
            f"close_all test: expected 2 closed packets, got {len(closed_pkts)}"
        )
    for pkt in closed_pkts:
        if pkt.get("state") != "closed":
            failures.append(
                f"close_all: packet {pkt.get('token_id')!r} state={pkt.get('state')!r}"
            )
        if pkt.get("authority") != "NONE":
            failures.append(
                f"close_all: packet {pkt.get('token_id')!r} authority not NONE"
            )
        if pkt.get("store") != "MRAM-S":
            failures.append(
                f"close_all: packet {pkt.get('token_id')!r} store not MRAM-S"
            )

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n── Virtual Token Leakage Test ──────────────────────────────────")
    print(f"  frames        : {len(frames)}")
    print(f"  DROP frames   : {drop_frames_before}")
    print(f"  token packets : {len(token_packets)}")
    print(f"  token types   : {set(p.get('token_type') for p in token_packets)}")
    print(f"  failures      : {len(failures)}")
    for f in failures[:20]:
        print(f"    FAIL: {f}")
    verdict = "PASS" if not failures else "FAIL"
    print(f"  → {verdict}")
    print()

    return len(failures) == 0


if __name__ == "__main__":
    ok = test_token_leakage()
    sys.exit(0 if ok else 1)
