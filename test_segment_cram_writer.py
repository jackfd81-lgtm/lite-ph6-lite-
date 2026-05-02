"""
Tests for SegmentCRAMWriter (v0.4 — full tmp→rename segment atomicity).
"""

import json
import hashlib
import tempfile
from pathlib import Path

from cram_writer import SegmentCRAMWriter, _GENESIS_HASH


def _pseudo(frame_index, seq_hint=None):
    p = {
        "packet_type": "pseudo",
        "ts_utc": "2026-05-02T00:00:00Z",
        "frame_index": frame_index,
        "mean_brightness": 100.0,
        "laplacian_var": 50.0,
        "motion_fraction": 0.1,
        "verdict": "PASS",
        "reasons": [],
    }
    return p


def _segments(log_dir):
    return sorted(Path(log_dir).glob("seg_*.jsonl"))


def _tmps(log_dir):
    return list(Path(log_dir).glob("*.tmp"))


# ── segment creation ──────────────────────────────────────────────────────────

def test_segments_created_in_order():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=2)
        for i in range(6):
            w.write(_pseudo(i))
        w.close()
        segs = _segments(d)
        assert len(segs) == 3, segs
        assert [s.name for s in segs] == [
            "seg_000001.jsonl", "seg_000002.jsonl", "seg_000003.jsonl"
        ]


def test_segment_contains_correct_packet_count():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=3)
        for i in range(6):
            w.write(_pseudo(i))
        w.close()
        for seg in _segments(d):
            lines = [l for l in seg.read_text().splitlines() if l.strip()]
            assert len(lines) == 3, f"{seg.name}: expected 3 packets, got {len(lines)}"


def test_no_tmp_files_after_normal_commit():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=2)
        for i in range(4):
            w.write(_pseudo(i))
        w.close()
        assert _tmps(d) == [], f"stray .tmp files: {_tmps(d)}"


# ── hash chain ────────────────────────────────────────────────────────────────

def test_hash_chain_within_segment():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=4)
        for i in range(4):
            w.write(_pseudo(i))
        w.close()
        packets = [json.loads(l) for l in _segments(d)[0].read_text().splitlines() if l.strip()]
        assert packets[0]["prev_hash"] == _GENESIS_HASH
        for i in range(1, len(packets)):
            assert packets[i]["prev_hash"] == packets[i - 1]["packet_hash"]


def test_hash_chain_spans_segments():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=2)
        for i in range(4):
            w.write(_pseudo(i))
        w.close()
        segs = _segments(d)
        seg1 = [json.loads(l) for l in segs[0].read_text().splitlines() if l.strip()]
        seg2 = [json.loads(l) for l in segs[1].read_text().splitlines() if l.strip()]
        assert seg2[0]["prev_hash"] == seg1[-1]["packet_hash"]


def test_packet_hash_is_correct():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=1)
        w.write(_pseudo(0))
        w.close()
        pkt = json.loads(_segments(d)[0].read_text().strip())
        stored_hash = pkt.pop("packet_hash")
        pkt["prev_hash"] = pkt["prev_hash"]  # already present
        canonical = json.dumps(pkt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        expected = hashlib.blake2b(canonical.encode()).hexdigest()
        assert stored_hash == expected


# ── store / durable annotation ────────────────────────────────────────────────

def test_store_durable_annotated():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=1)
        w.write(_pseudo(0))
        w.close()
        pkt = json.loads(_segments(d)[0].read_text().strip())
        assert pkt["store"] == "CRAM"
        assert pkt["durable"] is True


def test_mram_s_store_not_overridden():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=1)
        p = {
            "packet_type": "virtual_token",
            "ts_utc": "2026-05-02T00:00:00Z",
            "frame_index": 0,
            "token_type": "RT",
            "token_id": "rt_0",
            "authority": "NONE",
            "store": "MRAM-S",
        }
        w.write(p)
        w.close()
        pkt = json.loads(_segments(d)[0].read_text().strip())
        assert pkt["store"] == "MRAM-S"
        assert "durable" not in pkt


# ── flush / partial segment ───────────────────────────────────────────────────

def test_flush_commits_partial_segment():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=10)
        for i in range(3):
            w.write(_pseudo(i))
        assert _segments(d) == []
        w.flush()
        segs = _segments(d)
        assert len(segs) == 1
        lines = [l for l in segs[0].read_text().splitlines() if l.strip()]
        assert len(lines) == 3


def test_close_flushes_partial_segment():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=10)
        w.write(_pseudo(0))
        w.close()
        segs = _segments(d)
        assert len(segs) == 1


# ── packet_seq / segment_id properties ───────────────────────────────────────

def test_packet_seq_increments():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=5)
        for i in range(7):
            seq = w.write(_pseudo(i))
            assert seq == i + 1
        assert w.packet_seq == 7


def test_segment_id_after_commits():
    with tempfile.TemporaryDirectory() as d:
        w = SegmentCRAMWriter(d, segment_size=2)
        assert w.segment_id == 0
        for i in range(4):
            w.write(_pseudo(i))
        assert w.segment_id == 2
        w.flush()
        assert w.segment_id == 2  # nothing pending


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(failed)
