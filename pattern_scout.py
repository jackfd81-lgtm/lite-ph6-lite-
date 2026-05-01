#!/usr/bin/env python3
"""
PH6-Lite PatternScout-lite v0.1

Reads:
    spike_events.jsonl
    soso_swarm_tokens.jsonl
    run_log.jsonl (pseudo packets + phase labels)

Emits:
    pattern_records.jsonl  (MRAM-S, advisory only)

Authority:
    NONE

PatternScout recognizes patterns.
PatternScout does not decide truth.
PatternScout does not alter PSEUDO.
PatternScout does not change PASS/DROP.
PatternScout writes only advisory records to MRAM-S.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA       = "ph6.pattern_scout.v0.1"
AUTHORITY    = "NONE"
STORE        = "MRAM-S"
REPEAT_MIN   = 3      # minimum occurrences to call a repeating pattern
REPEAT_SPAN  = 60     # max frame window for a repeating pattern
CASCADE_WIN  = 10     # frames within which A→B→C counts as a cascade

EXPECTED_PHASE_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "quiet":      {"dominant": None,             "max_spikes": 10, "max_vlt": 0},
    "camera":     {"dominant": "SPIKE_MOTION",   "min_spikes": 10},
    "music":      {"dominant": "SPIKE_SOUND",    "min_spikes": 10},
    "flashlight": {"dominant": "SPIKE_OVERLIGHT","min_spikes": 10},
    "combined":   {"dominant": "SPIKE_COMBINED", "min_spikes": 20},
}


def _record(pattern_type: str, frame: Optional[int], phase: Optional[str],
            **kwargs) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "schema":       SCHEMA,
        "pattern_type": pattern_type,
        "authority":    AUTHORITY,
        "store":        STORE,
    }
    if frame is not None:
        r["frame"] = frame
    if phase is not None:
        r["phase"] = phase
    r.update(kwargs)
    return r


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# ── pattern detectors ─────────────────────────────────────────────────────────

def detect_repeating_events(spikes: List[Dict]) -> List[Dict]:
    """Detect repeated spike events of the same type within a frame window."""
    records = []
    by_type: Dict[str, List[int]] = defaultdict(list)
    for s in spikes:
        fi = s.get("frame_index", 0)
        for st in s.get("spikes", []):
            by_type[st].append(fi)

    for event_type, frames in by_type.items():
        frames.sort()
        # Sliding window grouping
        group = [frames[0]]
        for f in frames[1:]:
            if f - group[0] <= REPEAT_SPAN:
                group.append(f)
            else:
                if len(group) >= REPEAT_MIN:
                    conf = round(min(1.0, len(group) / 10), 2)
                    records.append(_record(
                        "REPEATING_EVENT_PATTERN",
                        frame=group[0],
                        phase=None,
                        event_type=event_type,
                        count=len(group),
                        frame_span=group[-1] - group[0],
                        confidence=conf,
                    ))
                group = [f]
        if len(group) >= REPEAT_MIN:
            conf = round(min(1.0, len(group) / 10), 2)
            records.append(_record(
                "REPEATING_EVENT_PATTERN",
                frame=group[0],
                phase=None,
                event_type=event_type,
                count=len(group),
                frame_span=group[-1] - group[0],
                confidence=conf,
            ))
    return records


def detect_cascade_patterns(spikes: List[Dict]) -> List[Dict]:
    """Detect A→B→C chains within CASCADE_WIN frames."""
    records = []
    # Build timeline: (frame, spike_type)
    timeline: List[Tuple[int, str]] = []
    for s in spikes:
        fi = s.get("frame_index", 0)
        for st in s.get("spikes", []):
            timeline.append((fi, st))
    timeline.sort()

    # Count two-step and three-step chains
    chain_counts: Counter = Counter()
    for i, (fi, st) in enumerate(timeline):
        for j in range(i + 1, len(timeline)):
            fj, sj = timeline[j]
            if fj - fi > CASCADE_WIN:
                break
            if sj == st:
                continue
            chain_counts[(st, sj)] += 1
            for k in range(j + 1, len(timeline)):
                fk, sk = timeline[k]
                if fk - fi > CASCADE_WIN:
                    break
                if sk in (st, sj):
                    continue
                chain_counts[(st, sj, sk)] += 1
                break

    for chain, count in chain_counts.most_common(10):
        if count >= REPEAT_MIN and len(chain) >= 2:
            interp = ""
            if set(chain) >= {"SPIKE_UNDERLIGHT", "SPIKE_BLUR"}:
                interp = "possible sensor cascade (exposure hunting)"
            elif "SPIKE_MOTION" in chain and "SPIKE_BLUR" in chain:
                interp = "motion-blur coupling"
            records.append(_record(
                "EVENT_CASCADE_PATTERN",
                frame=None,
                phase=None,
                chain=list(chain),
                count=count,
                interpretation=interp or "recurring multi-sensor sequence",
            ))
    return records


def detect_phase_signatures(
    spikes: List[Dict],
    pseudo_pkts: List[Dict],
    token_pkts: List[Dict],
) -> List[Dict]:
    """Compare observed per-phase dominant event vs expected signature."""
    records = []

    # Group spikes by phase
    phase_spike_types: Dict[str, Counter] = defaultdict(Counter)
    phase_spike_count: Dict[str, int] = defaultdict(int)
    for s in spikes:
        ph = s.get("phase", "unphased")
        for st in s.get("spikes", []):
            phase_spike_types[ph][st] += 1
        phase_spike_count[ph] += 1

    # Per-phase PASS/DROP from pseudo
    phase_pass:  Dict[str, int] = defaultdict(int)
    phase_drop:  Dict[str, int] = defaultdict(int)
    for p in pseudo_pkts:
        ph = p.get("phase", "unphased")
        if p.get("verdict") == "PASS":
            phase_pass[ph] += 1
        else:
            phase_drop[ph] += 1

    # Per-phase VLT count
    phase_vlt: Dict[str, int] = defaultdict(int)
    for t in token_pkts:
        ph = t.get("phase", "unphased")
        if t.get("token_type") == "VLT":
            phase_vlt[ph] += 1

    all_phases = set(phase_spike_types) | set(phase_pass) | set(phase_drop)
    for ph in sorted(all_phases):
        spike_count  = phase_spike_count.get(ph, 0)
        dominant     = phase_spike_types[ph].most_common(1)
        dom_event    = dominant[0][0] if dominant else None
        drop_count   = phase_drop.get(ph, 0)
        total_frames = phase_pass.get(ph, 0) + drop_count
        drop_rate    = round(drop_count / total_frames, 3) if total_frames else 0.0
        vlt_count    = phase_vlt.get(ph, 0)

        expected     = EXPECTED_PHASE_SIGNATURES.get(ph)
        if expected is None:
            status = "NO_EXPECTATION"
        elif ph == "quiet":
            ok = spike_count <= expected["max_spikes"] and vlt_count <= expected["max_vlt"]
            status = "MATCHED_EXPECTATION" if ok else "EXCEEDED_QUIET_LIMIT"
        else:
            exp_dom   = expected.get("dominant")
            min_sp    = expected.get("min_spikes", 0)
            dom_ok    = (dom_event == exp_dom)
            count_ok  = (spike_count >= min_sp)
            if dom_ok and count_ok:
                status = "MATCHED_EXPECTATION"
            elif dom_ok and not count_ok:
                status = "WEAK_SIGNAL"
            elif not dom_ok and count_ok:
                status = "WRONG_DOMINANT_EVENT"
            else:
                status = "MISSED_EXPECTATION"

        r = _record(
            "PHASE_SIGNATURE",
            frame=None,
            phase=ph,
            dominant_event=dom_event,
            spike_count=spike_count,
            drop_rate=drop_rate,
            vlt_count=vlt_count,
            signature_status=status,
        )
        if expected and ph != "quiet":
            exp_dom = expected.get("dominant")
            if dom_event and dom_event != exp_dom:
                r["misclassification"] = _record(
                    "POSSIBLE_MISCLASSIFICATION_PATTERN",
                    frame=None,
                    phase=ph,
                    expected=exp_dom,
                    observed=dom_event,
                    recommendation=f"inspect threshold for {exp_dom}",
                )
        records.append(r)
    return records


def detect_token_continuity(token_pkts: List[Dict]) -> List[Dict]:
    """Detect RT→VDT→VLT continuity chains by linked_tokens graph."""
    records = []
    by_event: Dict[str, List[Dict]] = defaultdict(list)
    for t in token_pkts:
        ev = t.get("event_type", "UNKNOWN")
        by_event[ev].append(t)

    for event_type, tokens in by_event.items():
        has_rt  = any(t.get("token_type") == "RT"  for t in tokens)
        has_vdt = any(t.get("token_type") == "VDT" for t in tokens)
        has_vlt = any(t.get("token_type") == "VLT" for t in tokens)

        if not (has_rt and has_vdt):
            continue

        chain = ["RT"]
        if has_vdt:
            chain.append("VDT")
        if has_vlt:
            chain.append("VLT")

        frames = sorted(t.get("source_frame", 0) for t in tokens)
        length = frames[-1] - frames[0] if len(frames) > 1 else 0
        strength = round(min(1.0, len(tokens) / 50), 2)

        records.append(_record(
            "TOKEN_CONTINUITY_PATTERN",
            frame=frames[0] if frames else None,
            phase=tokens[0].get("phase"),
            chain=chain,
            event_type=event_type,
            length_frames=length,
            token_count=len(tokens),
            continuity_strength=strength,
        ))
    return records


# ── summary builder ───────────────────────────────────────────────────────────

def build_summary(records: List[Dict]) -> Dict[str, Any]:
    phase_sigs   = [r for r in records if r["pattern_type"] == "PHASE_SIGNATURE"]
    misclass     = [r["misclassification"] for r in phase_sigs if "misclassification" in r]
    cascades     = [r for r in records if r["pattern_type"] == "EVENT_CASCADE_PATTERN"]
    repeats      = [r for r in records if r["pattern_type"] == "REPEATING_EVENT_PATTERN"]
    continuities = [r for r in records if r["pattern_type"] == "TOKEN_CONTINUITY_PATTERN"]

    longest_chain = max(continuities, key=lambda r: r["length_frames"], default=None)

    phase_status = {r["phase"]: r["signature_status"] for r in phase_sigs if r.get("phase")}

    return {
        "schema":         SCHEMA,
        "authority":      AUTHORITY,
        "store":          STORE,
        "phase_signatures": phase_status,
        "misclassifications": [
            {"phase": m["phase"], "expected": m["expected"], "observed": m["observed"]}
            for m in misclass
        ],
        "cascade_patterns":  len(cascades),
        "repeating_patterns": len(repeats),
        "longest_token_chain": {
            "event_type":    longest_chain["event_type"],
            "length_frames": longest_chain["length_frames"],
            "chain":         longest_chain["chain"],
        } if longest_chain else None,
        "authority_effect": "NONE",
    }


# ── public API ────────────────────────────────────────────────────────────────

def run_pattern_scout(run_dir: Path) -> Tuple[List[Dict], Dict]:
    """Read run artifacts, detect patterns, write pattern_records.jsonl."""
    spike_pkts  = _load_jsonl(run_dir / "hot"   / "spike_events.jsonl")
    token_pkts  = _load_jsonl(run_dir / "mram_s" / "soso_swarm_tokens.jsonl")
    all_log     = _load_jsonl(run_dir / "hot"   / "run_log.jsonl")
    pseudo_pkts = [p for p in all_log if p.get("packet_type") == "pseudo"]
    vt_pkts     = [p for p in all_log if p.get("packet_type") == "virtual_token"]

    records: List[Dict] = []
    records.extend(detect_repeating_events(spike_pkts))
    records.extend(detect_cascade_patterns(spike_pkts))
    records.extend(detect_phase_signatures(spike_pkts, pseudo_pkts, vt_pkts))
    records.extend(detect_token_continuity(token_pkts))

    out_path = run_dir / "post" / "pattern_records.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")

    summary = build_summary(records)
    summary_path = run_dir / "post" / "pattern_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return records, summary


def print_scout_summary(summary: Dict) -> None:
    print("\n=== PatternScout-lite Summary ===")
    print("Phase signatures:")
    for phase, status in summary.get("phase_signatures", {}).items():
        print(f"  {phase:<12} {status}")
    misclass = summary.get("misclassifications", [])
    if misclass:
        print("Possible misclassifications:")
        for m in misclass:
            print(f"  {m['phase']}: expected {m['expected']}, observed {m['observed']}")
    lc = summary.get("longest_token_chain")
    if lc:
        print(f"Longest token chain: {lc['event_type']}  {lc['length_frames']} frames  {lc['chain']}")
    print(f"Cascade patterns:   {summary.get('cascade_patterns', 0)}")
    print(f"Repeating patterns: {summary.get('repeating_patterns', 0)}")
    print(f"Authority effect:   {summary.get('authority_effect', 'NONE')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 pattern_scout.py <run_dir>")
        return 2
    run_dir = Path(sys.argv[1])
    records, summary = run_pattern_scout(run_dir)
    print(f"PatternScout: {len(records)} records → {run_dir / 'post' / 'pattern_records.jsonl'}")
    print_scout_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
