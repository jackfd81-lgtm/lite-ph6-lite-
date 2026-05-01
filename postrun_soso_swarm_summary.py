#!/usr/bin/env python3
"""
PH6-Lite PostRun SoSo Swarm Summary

Reads:
    soso_swarm_tokens.jsonl

Emits:
    soso_swarm_summary.json

Authority:
    NONE

This analyzer is read-only with respect to SoSo Swarm tokens.
It must never write CRAM, alter PSEUDO verdicts, alter thresholds, or affect replay.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_TOKEN_TYPES = {"RT", "VDT", "VLT"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            records.append(obj)
    return records


def validate_token_boundary(token: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if token.get("authority") != "NONE":
        errors.append("authority_not_NONE")
    if token.get("store") != "MRAM-S":
        errors.append("store_not_MRAM_S")
    if token.get("lane") != "LANE_2":
        errors.append("lane_not_LANE_2")
    if token.get("advisory_only") is not True:
        errors.append("advisory_only_not_true")
    if token.get("may_influence_verdict") is not False:
        errors.append("may_influence_verdict_not_false")
    token_type = token.get("token_type")
    if token_type not in ALLOWED_TOKEN_TYPES:
        errors.append(f"invalid_token_type:{token_type}")
    if "verdict" in token:
        errors.append("forbidden_verdict_field_present")
    return errors


def strongest_event_type(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Strongest event by summed strength; tie-break by count then lexical."""
    strength_by_event: Dict[str, int] = defaultdict(int)
    count_by_event: Dict[str, int] = defaultdict(int)
    for token in tokens:
        event_type = str(token.get("event_type", "UNKNOWN"))
        strength = int(token.get("strength", 0))
        strength_by_event[event_type] += strength
        count_by_event[event_type] += 1
    if not strength_by_event:
        return {"event_type": None, "strength_sum": 0, "token_count": 0}
    ranked = sorted(
        strength_by_event.keys(),
        key=lambda e: (-strength_by_event[e], -count_by_event[e], e),
    )
    winner = ranked[0]
    return {
        "event_type": winner,
        "strength_sum": strength_by_event[winner],
        "token_count": count_by_event[winner],
    }


def longest_advisory_continuity_chain(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Longest linked_tokens chain. Advisory topology only — no authority effect."""
    by_id: Dict[str, Dict[str, Any]] = {
        str(t.get("token_id")): t for t in tokens if t.get("token_id")
    }
    memo: Dict[str, int] = {}

    def chain_len(token_id: str, visiting: set) -> int:
        if token_id in memo:
            return memo[token_id]
        if token_id in visiting:
            return 0
        visiting.add(token_id)
        token = by_id.get(token_id)
        if not token:
            return 0
        linked = token.get("linked_tokens") or []
        best = max((chain_len(str(p), visiting.copy()) for p in linked), default=0)
        memo[token_id] = 1 + best
        return memo[token_id]

    if not by_id:
        return {"length": 0, "terminal_token_id": None,
                "terminal_token_type": None, "event_type": None}

    ranked = sorted(
        by_id.items(),
        key=lambda kv: (-chain_len(kv[0], set()), kv[0]),
    )
    terminal_id, terminal = ranked[0]
    return {
        "length": memo.get(terminal_id, 0),
        "terminal_token_id": terminal_id,
        "terminal_token_type": terminal.get("token_type"),
        "event_type": terminal.get("event_type"),
    }


def summarize(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_counts = Counter(str(t.get("token_type", "UNKNOWN")) for t in tokens)
    boundary_errors: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        errors = validate_token_boundary(token)
        if errors:
            boundary_errors.append(
                {"index": idx, "token_id": token.get("token_id"), "errors": errors}
            )
    authority_effect_confirmed_absent = len(boundary_errors) == 0
    return {
        "schema": "ph6.postrun.soso_swarm_summary.v0.1",
        "authority": "NONE",
        "store": "POSTRUN_REPORT",
        "source": "MRAM-S",
        "token_counts": {
            "RT":  token_counts.get("RT", 0),
            "VDT": token_counts.get("VDT", 0),
            "VLT": token_counts.get("VLT", 0),
        },
        "total_tokens": len(tokens),
        "strongest_event_type": strongest_event_type(tokens),
        "longest_advisory_continuity_chain": longest_advisory_continuity_chain(tokens),
        "authority_effect_check": {
            "passed": authority_effect_confirmed_absent,
            "confirmation": (
                "Token subsystem had no authority effect"
                if authority_effect_confirmed_absent
                else "Boundary errors detected; token subsystem must remain quarantined"
            ),
            "checked_fields": [
                "authority == NONE",
                "store == MRAM-S",
                "lane == LANE_2",
                "advisory_only == true",
                "may_influence_verdict == false",
                "no verdict field present",
                "token_type in RT/VDT/VLT",
            ],
            "boundary_errors": boundary_errors,
        },
        "may_influence_pseudo": False,
        "may_influence_pass_drop": False,
        "may_write_cram": False,
        "replay_dependency": False,
    }


def run_soso_swarm_postrun(run_dir: Path) -> dict:
    token_path  = run_dir / "mram_s" / "soso_swarm_tokens.jsonl"
    output_path = run_dir / "post" / "soso_swarm_summary.json"
    tokens  = load_jsonl(token_path)
    summary = summarize(tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, sort_keys=True, ensure_ascii=False, indent=2)
    return summary


def print_swarm_summary(summary: dict) -> None:
    print("\n=== SoSo Swarm Lite Summary ===")
    print(f"RT count:  {summary['token_counts']['RT']}")
    print(f"VDT count: {summary['token_counts']['VDT']}")
    print(f"VLT count: {summary['token_counts']['VLT']}")
    strongest = summary["strongest_event_type"]
    print(f"Strongest event_type: {strongest['event_type']}")
    print(f"Strongest event strength: {strongest['strength_sum']}")
    chain = summary["longest_advisory_continuity_chain"]
    print(f"Longest advisory continuity chain: {chain['length']}")
    print(f"Terminal token type: {chain['terminal_token_type']}")
    auth = summary["authority_effect_check"]
    print(f"Authority effect check: {'PASS' if auth['passed'] else 'FAIL'}")
    print(auth["confirmation"])


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 postrun_soso_swarm_summary.py <soso_swarm_tokens.jsonl> [output.json]")
        return 2
    input_path = Path(sys.argv[1])
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) >= 3
        else input_path.parent.parent / "post" / "soso_swarm_summary.json"
    )
    tokens  = load_jsonl(input_path)
    summary = summarize(tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, sort_keys=True, ensure_ascii=False, indent=2)
    print(f"SoSo Swarm PostRun Summary written: {output_path}")
    print_swarm_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
