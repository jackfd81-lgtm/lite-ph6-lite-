#!/usr/bin/env python3
"""
check_log.py — Structural + schema validator for frame_filter JSONL logs (v2.2).

Single-session logs: validated as before.
Multi-session (appended) logs: split by session_start/session_end boundaries and
validated independently. packet_seq resets per session — no false gaps across sessions.

Validates per session:
  - JSONL parse (each line is valid JSON)
  - Each packet against its JSON Schema (schemas/ directory)
  - Packet count formula: 2 + (3×N) + 1
  - Sequence: session_start → config → [pseudo→observation→soso]×N → session_end
  - packet_seq is strictly monotonic, 1-based, no gaps (within session)
  - session_id consistency across all session-scoped packets
  - frame_index matches across each pseudo/observation/soso triplet
"""

import json
import sys
import os
import argparse
import jsonschema
from jsonschema import Draft202012Validator

SESSION_SCOPED = {"session_start", "config", "session_end"}


def load_schemas(schemas_dir):
    manifest_path = os.path.join(schemas_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Schema manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    validators = {}
    for ptype, filename in manifest["schemas"].items():
        schema_path = os.path.join(schemas_dir, filename)
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        validators[ptype] = Draft202012Validator(schema)

    return manifest, validators


def split_sessions(packets):
    """Split flat packet list into per-session groups by session_start/session_end."""
    sessions = []
    orphans = []
    current = None

    for line_num, p in packets:
        ptype = p.get("packet_type")
        if ptype == "session_start":
            if current is not None:
                sessions.append(("unclosed", current))
            current = [(line_num, p)]
        elif current is None:
            orphans.append((line_num, p))
        else:
            current.append((line_num, p))
            if ptype == "session_end":
                sessions.append(("closed", current))
                current = None

    if current is not None:
        sessions.append(("unclosed", current))

    return sessions, orphans


def validate_session(idx, session_packets, schema_pack, validators):
    """Validate one session's packets. Returns (errors, frames_processed|None)."""
    errors = []
    prefix = f"[session {idx}] " if idx is not None else ""

    def err(line_num, msg):
        errors.append(f"  {prefix}line {line_num}: {msg}")

    n_packets = len(session_packets)
    _, first = session_packets[0]
    _, last = session_packets[-1]

    if first.get("packet_type") != "session_start":
        err(session_packets[0][0], f"first packet must be session_start, got {first.get('packet_type')!r}")
    if last.get("packet_type") != "session_end":
        err(session_packets[-1][0], f"last packet must be session_end, got {last.get('packet_type')!r}")

    session_id = first.get("session_id")

    frames_processed = last.get("frames_processed")
    if not isinstance(frames_processed, int):
        err(session_packets[-1][0], "session_end frames_processed must be int")
    else:
        expected_total = 2 + 3 * frames_processed + 1
        if n_packets != expected_total:
            errors.append(
                f"  {prefix}packet count mismatch: got {n_packets}, "
                f"expected {expected_total} (2 + 3×{frames_processed} + 1)"
            )

    expected_seq = 1
    triplet = []

    for line_num, p in session_packets:
        ptype = p.get("packet_type")

        validator = validators.get(ptype)
        if validator is None:
            err(line_num, f"unknown packet_type: {ptype!r} (no schema)")
            continue

        schema_errors = sorted(validator.iter_errors(p), key=lambda e: e.path)
        for se in schema_errors:
            path = ".".join(str(x) for x in se.absolute_path) or "(root)"
            err(line_num, f"[schema] {path}: {se.message}")

        seq = p.get("packet_seq")
        if isinstance(seq, int):
            if seq != expected_seq:
                err(line_num, f"packet_seq gap: expected {expected_seq}, got {seq}")
            expected_seq = seq + 1
        else:
            expected_seq += 1

        if ptype in SESSION_SCOPED:
            pid = p.get("session_id")
            if pid != session_id:
                err(line_num, f"session_id mismatch: {pid!r} != {session_id!r}")

        if ptype == "pseudo":
            if triplet:
                err(line_num, f"incomplete triplet before frame {p.get('frame_index')} (got {[t[0] for t in triplet]})")
            triplet = [("pseudo", line_num, p)]
        elif ptype == "observation":
            if not triplet or triplet[-1][0] != "pseudo":
                err(line_num, "observation not immediately preceded by pseudo")
            else:
                triplet.append(("observation", line_num, p))
        elif ptype == "soso":
            if len(triplet) != 2 or triplet[-1][0] != "observation":
                err(line_num, "soso not immediately preceded by observation")
            else:
                triplet.append(("soso", line_num, p))
                idxs = [t[2].get("frame_index") for t in triplet]
                if len(set(idxs)) != 1:
                    err(line_num, f"frame_index mismatch in triplet: {idxs}")
                triplet = []

    if triplet:
        last_line = triplet[-1][1]
        err(last_line, f"incomplete triplet at end of session (got {[t[0] for t in triplet]})")

    return errors, frames_processed if isinstance(frames_processed, int) else None


def validate(log_path, schemas_dir, verbose=False):
    warnings = []

    def warn(msg):
        warnings.append(f"  {msg}")

    try:
        manifest, validators = load_schemas(schemas_dir)
    except FileNotFoundError as e:
        print(f"FAIL — {e}")
        return False

    schema_pack = manifest.get("schema_pack", "unknown")

    # Parse JSONL
    parse_errors = []
    packets = []
    with open(log_path, encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw:
                warn(f"line {i}: empty line (skipped)")
                continue
            try:
                packets.append((i, json.loads(raw)))
            except json.JSONDecodeError as e:
                parse_errors.append(f"  line {i}: JSON parse error: {e}")

    if parse_errors:
        print(f"FAIL [{schema_pack}] — JSON parse errors, cannot continue.")
        for e in parse_errors:
            print(e)
        return False

    if len(packets) < 3:
        print(f"FAIL [{schema_pack}] — too few packets: {len(packets)}")
        return False

    sessions, orphans = split_sessions(packets)

    for line_num, p in orphans:
        warn(f"line {line_num}: orphan packet before first session_start: {p.get('packet_type')!r}")

    multi = len(sessions) > 1
    all_errors = []
    total_packets = 0
    total_frames = 0
    session_summaries = []

    for i, (status, session_packets) in enumerate(sessions, 1):
        idx = i if multi else None
        s_errors, s_frames = validate_session(idx, session_packets, schema_pack, validators)
        all_errors.extend(s_errors)
        n = len(session_packets)
        total_packets += n
        if s_frames is not None:
            total_frames += s_frames
        sid = session_packets[0][1].get("session_id", "?")[:12]
        if s_errors:
            session_summaries.append(
                f"  session {i} (id={sid}): FAIL — {len(s_errors)} error(s), {n} packets, {s_frames} frames"
            )
        else:
            session_summaries.append(
                f"  session {i} (id={sid}): OK — {n} packets, {s_frames} frames"
            )

    if verbose and warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(w)

    if all_errors:
        if multi:
            for s in session_summaries:
                print(s)
        print(f"FAIL [{schema_pack}] — {len(all_errors)} error(s):")
        for e in all_errors:
            print(e)
        return False

    if multi:
        for s in session_summaries:
            print(s)
        print(
            f"OK [{schema_pack}] — {len(sessions)} sessions, "
            f"{total_packets} packets, {total_frames} frames total, all checks passed."
        )
    else:
        print(f"OK [{schema_pack}] — {total_packets} packets, {total_frames} frames, all checks passed.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Validate frame_filter JSONL log (v2.2).")
    ap.add_argument("log", nargs="?", default="logs/packets.jsonl")
    ap.add_argument(
        "--schemas", default=os.path.join(os.path.dirname(__file__), "schemas"),
        help="path to schemas directory (default: schemas/ next to this script)"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="show warnings")
    args = ap.parse_args()

    ok = validate(args.log, args.schemas, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
