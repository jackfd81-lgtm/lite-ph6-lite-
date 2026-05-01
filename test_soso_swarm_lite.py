#!/usr/bin/env python3
"""
PH6-Lite SoSo Swarm Lite tests.

These tests prove:
- RT emits on first event
- VDT emits on repeated event
- VLT emits after stability threshold
- all tokens are authority NONE
- all tokens are MRAM-S only
- no PASS/DROP vocabulary appears
- CRAM paths are rejected
"""

import json
import os
import tempfile

from soso_swarm_lite import SoSoSwarmLite


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status:4} | {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(name)


def main():
    tracker = SoSoSwarmLite(
        link_window_frames=30,
        promote_min_strength=5,
        close_after_frames=90,
        min_vlt_frame_span=15,
        min_vdt_count=5,
        vlt_cooldown_frames=30,
    )

    # First event creates RT.
    pkts0 = tracker.observe_event(
        frame_id=100, event_type="motion",
        reasons=["motion_veto"], source_object_id="obj_100",
    )
    check("first_event_creates_RT", any(p.token_type == "RT" for p in pkts0))

    # Repeated event (within window) creates VDT.
    pkts1 = tracker.observe_event(
        frame_id=110, event_type="motion",
        reasons=["motion_veto"], source_object_id="obj_110",
    )
    check("repeated_event_creates_VDT", any(p.token_type == "VDT" for p in pkts1))

    # Drive 3 more events — gates 1-3 not yet all satisfied:
    #   frame 115: strength=3, vdt_count=2, span=15 — gate 1 fails (strength<5)
    #   frame 118: strength=4, vdt_count=3, span=18 — gate 1 fails
    #   frame 121: strength=5, vdt_count=4, span=21 — gate 3 fails (vdt_count<5)
    pkts_mid = []
    for fi in [115, 118, 121]:
        pkts_mid.extend(tracker.observe_event(
            frame_id=fi, event_type="motion",
            reasons=["motion_veto"],
        ))
    check("pre_vlt_no_premature_promotion",
          not any(p.token_type == "VLT" for p in pkts_mid))

    # 5th VDT (frame 124): strength=6, vdt_count=5, span=24 — all gates pass → VLT.
    pkts2 = tracker.observe_event(
        frame_id=124, event_type="motion",
        reasons=["motion_veto"], source_object_id="obj_124",
    )
    check("stable_event_creates_VLT", any(p.token_type == "VLT" for p in pkts2))

    # Second VLT within cooldown window (frame 130, gap=6 < 30) must be suppressed.
    pkts3 = tracker.observe_event(
        frame_id=130, event_type="motion",
        reasons=["motion_veto"],
    )
    check("vlt_cooldown_suppresses_early_repeat",
          not any(p.token_type == "VLT" for p in pkts3))

    all_pkts = pkts0 + pkts1 + pkts_mid + pkts2 + pkts3

    for p in all_pkts:
        d = json.loads(p.to_json())
        check("authority_NONE", d["authority"] == "NONE", p.token_id)
        check("store_MRAM_S", d["store"] == "MRAM-S", p.token_id)
        check("lane_LANE_2", d["lane"] == "LANE_2", p.token_id)
        check("advisory_only_true", d["advisory_only"] is True, p.token_id)
        check("may_influence_verdict_false", d["may_influence_verdict"] is False, p.token_id)

        txt = json.dumps(d, sort_keys=True)
        check("no_PASS_DROP_vocab", "PASS" not in txt and "DROP" not in txt, p.token_id)

    # Write to MRAM-S path.
    with tempfile.TemporaryDirectory() as td:
        mram_s_path = os.path.join(td, "mram_s_tokens.jsonl")
        count = tracker.write_mram_s(mram_s_path, all_pkts)
        check("writes_to_mram_s_jsonl", count == len(all_pkts))

        # Refuse CRAM path.
        bad_path = os.path.join(td, "cram-a", "tokens.jsonl")
        os.makedirs(os.path.dirname(bad_path), exist_ok=True)
        try:
            tracker.write_mram_s(bad_path, all_pkts)
            refused = False
        except ValueError:
            refused = True

        check("refuses_cram_path", refused)

    print("\nSoSo Swarm Lite: PASS")


if __name__ == "__main__":
    main()
