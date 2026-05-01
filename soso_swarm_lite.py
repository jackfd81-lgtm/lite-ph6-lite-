#!/usr/bin/env python3
"""
PH6-Lite SoSo Swarm Lite v0.1

Lane: 2 only
Authority: NONE
Store: MRAM-S only

This module observes PSEUDO/SoSo events and emits advisory RT/VDT/VLT tokens.
It must never emit PASS/DROP, modify thresholds, write CRAM tiers, or affect replay.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import uuid


ALLOWED_TOKEN_TYPES = {"RT", "VDT", "VLT"}
SCHEMA = "ph6.soso_swarm_lite.v0.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SoSoSwarmToken:
    schema: str
    packet_type: str
    token_type: str
    token_id: str
    source_frame: int
    event_type: str
    linked_reasons: List[str]
    linked_tokens: List[str]
    strength: int
    created_at: str
    source_object_id: Optional[str]
    authority: str
    store: str
    lane: str
    advisory_only: bool
    may_influence_verdict: bool

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("invalid schema")

        if self.packet_type != "SOSO_SWARM_TOKEN":
            raise ValueError("invalid packet_type")

        if self.token_type not in ALLOWED_TOKEN_TYPES:
            raise ValueError("invalid token_type")

        if self.authority != "NONE":
            raise ValueError("authority violation")

        if self.store != "MRAM-S":
            raise ValueError("store violation")

        if self.lane != "LANE_2":
            raise ValueError("lane violation")

        if self.advisory_only is not True:
            raise ValueError("advisory_only must be true")

        if self.may_influence_verdict is not False:
            raise ValueError("may_influence_verdict must be false")

        # Check for forbidden authority vocabulary in the string-valued fields
        # that could carry Lane-1 vocabulary. Exclude field names themselves to
        # avoid false positives (e.g. "verdict" inside "may_influence_verdict").
        forbidden_values = {"PASS", "DROP", "threshold_override", "gate_override"}
        string_values = [self.event_type] + self.linked_reasons + self.linked_tokens
        if self.source_object_id:
            string_values.append(self.source_object_id)
        for val in forbidden_values:
            for sv in string_values:
                if val == sv:
                    raise ValueError(f"forbidden authority vocabulary in token field: {val}")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class SoSoSwarmLite:
    """
    Tiny local advisory swarm.

    - RT:  created on first meaningful event observation
    - VDT: created when similar events recur inside link_window_frames
    - VLT: created only when ALL four gates pass:
             1. strength >= promote_min_strength
             2. pattern spans >= min_vlt_frame_span frames since first RT
             3. at least min_vdt_count VDTs accumulated for this event type
             4. vlt_cooldown_frames have elapsed since last VLT for this event type

    VDT should be common. VLT should be selective.
    VLT means "this pattern persisted enough to matter."
    """

    def __init__(
        self,
        link_window_frames: int = 30,
        promote_min_strength: int = 5,
        close_after_frames: int = 90,
        min_vlt_frame_span: int = 15,
        min_vdt_count: int = 5,
        vlt_cooldown_frames: int = 30,
    ) -> None:
        self.link_window_frames    = int(link_window_frames)
        self.promote_min_strength  = int(promote_min_strength)
        self.close_after_frames    = int(close_after_frames)
        self.min_vlt_frame_span    = int(min_vlt_frame_span)
        self.min_vdt_count         = int(min_vdt_count)
        self.vlt_cooldown_frames   = int(vlt_cooldown_frames)

        self.active_by_event: Dict[str, List[SoSoSwarmToken]] = {}
        self.last_seen_frame_by_event: Dict[str, int] = {}
        self.first_seen_frame_by_event: Dict[str, int] = {}
        self.last_vlt_frame_by_event: Dict[str, int] = {}

    def _make_token(
        self,
        token_type: str,
        source_frame: int,
        event_type: str,
        linked_reasons: Optional[List[str]] = None,
        linked_tokens: Optional[List[str]] = None,
        strength: int = 1,
        source_object_id: Optional[str] = None,
    ) -> SoSoSwarmToken:
        token = SoSoSwarmToken(
            schema=SCHEMA,
            packet_type="SOSO_SWARM_TOKEN",
            token_type=token_type,
            token_id=f"{token_type}_{uuid.uuid4().hex[:12]}",
            source_frame=int(source_frame),
            event_type=str(event_type),
            linked_reasons=list(linked_reasons or []),
            linked_tokens=list(linked_tokens or []),
            strength=int(strength),
            created_at=utc_now(),
            source_object_id=source_object_id,
            authority="NONE",
            store="MRAM-S",
            lane="LANE_2",
            advisory_only=True,
            may_influence_verdict=False,
        )
        token.validate()
        return token

    def observe_event(
        self,
        frame_id: int,
        event_type: str,
        reasons: Optional[List[str]] = None,
        source_object_id: Optional[str] = None,
    ) -> List[SoSoSwarmToken]:
        """
        Observe a Lane-1/PSEUDO result from the advisory side.

        This function emits advisory tokens only.
        It does not return verdicts.
        """
        frame_id = int(frame_id)
        reasons = list(reasons or [])

        emitted: List[SoSoSwarmToken] = []
        existing = self.active_by_event.get(event_type, [])
        last_seen = self.last_seen_frame_by_event.get(event_type)

        # First reference for this event type — emit RT and record origin frame.
        if last_seen is None:
            rt = self._make_token(
                token_type="RT",
                source_frame=frame_id,
                event_type=event_type,
                linked_reasons=reasons,
                strength=1,
                source_object_id=source_object_id,
            )
            emitted.append(rt)
            self.active_by_event.setdefault(event_type, []).append(rt)
            self.last_seen_frame_by_event[event_type]  = frame_id
            self.first_seen_frame_by_event[event_type] = frame_id
            return emitted

        frame_gap = frame_id - last_seen

        # Pattern is recent — emit VDT.
        if 0 <= frame_gap <= self.link_window_frames:
            linked_ids = [t.token_id for t in existing[-3:]]
            strength = min(len(existing) + 1, 999)

            vdt = self._make_token(
                token_type="VDT",
                source_frame=frame_id,
                event_type=event_type,
                linked_reasons=reasons,
                linked_tokens=linked_ids,
                strength=strength,
                source_object_id=source_object_id,
            )
            emitted.append(vdt)
            self.active_by_event[event_type].append(vdt)

            # VLT gate — all four conditions must pass.
            vdt_count    = sum(1 for t in self.active_by_event[event_type] if t.token_type == "VDT")
            frame_span   = frame_id - self.first_seen_frame_by_event.get(event_type, frame_id)
            last_vlt     = self.last_vlt_frame_by_event.get(event_type, -self.vlt_cooldown_frames)
            cooldown_ok  = (frame_id - last_vlt) >= self.vlt_cooldown_frames

            if (
                strength    >= self.promote_min_strength   # gate 1: accumulated weight
                and frame_span  >= self.min_vlt_frame_span    # gate 2: pattern age
                and vdt_count   >= self.min_vdt_count          # gate 3: observation count
                and cooldown_ok                                 # gate 4: emission cooldown
            ):
                recent_vdts = [
                    t.token_id for t in self.active_by_event[event_type]
                    if t.token_type == "VDT"
                ][-3:]
                vlt = self._make_token(
                    token_type="VLT",
                    source_frame=frame_id,
                    event_type=event_type,
                    linked_reasons=reasons,
                    linked_tokens=recent_vdts,
                    strength=strength,
                    source_object_id=source_object_id,
                )
                emitted.append(vlt)
                self.active_by_event[event_type].append(vlt)
                self.last_vlt_frame_by_event[event_type] = frame_id

        # Pattern is stale — restart with a new RT.
        else:
            rt = self._make_token(
                token_type="RT",
                source_frame=frame_id,
                event_type=event_type,
                linked_reasons=reasons,
                strength=1,
                source_object_id=source_object_id,
            )
            emitted.append(rt)
            self.active_by_event[event_type]            = [rt]
            self.first_seen_frame_by_event[event_type]  = frame_id

        self.last_seen_frame_by_event[event_type] = frame_id
        return emitted

    def write_mram_s(self, path: str, tokens: List[SoSoSwarmToken]) -> int:
        """
        Append advisory tokens to MRAM-S jsonl.

        Caller must pass an MRAM-S path. This function refuses obvious CRAM paths.
        """
        forbidden_path_parts = ("cram-0", "cram-a", "cram-r", "CRAM-0", "CRAM-A", "CRAM-R")
        if any(part in path for part in forbidden_path_parts):
            raise ValueError("refusing to write SoSo Swarm Lite tokens to a CRAM path")

        count = 0
        with open(path, "a", encoding="utf-8") as f:
            for token in tokens:
                f.write(token.to_json() + "\n")
                count += 1
        return count
