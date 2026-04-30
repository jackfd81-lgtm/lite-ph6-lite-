from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VirtualToken:
    token_id: str
    token_type: str          # RT, VDT, VLT
    event_kind: str          # motion, blur, underlight, overlight, sound, degraded, combined
    root_frame: int
    last_frame: int
    strength: int = 1
    state: str = "active"
    linked_frames: List[int] = field(default_factory=list)

    def to_packet(self, frame_index: int, ts: str) -> dict:
        return {
            "packet_type": "virtual_token",
            "ts_utc": ts,
            "authority": "NONE",
            "store": "MRAM-S",
            "token_type": self.token_type,
            "token_id": self.token_id,
            "frame_index": frame_index,
            "root_frame": self.root_frame,
            "last_frame": self.last_frame,
            "event_kind": self.event_kind,
            "state": self.state,
            "strength": self.strength,
            "linked_frames": list(self.linked_frames[-20:]),
            "advisory": self._advisory(),
        }

    def _advisory(self) -> str:
        if self.token_type == "RT":
            return "real evidence anchor"
        if self.token_type == "VDT":
            return f"{self.event_kind} continuity candidate"
        if self.token_type == "VLT":
            return f"{self.event_kind} continuity stabilized"
        return "unknown advisory token"


class VirtualTokenTracker:
    def __init__(
        self,
        link_window_frames: int = 30,
        promote_min_frames: int = 12,
        promote_min_strength: int = 3,
        close_after_frames: int = 60,
    ):
        self.link_window_frames = link_window_frames
        self.promote_min_frames = promote_min_frames
        self.promote_min_strength = promote_min_strength
        self.close_after_frames = close_after_frames
        self.active: Dict[str, VirtualToken] = {}
        self.counter = 0

    def observe_event(self, frame_index: int, event_kind: str, ts: str) -> List[dict]:
        packets = []

        token = self._find_active(event_kind, frame_index)

        if token is None:
            self.counter += 1

            rt = VirtualToken(
                token_id=f"RT-{event_kind}-{self.counter:06d}",
                token_type="RT",
                event_kind=event_kind,
                root_frame=frame_index,
                last_frame=frame_index,
                linked_frames=[frame_index],
            )
            packets.append(rt.to_packet(frame_index, ts))

            token = VirtualToken(
                token_id=f"VDT-{event_kind}-{self.counter:06d}",
                token_type="VDT",
                event_kind=event_kind,
                root_frame=frame_index,
                last_frame=frame_index,
                linked_frames=[frame_index],
            )
            self.active[token.token_id] = token
            packets.append(token.to_packet(frame_index, ts))
            return packets

        token.last_frame = frame_index
        token.strength += 1
        token.linked_frames.append(frame_index)

        if token.token_type == "VDT":
            span = token.last_frame - token.root_frame
            if span >= self.promote_min_frames or token.strength >= self.promote_min_strength:
                token.token_type = "VLT"
        packets.append(token.to_packet(frame_index, ts))
        return packets

    def close_expired(self, frame_index: int, ts: str) -> List[dict]:
        packets = []
        for token_id, token in list(self.active.items()):
            if frame_index - token.last_frame >= self.close_after_frames:
                token.state = "closed"
                packets.append(token.to_packet(frame_index, ts))
                del self.active[token_id]
        return packets

    def _find_active(self, event_kind: str, frame_index: int) -> Optional[VirtualToken]:
        candidates = [
            t for t in self.active.values()
            if t.event_kind == event_kind
            and t.state == "active"
            and frame_index - t.last_frame <= self.link_window_frames
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda t: t.last_frame, reverse=True)[0]
