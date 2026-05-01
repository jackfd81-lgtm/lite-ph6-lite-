import os
import json
import time
import hashlib
from pathlib import Path

OUT_DIR = Path.home() / "frame_filter" / "logs" / "cram_speed_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log_path = OUT_DIR / "hot_speed.jsonl"

N = 3000
payload_size = 1024  # 1 KB simulated packet payload

prev_hash = "GENESIS"
start = time.perf_counter()

with open(log_path, "w", buffering=1024 * 1024) as f:
    for i in range(1, N + 1):
        payload = {
            "packet_type": "frame_packet",
            "frame": i,
            "verdict": "PASS",
            "confidence": "0.95",
            "reason": "speed_test",
            "payload": "X" * payload_size,
            "prev_hash": prev_hash,
        }

        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        packet_hash = hashlib.blake2b(
            canonical.encode("utf-8"),
            digest_size=32
        ).hexdigest()

        payload["packet_hash"] = packet_hash
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

        prev_hash = packet_hash

    f.flush()
    os.fsync(f.fileno())

end = time.perf_counter()
elapsed = end - start
size_bytes = log_path.stat().st_size

print("CRAM SPEED TEST COMPLETE")
print("------------------------")
print(f"packets:        {N}")
print(f"elapsed_sec:    {elapsed:.3f}")
print(f"packets_sec:    {N / elapsed:.1f}")
print(f"file_size_mb:   {size_bytes / 1024 / 1024:.2f}")
print(f"write_mb_sec:   {(size_bytes / 1024 / 1024) / elapsed:.2f}")
print(f"output:         {log_path}")
print(f"final_hash:     {prev_hash}")
