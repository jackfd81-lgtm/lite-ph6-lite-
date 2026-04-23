# PH6 / CRAM Lite

Deterministic single-node frame evaluation system for Raspberry Pi 5, with local LLM advisory via Ollama.

Captures frames from a camera or video file, computes metrics, writes structured packetized logs, and optionally generates LLM-powered advisories for dropped frames.

## Requirements

```bash
sudo apt install -y python3-opencv python3-numpy
```

## Usage

**Camera:**
```bash
python3 frame_filter.py --camera 0 --max_frames 100 --save_mode pass_only
```

**Video file:**
```bash
python3 frame_filter.py --source video.mp4 --save_mode all
```

## Options

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | `0` for camera, or path to video file |
| `--camera` | `0` | Camera index |
| `--log` | `logs/packets.jsonl` | JSONL output path |
| `--frames_dir` | `frames` | Frame save directory |
| `--save_mode` | `pass_only` | `pass_only`, `all`, or `none` |
| `--bright_min` | `40.0` | Minimum brightness |
| `--bright_max` | `220.0` | Maximum brightness |
| `--lap_min` | `40.0` | Minimum Laplacian variance (blur) |
| `--motion_max` | `0.15` | Maximum motion fraction |
| `--max_frames` | `300` | Max frames to process (0 = unlimited) |
| `--llm` | off | Enable LLM advisory for dropped frames |

## Packet structure

Each frame produces 3 packets in order: `pseudo → observation → soso`

Full session: `session_start → config → [pseudo, observation, soso] × N → session_end`

Total packets for N frames: `2 + (3 × N) + 1`

## Verdicts

| Verdict | Meaning |
|---|---|
| `PASS` | Frame meets all thresholds |
| `DROP` | One or more thresholds violated |

---

## `ph6lite/` package

Advisory and schema modules for the capture node.

| Module | Purpose |
|--------|---------|
| `backend.py` | Detects whether a Jetson advisory node is reachable |
| `advisory_client.py` | Routes LLM inference to Jetson or local Ollama |
| `schema_validator.py` | Validates receipt, measurement, SoSo token, and SoSo edge records |
| `atomic_write.py` | Crash-safe JSON file writes (tmp → fsync → rename → dir fsync) |

## `jetson_service/`

Standalone HTTP service to run on a Jetson (or any node with Ollama). No external dependencies — stdlib only.

```
POST /chat    {"prompt": "..."}  →  {"status": "ok", "model": "llama3.2", "output": "..."}
POST /code    {"prompt": "..."}  →  {"status": "ok", "model": "qwen2.5-coder:1.5b", "output": "..."}
POST /reason  {"prompt": "..."}  →  {"status": "ok", "model": "qwen3:1.7b", "output": "..."}
GET  /health                     →  {"status": "ok"}
```

Run on Jetson:
```bash
cd jetson_service
python3 app.py
```

Set env vars on the capture node to route inference to Jetson:
```bash
export PH6_JETSON_HOST=<jetson-ip>
export PH6_JETSON_PORT=8765   # default
```

Leave `PH6_JETSON_HOST` unset to use local Ollama.

## Models

| Mode | Model |
|------|-------|
| chat | llama3.2 |
| code | qwen2.5-coder:1.5b |
| reason | qwen3:1.7b |
