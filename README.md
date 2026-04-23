# PH6 / CRAM Lite v2.1

Deterministic single-node frame evaluation system for Raspberry Pi 5.

Captures frames from a camera or video file, computes metrics, and writes structured packetized logs.

## What it does

- Evaluates every frame for brightness, blur, and motion
- Writes append-only JSONL packet logs
- Optionally saves frame images
- Produces traceable, auditable session records

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

## Packet structure

Each frame produces 3 packets in order: `pseudo → observation → soso`

Full session: `session_start → config → [pseudo, observation, soso] × N → session_end`

Total packets for N frames: `2 + (3 × N) + 1`

## Verdicts

| Verdict | Meaning |
|---|---|
| `PASS` | Frame meets all thresholds |
| `DROP` | One or more thresholds violated |
