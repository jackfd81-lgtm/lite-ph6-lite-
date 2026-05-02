#!/usr/bin/env bash
set -u

cd "$HOME/frame_filter" || {
  echo "FAIL — cannot cd into ~/frame_filter"
  exit 1
}

echo "=========================================="
echo "PH6-Lite Full Coherence Check"
echo "Root: $(pwd)"
echo "Time: $(date -Is)"
echo "=========================================="

echo
echo "[1/6] Python version"
python3 --version || exit 1

echo
echo "[2/6] Project files"
ls -lah

echo
echo "[3/6] Existing tests"
python3 test_token_leakage.py || exit 1
python3 test_ph6lite_phase2.py || exit 1

echo
echo "[4/6] Master coherence checker"
python3 ph6lite_coherence_check.py || exit 1

echo
echo "[5/6] Camera devices"
ls -lah /dev/video* 2>/dev/null || echo "WARN — no /dev/video* devices found"

echo
echo "[6/6] Disk and mounts"
df -h .
findmnt -T . -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS

echo
echo "=========================================="
echo "FINAL: PASS — PH6-Lite check completed"
echo "=========================================="
