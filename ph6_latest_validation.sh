#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/frame_filter/validation_runs"
LATEST="$(find "$ROOT" -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"

[ -n "$LATEST" ] || { echo "No validation runs found."; exit 1; }

echo "Latest validation: $LATEST"
echo
echo "Final verdict:"
grep "FINAL_VERDICT=" "$LATEST/validation.log" | tail -1 || echo "(no verdict)"
echo
echo "Run dir:"
cat "$LATEST/latest_run.txt" 2>/dev/null || echo "(none)"
echo
echo "Receipt:"
cat "$LATEST/validation_receipt.json" 2>/dev/null || echo "(none)"
