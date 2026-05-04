#!/usr/bin/env bash
set -euo pipefail

# PH6-Lite Repeatable Validation Workflow
#
# Verdicts:
#   PH6_LITE_VALIDATION_PASS
#   PH6_LITE_VALIDATION_WARN
#   PH6_LITE_VALIDATION_FAIL
#   PH6_LITE_VALIDATION_INVALID  (manual stop / under-300 with no evidence)

ROOT="$HOME/frame_filter"
CRAM_PU_LAUNCHER="$HOME/cram_pu/run_cram_pu.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/validation_runs/$STAMP"
mkdir -p "$OUT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/validation.log"; }

_verdict() {
    local v="$1" code="$2"
    log "FINAL_VERDICT=$v"
    echo "$v"
    exit "$code"
}
pass()    { _verdict PH6_LITE_VALIDATION_PASS    0; }
warn()    { _verdict PH6_LITE_VALIDATION_WARN    0; }
fail()    { _verdict PH6_LITE_VALIDATION_FAIL    1; }
invalid() { _verdict PH6_LITE_VALIDATION_INVALID 2; }

# ── preflight ──────────────────────────────────────────────────────────────────
log "PH6-Lite validation started — $STAMP"
log "Output: $OUT"
cd "$ROOT"

git status --short > "$OUT/git_status_before.txt" 2>/dev/null || true
git log --oneline -5 > "$OUT/git_log_before.txt"  2>/dev/null || true

for f in frame_filter.py ph6_synthetic_oracle_300.py ph6_full_stack_coherence.py replay_cram.py; do
    [ -f "$f" ] || { log "Missing required file: $f"; fail; }
done

# ── CRAM-PU health ────────────────────────────────────────────────────────────
log "Checking CRAM-PU health"
if curl -fsS http://127.0.0.1:8765/health > "$OUT/cram_pu_health.json" 2>"$OUT/cram_pu_health.err"; then
    log "CRAM-PU health PASS"
else
    log "CRAM-PU not responding — attempting start"
    [ -f "$CRAM_PU_LAUNCHER" ] || { log "CRAM-PU launcher missing"; fail; }
    nohup bash "$CRAM_PU_LAUNCHER" > "$OUT/cram_pu_stdout.log" 2>"$OUT/cram_pu_stderr.log" &
    sleep 3
    if curl -fsS http://127.0.0.1:8765/health > "$OUT/cram_pu_health.json" 2>"$OUT/cram_pu_health.err"; then
        log "CRAM-PU started — health PASS"
    else
        log "CRAM-PU failed health after start"; fail
    fi
fi

# ── token leakage + ordering invariant test ───────────────────────────────────
log "Running token leakage test (7 assertions)"
if python3 test_token_leakage.py > "$OUT/token_leakage.stdout" 2>"$OUT/token_leakage.stderr"; then
    log "Token leakage PASS"
else
    log "Token leakage FAIL"; fail
fi

# ── oracle known-answer test ──────────────────────────────────────────────────
log "Running oracle known-answer test"
if python3 ph6_synthetic_oracle_300.py > "$OUT/oracle_300.stdout" 2>"$OUT/oracle_300.stderr"; then
    log "Oracle 300 PASS"
else
    log "Oracle 300 FAIL"; fail
fi

# ── 300-frame pipeline run ─────────────────────────────────────────────────────
log "Running 300-frame oracle pipeline"
BEFORE_RUNS="$(find "$ROOT/logs" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort || true)"
set +e
python3 frame_filter.py --source oracle --max_frames 300 --postrun \
    > "$OUT/frame_filter_300.stdout" 2>"$OUT/frame_filter_300.stderr"
FRAME_EXIT=$?
set -e
AFTER_RUNS="$(find "$ROOT/logs" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort || true)"
LATEST_RUN="$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | tail -1 || true)"
[ -n "$LATEST_RUN" ] || LATEST_RUN="$(find "$ROOT/logs" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort | tail -1 || true)"
echo "$LATEST_RUN" > "$OUT/latest_run.txt"

if [ "$FRAME_EXIT" -ne 0 ]; then
    log "frame_filter.py exited $FRAME_EXIT"
    [ -n "$LATEST_RUN" ] && log "Run dir present: $LATEST_RUN" || { log "No run dir — invalid"; invalid; }
fi
[ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN" ] || { log "No valid run dir"; invalid; }
log "Run dir: $LATEST_RUN"

# ── coherence audit ───────────────────────────────────────────────────────────
log "Running full-stack coherence audit"
python3 ph6_full_stack_coherence.py --run-dir "$LATEST_RUN" \
    > "$OUT/coherence.stdout" 2>"$OUT/coherence.stderr" || { log "Coherence audit FAIL"; fail; }

# ── replay audit ──────────────────────────────────────────────────────────────
log "Running CRAM replay audit"
python3 replay_cram.py --run "$LATEST_RUN" \
    > "$OUT/replay.stdout" 2>"$OUT/replay.stderr" || { log "Replay audit FAIL"; fail; }

# ── receipt + hashes ──────────────────────────────────────────────────────────
python3 - <<PYEOF > "$OUT/validation_receipt.json"
import json, datetime
print(json.dumps({
    "schema": "ph6.lite.validation.receipt.v1",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "validation_dir": "$OUT",
    "latest_run": "$LATEST_RUN",
    "required_min_frames": 300,
    "source": "oracle",
    "authority_rule": "Lane-1 only; Lane-2 Authority ZERO",
    "manual_stop_before_300": "INVALID",
    "verdicts": ["PH6_LITE_VALIDATION_PASS","PH6_LITE_VALIDATION_WARN",
                 "PH6_LITE_VALIDATION_FAIL","PH6_LITE_VALIDATION_INVALID"]
}, indent=2))
PYEOF

find "$OUT" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/artifact_hashes.sha256" 2>/dev/null || true
log "Artifacts: $OUT"

# ── final verdict ─────────────────────────────────────────────────────────────
if grep -qE "^\s+(FAIL|HARD FAIL|REJECT BUILD)" "$OUT/coherence.stdout" "$OUT/replay.stdout" 2>/dev/null; then
    log "FAIL marker in audit output"; fail
fi
if grep -qE "^\s+WARN" "$OUT/coherence.stdout" "$OUT/replay.stdout" 2>/dev/null; then
    log "WARN marker in audit output"; warn
fi
pass
