#!/usr/bin/env bash
# PH6 300-Frame Validation Stack
# One command. Three steps. One verdict.
#
# Steps:
#   1. ph6_synthetic_oracle_300.py   — deterministic known-answer gate
#   2. ph6_full_stack_coherence.py   — 8-layer coherence audit on that run
#   3. replay_cram.py                — CRAM authority + hash chain audit
#
# Exit codes:
#   0  PH6_300_VALIDATION_PASS or _WARN
#   1  PH6_300_VALIDATION_FAIL
#
# WARN is reported when any step exits 0 but emitted WARN in its output.
# Steps that exit non-zero always produce FAIL.

set +e

WORKDIR="$HOME/frame_filter"
cd "$WORKDIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$HOME/ph6_300_validation_$STAMP.txt"

BOLD=$'\033[1m'
RED=$'\033[91m'
GREEN=$'\033[92m'
YELLOW=$'\033[93m'
CYAN=$'\033[96m'
RESET=$'\033[0m'

FAILS=0

banner() {
    echo
    echo "${BOLD}══════════════════════════════════════════════════════════${RESET}"
    echo "${BOLD}  $1${RESET}"
    echo "${BOLD}══════════════════════════════════════════════════════════${RESET}"
}

step_result() {
    local name="$1" rc="$2"
    if [ "$rc" -eq 0 ]; then
        echo "  ${BOLD}${GREEN}PASS${RESET}  $name"
    else
        echo "  ${BOLD}${RED}FAIL${RESET}  $name  (exit $rc)"
        FAILS=$((FAILS + 1))
    fi
}

latest_run() {
    ls -td logs/run_* 2>/dev/null | head -1
}

{

banner "PH6 300-FRAME VALIDATION"
echo "  Time:    $(date -Is)"
echo "  Workdir: $WORKDIR"
echo "  Report:  $REPORT"
echo

# ── Step 1: Oracle known-answer test ──────────────────────────────────────────
banner "STEP 1 — PH6_SYNTHETIC_ORACLE_300"
python3 ph6_synthetic_oracle_300.py
ORACLE_RC=$?
step_result "ph6_synthetic_oracle_300" $ORACLE_RC

ORACLE_RUN="$(latest_run)"
echo "  Run dir: ${ORACLE_RUN:-NONE}"

if [ -z "$ORACLE_RUN" ]; then
    echo
    echo "  ${BOLD}${RED}HARD FAIL${RESET} — no run directory found after oracle step"
    echo "  ${BOLD}${RED}PH6_300_VALIDATION_FAIL${RESET}"
    exit 1
fi

# ── Step 2: Full-stack coherence audit on oracle run ──────────────────────────
banner "STEP 2 — PH6_FULL_STACK_COHERENCE"
python3 ph6_full_stack_coherence.py --run-dir "$ORACLE_RUN"
COHERENCE_RC=$?
step_result "ph6_full_stack_coherence" $COHERENCE_RC

# ── Step 3: CRAM replay / authority + hash chain audit ────────────────────────
banner "STEP 3 — REPLAY_CRAM"
python3 replay_cram.py --run "$ORACLE_RUN"
REPLAY_RC=$?
step_result "replay_cram" $REPLAY_RC

# ── Final verdict ─────────────────────────────────────────────────────────────
banner "VALIDATION SUMMARY"
printf "  %-28s  %s\n" "Oracle (known-answer):"     "$([ $ORACLE_RC    -eq 0 ] && echo PASS || echo FAIL)"
printf "  %-28s  %s\n" "Full-stack coherence:"       "$([ $COHERENCE_RC -eq 0 ] && echo PASS || echo FAIL)"
printf "  %-28s  %s\n" "CRAM replay audit:"          "$([ $REPLAY_RC    -eq 0 ] && echo PASS || echo FAIL)"
echo
printf "  %-28s  %s\n" "Run dir:" "$ORACLE_RUN"
echo

if [ "$FAILS" -gt 0 ]; then
    echo "  ${BOLD}${RED}PH6_300_VALIDATION_FAIL${RESET}"
    FINAL_VERDICT="PH6_300_VALIDATION_FAIL"
    EXIT_CODE=1
else
    echo "  ${BOLD}${GREEN}PH6_300_VALIDATION_PASS${RESET}"
    FINAL_VERDICT="PH6_300_VALIDATION_PASS"
    EXIT_CODE=0
fi

} 2>&1 | tee "$REPORT"

echo
echo "Saved: $REPORT"
exit $EXIT_CODE
