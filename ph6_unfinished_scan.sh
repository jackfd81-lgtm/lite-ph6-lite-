#!/usr/bin/env bash
set +e

REPORT="$HOME/ph6_unfinished_scan_$(date +%Y%m%d_%H%M%S).txt"

{
echo "=================================================="
echo "PH6 UNFINISHED WORK SCAN"
echo "=================================================="
date
hostname
echo

echo "=================================================="
echo "FRAME_FILTER"
echo "=================================================="
if [ -d "$HOME/frame_filter" ]; then
  cd "$HOME/frame_filter"
  pwd
  git status --short
  echo
  echo "--- recent commits ---"
  git log --oneline -5
  echo
  echo "--- known generated leftovers ---"
  find . -maxdepth 2 -type f \( -name "*report*.json" -o -name "*report*.txt" -o -name "*.bak.*" \) -print
else
  echo "MISSING ~/frame_filter"
fi

echo
echo "=================================================="
echo "FRAME_FILTER TARGETED TESTS"
echo "=================================================="
if [ -d "$HOME/frame_filter" ]; then
  cd "$HOME/frame_filter"

  if [ -f test_segment_cram_writer.py ]; then
    python3 test_segment_cram_writer.py
  else
    echo "MISSING test_segment_cram_writer.py"
  fi

  echo
  if [ -f ph6lite_coherence_check.py ]; then
    python3 ph6lite_coherence_check.py
  else
    echo "MISSING ph6lite_coherence_check.py"
  fi

  echo
  echo "--- phase2 log path scan ---"
  grep -nE "run_log|spike_events|jsonl|hot/" test_ph6lite_phase2.py frame_filter.py ph6lite_coherence_check.py run_ph6lite_check.sh 2>/dev/null
fi

echo
echo "=================================================="
echo "PH6 STORAGE MONITOR"
echo "=================================================="
if [ -d "$HOME/ph6_storage_monitor" ]; then
  cd "$HOME/ph6_storage_monitor"
  pwd
  git status --short
  echo
  git log --oneline -5
  echo
  if [ -f test_storage_monitor.py ]; then
    python3 test_storage_monitor.py
  else
    echo "MISSING test_storage_monitor.py"
  fi
else
  echo "MISSING ~/ph6_storage_monitor"
fi

echo
echo "=================================================="
echo "SYSTEM PACKAGE STATE"
echo "=================================================="
sudo dpkg --audit
sudo apt --fix-broken install --dry-run
apt list --upgradable 2>/dev/null | tail -50

echo
echo "=================================================="
echo "SERVICES / REBOOT"
echo "=================================================="
if [ -f /var/run/reboot-required ]; then
  echo "REBOOT REQUIRED"
  cat /var/run/reboot-required.pkgs 2>/dev/null
else
  echo "No reboot-required flag found."
fi

systemctl --failed

} 2>&1 | tee "$REPORT"

echo
echo "Saved report:"
echo "$REPORT"
