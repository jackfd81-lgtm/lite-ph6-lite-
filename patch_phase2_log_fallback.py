#!/usr/bin/env python3
"""
Patch test_ph6lite_phase2.py so stale run_log.jsonl checks can fall back to
hot/spike_events.jsonl.

NOTE: As of commit 89be117, _latest_run() already skips spike-only run dirs
and returns the latest dir that contains run_log.jsonl. This script is kept
as a compatibility patcher for cases where the fallback helper itself is
needed (e.g. if run_log.jsonl is absent from ALL run dirs).
"""

from pathlib import Path
import datetime as dt
import sys

root = Path.home() / "frame_filter"
target = root / "test_ph6lite_phase2.py"

if not target.exists():
    print(f"BLOCK: missing {target}")
    sys.exit(1)

text = target.read_text(encoding="utf-8")

# Check if the 89be117 fix is already present.
if "for d in reversed(runs)" in text:
    print("PASS: _latest_run already contains the 89be117 skip-spike-only fix.")
    print("No patch needed.")
    sys.exit(0)

backup = target.with_suffix(target.suffix + ".bak." + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.write_text(text, encoding="utf-8")

helper = '''
def _resolve_run_jsonl(run_dir):
    """Return path to run event log, falling back to hot/spike_events.jsonl."""
    from pathlib import Path
    run_dir = Path(run_dir)
    for candidate in [
        run_dir / "hot" / "run_log.jsonl",
        run_dir / "run_log.jsonl",
        run_dir / "hot" / "spike_events.jsonl",
    ]:
        if candidate.exists():
            return candidate
    return run_dir / "hot" / "run_log.jsonl"
'''

if "_resolve_run_jsonl" not in text:
    lines = text.splitlines()
    idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            idx = i + 1
    lines.insert(idx, helper)
    text = "\n".join(lines) + "\n"

replacements = {
    'run_dir / "hot" / "run_log.jsonl"': '_resolve_run_jsonl(run_dir)',
    'run_dir / "run_log.jsonl"': '_resolve_run_jsonl(run_dir)',
}

changed = False
for old, new in replacements.items():
    if old in text and new not in text:
        text = text.replace(old, new)
        changed = True

if not changed:
    print("HOLD: helper inserted but no direct pattern replaced — inspect manually.")
    print("Backup:", backup)
    target.write_text(text, encoding="utf-8")
    sys.exit(2)

target.write_text(text, encoding="utf-8")
print("PASS: patched test_ph6lite_phase2.py")
print("Backup:", backup)
