#!/usr/bin/env python3
"""
ph6_agents.py

PH6Agents:
A local multi-agent engineering workflow controller for PH6 / CRAM / EVE work.

This is not a trading system.
It adapts the useful multi-agent pattern:
- planner
- builder
- auditor
- governor
- memory
- checkpoint

Authority:
- Advisory only
- No autonomous destructive changes
- No Lane-1 authority mutation
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Tuple


APP_DIR = pathlib.Path.home() / ".ph6_agents"
MEMORY_DIR = APP_DIR / "memory"
CACHE_DIR = APP_DIR / "cache"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
MEMORY_LOG = MEMORY_DIR / "ph6_agents_memory.jsonl"


PH6_RULES = [
    "Lane-1 authority remains deterministic only.",
    "Lane-2 advisory output has Authority ZERO.",
    "No AI/advisory output may alter PASS/DROP directly.",
    "CRAM atomic write contract must not be weakened.",
    "RSYNC Priority Zero must not be blocked.",
    "Generated commands must be reviewable before execution.",
    "Destructive commands require explicit human confirmation.",
]


DANGEROUS_PATTERNS = [
    "rm -rf /",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "sudo rm -rf",
    ":(){",
    "chmod -R 777 /",
    "chown -R",
]


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for p in [APP_DIR, MEMORY_DIR, CACHE_DIR, CHECKPOINT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def canon_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def blake2b256_text(text: str) -> str:
    h = hashlib.blake2b(digest_size=32)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def run_cmd(cmd: List[str], cwd: str | None = None, timeout: int = 20) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": p.returncode,
            "stdout": p.stdout[-6000:],
            "stderr": p.stderr[-6000:],
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": 999,
            "stdout": "",
            "stderr": repr(e),
        }


def append_memory(entry: Dict[str, Any]) -> None:
    ensure_dirs()
    entry = dict(entry)
    entry["ts"] = now_iso()
    line = canon_json(entry)
    with MEMORY_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_checkpoint(name: str, state: Dict[str, Any]) -> pathlib.Path:
    ensure_dirs()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = CHECKPOINT_DIR / f"{safe}.json"
    payload = {
        "checkpoint": name,
        "ts": now_iso(),
        "state": state,
        "hash": blake2b256_text(canon_json(state)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_recent_memory(limit: int = 8) -> List[Dict[str, Any]]:
    if not MEMORY_LOG.exists():
        return []
    lines = MEMORY_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def intake_agent(task: str) -> Dict[str, Any]:
    lower = task.lower()

    intent = "general"
    if any(w in lower for w in ["unfinished", "status", "pending", "git"]):
        intent = "status_audit"
    elif any(w in lower for w in ["fix", "patch", "repair", "make it work"]):
        intent = "repair_plan"
    elif any(w in lower for w in ["test", "run", "verify"]):
        intent = "test_plan"
    elif any(w in lower for w in ["install", "setup", "load"]):
        intent = "setup_plan"

    return {
        "agent": "intake",
        "task": task,
        "intent": intent,
    }


def architect_agent(intake: Dict[str, Any]) -> Dict[str, Any]:
    intent = intake["intent"]

    if intent == "status_audit":
        plan = [
            "Inspect git state for PH6 repos.",
            "Check untracked/modified files.",
            "Check known generated artifacts.",
            "Recommend next commit/test action.",
        ]
    elif intent == "repair_plan":
        plan = [
            "Identify failing artifact or mismatch.",
            "Prefer compatibility patch over redesign.",
            "Preserve CRAM and Lane authority invariants.",
            "Generate commands for review.",
        ]
    elif intent == "test_plan":
        plan = [
            "Run targeted tests first.",
            "Avoid broad pytest if it hangs.",
            "Capture logs to report files.",
            "Emit PASS/HOLD/BLOCK verdict.",
        ]
    elif intent == "setup_plan":
        plan = [
            "Use isolated folder and virtual environment.",
            "Avoid modifying PH6 repos unless requested.",
            "Check imports and command availability.",
            "Write verification report.",
        ]
    else:
        plan = [
            "Classify request.",
            "Produce safe next commands.",
            "Apply PH6 governance checks.",
        ]

    return {
        "agent": "architect",
        "intent": intent,
        "plan": plan,
    }


def builder_agent(architect: Dict[str, Any]) -> Dict[str, Any]:
    intent = architect["intent"]

    commands: List[str] = []

    if intent == "status_audit":
        commands = [
            "cd ~/frame_filter && git status --short",
            "cd ~/frame_filter && python3 test_segment_cram_writer.py",
            "cd ~/frame_filter && python3 ph6lite_coherence_check.py",
            "cd ~/ph6_storage_monitor && git status --short",
            "cd ~/ph6_storage_monitor && python3 test_storage_monitor.py",
        ]
    elif intent == "repair_plan":
        commands = [
            "cd ~/frame_filter && grep -nE 'run_log|spike_events|jsonl|hot/' test_ph6lite_phase2.py frame_filter.py ph6lite_coherence_check.py run_ph6lite_check.sh",
            "cd ~/frame_filter && cp test_ph6lite_phase2.py test_ph6lite_phase2.py.bak.$(date +%Y%m%d_%H%M%S)",
            "# Patch should add fallback: run_log.jsonl -> hot/spike_events.jsonl",
            "cd ~/frame_filter && python3 test_ph6lite_phase2.py",
        ]
    elif intent == "test_plan":
        commands = [
            "cd ~/frame_filter && python3 test_segment_cram_writer.py 2>&1 | tee segment_test_report.txt",
            "cd ~/frame_filter && python3 ph6lite_coherence_check.py 2>&1 | tee coherence_report.txt",
            "cd ~/ph6_storage_monitor && python3 test_storage_monitor.py 2>&1 | tee storage_monitor_test_report.txt",
        ]
    elif intent == "setup_plan":
        commands = [
            "mkdir -p ~/ph6_agents",
            "cd ~/ph6_agents && python3 -m venv .venv",
            "cd ~/ph6_agents && source .venv/bin/activate && python -m pip install --upgrade pip",
        ]
    else:
        commands = [
            "cd ~/frame_filter && git status --short",
            "cd ~/ph6_storage_monitor && git status --short",
        ]

    return {
        "agent": "builder",
        "commands": commands,
    }


def auditor_agent(builder: Dict[str, Any]) -> Dict[str, Any]:
    violations = []

    for cmd in builder.get("commands", []):
        for bad in DANGEROUS_PATTERNS:
            if bad in cmd:
                violations.append({
                    "command": cmd,
                    "violation": f"dangerous pattern: {bad}",
                })

    return {
        "agent": "auditor",
        "rules": PH6_RULES,
        "violations": violations,
        "safe": len(violations) == 0,
    }


def governor_agent(audit: Dict[str, Any]) -> Dict[str, Any]:
    if audit["safe"]:
        verdict = "PASS"
        reason = "No blocked command patterns detected. Commands remain reviewable."
    else:
        verdict = "BLOCK"
        reason = "One or more commands violate PH6 safety patterns."

    return {
        "agent": "governor",
        "verdict": verdict,
        "reason": reason,
        "authority": "ADVISORY_ONLY",
    }


def inspect_system() -> Dict[str, Any]:
    result = {
        "frame_filter_status": run_cmd(["git", "status", "--short"], cwd=str(pathlib.Path.home() / "frame_filter")),
        "storage_monitor_status": run_cmd(["git", "status", "--short"], cwd=str(pathlib.Path.home() / "ph6_storage_monitor")),
        "python": run_cmd(["python3", "--version"]),
        "disk": run_cmd(["df", "-h"]),
        "memory": run_cmd(["free", "-h"]),
    }
    return result


def print_report(report: Dict[str, Any]) -> None:
    print("\n==================================================")
    print("PH6AGENTS REPORT")
    print("==================================================")
    print("Time:", report["ts"])
    print("Task:", report["intake"]["task"])
    print("Intent:", report["intake"]["intent"])

    print("\n--- Architect Plan ---")
    for i, step in enumerate(report["architect"]["plan"], 1):
        print(f"{i}. {step}")

    print("\n--- Proposed Commands ---")
    for cmd in report["builder"]["commands"]:
        print(cmd)

    print("\n--- Audit ---")
    print("Safe:", report["audit"]["safe"])
    if report["audit"]["violations"]:
        for v in report["audit"]["violations"]:
            print("VIOLATION:", v)

    print("\n--- Governor Verdict ---")
    print(report["governor"]["verdict"], "-", report["governor"]["reason"])
    print("Authority:", report["governor"]["authority"])

    print("\n--- Checkpoint ---")
    print(report.get("checkpoint_path", "none"))

    print("\n--- Memory ---")
    print("Memory log:", MEMORY_LOG)


def main() -> int:
    parser = argparse.ArgumentParser(description="PH6Agents local workflow controller")
    parser.add_argument("task", nargs="*", help="Task for PH6Agents")
    parser.add_argument("--inspect", action="store_true", help="Inspect local PH6 repo/system state")
    parser.add_argument("--memory", action="store_true", help="Show recent memory")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    ensure_dirs()

    if args.memory:
        print(json.dumps(load_recent_memory(), ensure_ascii=False, indent=2))
        return 0

    task = " ".join(args.task).strip() or "Check unfinished PH6 work"

    intake = intake_agent(task)
    architect = architect_agent(intake)
    builder = builder_agent(architect)
    audit = auditor_agent(builder)
    governor = governor_agent(audit)

    report: Dict[str, Any] = {
        "ts": now_iso(),
        "intake": intake,
        "architect": architect,
        "builder": builder,
        "audit": audit,
        "governor": governor,
    }

    if args.inspect:
        report["system"] = inspect_system()

    checkpoint_path = save_checkpoint("latest", report)
    report["checkpoint_path"] = str(checkpoint_path)

    append_memory({
        "task": task,
        "intent": intake["intent"],
        "verdict": governor["verdict"],
        "report_hash": blake2b256_text(canon_json(report)),
    })

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
