"""
pytest conftest for frame_filter test suite.
Provides the run_dir fixture from the latest log run that has hot/run_log.jsonl.
"""
import pytest
from pathlib import Path

LOGS = Path(__file__).parent / "logs"


def _latest_run_dir() -> Path:
    candidates = sorted(
        (d for d in LOGS.iterdir()
         if d.is_dir() and (d / "hot" / "run_log.jsonl").exists()),
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        # Fall back to legacy flat layout
        candidates = sorted(
            (d for d in LOGS.iterdir()
             if d.is_dir() and (d / "run_log.jsonl").exists()),
            key=lambda d: d.name,
            reverse=True,
        )
    if not candidates:
        pytest.skip("No frame_filter run data found — run frame_filter first")
    return candidates[0]


@pytest.fixture(scope="session")
def run_dir() -> Path:
    return _latest_run_dir()
