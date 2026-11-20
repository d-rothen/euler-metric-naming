"""Generated artifacts are up to date; the TS conformance suite passes under node."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_fixture_eval_files_are_up_to_date():
    result = _run_script("generate_fixtures.py", "--check")
    assert result.returncode == 0, result.stdout + result.stderr
