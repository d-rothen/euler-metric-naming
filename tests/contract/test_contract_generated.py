"""Generated artifacts are up to date; the TS conformance suite passes under node."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_ts_data_is_up_to_date():
    result = _run_script("generate_ts.py", "--check")
    assert result.returncode == 0, result.stdout + result.stderr


def _node():
    node = shutil.which("node")
    if node is None:
        nvm = Path(os.environ.get("NVM_DIR", Path.home() / ".nvm")) / "versions" / "node"
        candidates = sorted(nvm.glob("v*/bin/node")) if nvm.is_dir() else []
        node = str(candidates[-1]) if candidates else None
    if node is None:
        return None
    version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    major = int(version.lstrip("v").split(".")[0]) if version.startswith("v") else 0
    return node if major >= 22 else None


def test_ts_conformance_suite_under_node():
    node = _node()
    if node is None:
        pytest.skip("node >= 22 is not available")
    test_file = ROOT / "euler_metric_naming" / "contract" / "ts" / "conformance.test.ts"
    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings", "--test", str(test_file)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "# fail 0" in result.stdout
