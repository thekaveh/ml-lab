"""Tests for scripts/verify_repo.py — the four-check oracle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_repo.py"


def run_verify(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_help_lists_all_checks():
    r = run_verify("--help")
    assert r.returncode == 0
    for ch in ("structure", "execution", "docs", "comments", "all"):
        assert ch in r.stdout


def test_unknown_check_errors(tmp_path):
    r = run_verify("--check", "garbage")
    assert r.returncode != 0


def test_emits_valid_json_schema(tmp_path):
    out = tmp_path / "findings.json"
    r = run_verify("--check", "structure", "--out", str(out), "--fast")
    assert out.exists(), f"no output file; stderr={r.stderr}"
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "schema_version" in data
    assert data["schema_version"] == 1
    assert "findings" in data
    assert isinstance(data["findings"], list)
    assert "summary" in data
    assert "checks_run" in data["summary"]
    assert "structure" in data["summary"]["checks_run"]


def test_finding_shape():
    """Every finding must have id, check, severity, location, message."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    for f in data.get("findings", []):
        assert {"id", "check", "severity", "location", "message"} <= set(f.keys())
        assert f["severity"] in ("error", "warning")
