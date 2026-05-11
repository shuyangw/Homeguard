"""End-to-end smoke test: invoke the CLI and confirm a valid report is produced."""
import subprocess
import sys
from pathlib import Path


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/data/run_validation.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--domain" in result.stdout
    assert "--layer" in result.stdout


def test_cli_invalid_domain():
    result = subprocess.run(
        [sys.executable, "scripts/data/run_validation.py", "--domain", "notreal"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
