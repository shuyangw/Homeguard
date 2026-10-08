import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHONW = Path(sys.executable).with_name("pythonw.exe")


@pytest.mark.skipif(sys.platform != "win32" or not PYTHONW.exists(), reason="needs Windows with pythonw.exe")
def test_pythonw_run_logs_refusal_to_file(tmp_path):
    env = {
        **os.environ,
        "LOCALAPPDATA": str(tmp_path),
        "CONSOLE_AGENT_URL": "https://agent.invalid:8443",
        "CONSOLE_SNAPSHOT_BUCKET": "none",
        "EC2_INSTANCE_ID": "i-0",
        "EC2_REGION": "us-east-1",
        "CONSOLE_AWS_PROFILE": "does-not-exist",
    }

    # DETACHED_PROCESS with no redirection leaves sys.stdout and sys.stderr as None, as in the logon task.
    result = subprocess.run(
        [str(PYTHONW), "-m", "tools.console"],
        cwd=REPO_ROOT,
        env=env,
        timeout=60,
        creationflags=subprocess.DETACHED_PROCESS,
    )

    log_file = tmp_path / "Homeguard" / "console.log"
    assert result.returncode == 1
    assert log_file.exists()
    assert "refusing to start" in log_file.read_text(encoding="utf-8")
