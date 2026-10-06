"""The installer must refuse to install with a missing, empty or placeholder operator login."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[2] / "infra" / "ec2" / "setup" / "install_console_agent.sh"
# A bare "bash" on Windows resolves to System32 (WSL) before PATH, which cannot see C:/ paths.
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def run_installer(repo_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "REPO_DIR": repo_dir.as_posix()}
    return subprocess.run([BASH, INSTALLER.as_posix()], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize(
    "env_line",
    ["", 'CONSOLE_OPERATOR_LOGIN=""', "CONSOLE_OPERATOR_LOGIN=", 'CONSOLE_OPERATOR_LOGIN="<YOUR_TAILSCALE_LOGIN>"'],
    ids=["absent", "empty-quoted", "empty", "placeholder"],
)
def test_installer_refuses_an_unusable_login(tmp_path, env_line):
    (tmp_path / ".env").write_text(f"OTHER=1\n{env_line}\n")

    result = run_installer(tmp_path)

    assert result.returncode == 1
    assert "CONSOLE_OPERATOR_LOGIN" in result.stdout
    assert "Installing" not in result.stdout


def test_installer_accepts_a_real_login(tmp_path):
    (tmp_path / ".env").write_text('CONSOLE_OPERATOR_LOGIN="operator@github"\n')

    result = run_installer(tmp_path)

    assert "[+] Installing homeguard-console-agent.service" in result.stdout
