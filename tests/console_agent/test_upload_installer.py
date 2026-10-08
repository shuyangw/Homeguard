"""The uploader installer refuses a missing bucket, and the units keep the shutdown ordering."""

import configparser
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "infra" / "ec2" / "setup" / "install_console_upload.sh"
SERVICES = REPO_ROOT / "infra" / "ec2" / "services"
BASH = shutil.which("bash")


def read_unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    parser.read(SERVICES / name)
    return parser


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize(
    "env_line",
    ["", 'CONSOLE_SNAPSHOT_BUCKET=""', "CONSOLE_SNAPSHOT_BUCKET=", 'CONSOLE_SNAPSHOT_BUCKET="<YOUR_BUCKET_NAME>"'],
    ids=["absent", "empty-quoted", "empty", "placeholder"],
)
def test_installer_refuses_an_unusable_bucket(tmp_path, env_line):
    (tmp_path / ".env").write_text(f"OTHER=1\n{env_line}\n")
    env = {**os.environ, "REPO_DIR": tmp_path.as_posix()}

    result = subprocess.run([BASH, INSTALLER.as_posix()], env=env, capture_output=True, text=True, timeout=30)

    assert result.returncode == 1
    assert "CONSOLE_SNAPSHOT_BUCKET" in result.stdout
    assert "Installing" not in result.stdout


def test_shutdown_unit_stops_before_the_trading_units():
    unit = read_unit("homeguard-console-upload-shutdown.service")

    after = unit["Unit"]["After"].split()
    for dependency in ("network-online.target", "homeguard-multi.service", "homeguard-cscm.service",
                       "homeguard-gateway.service", "homeguard-console-agent.service"):
        assert dependency in after
    assert unit["Service"]["RemainAfterExit"] == "yes"
    assert "--reason shutdown" in unit["Service"]["ExecStop"]
    assert unit["Service"]["TimeoutStopSec"] == "60"


def test_periodic_unit_and_timer():
    service = read_unit("homeguard-console-upload.service")
    timer = read_unit("homeguard-console-upload.timer")

    assert service["Service"]["Type"] == "oneshot"
    assert "--reason periodic" in service["Service"]["ExecStart"]
    assert service["Service"]["MemoryMax"] == "256M"
    assert timer["Timer"]["OnUnitActiveSec"] == "5min"
    assert timer["Install"]["WantedBy"] == "timers.target"


def test_installer_verifies_the_upload_without_list_permission():
    # The instance role only has s3:PutObject; `aws s3 ls` would fail after a good upload.
    assert "aws s3 ls" not in INSTALLER.read_text()
