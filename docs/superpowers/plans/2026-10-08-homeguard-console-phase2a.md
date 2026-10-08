# Homeguard Console Phase 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only local console (FastAPI + htmx) that shows whether Homeguard is up and trading as intended, and keeps showing the last known state from an S3 snapshot after the instance stops.

**Architecture:** On the instance, `src/console_agent/upload.py` builds the agent's own `/status` document plus the latest decision records and puts it to S3 with the AWS CLI every 5 minutes and at shutdown. On the operator's machines, `tools/console/` polls the agent over the tailnet, falls back to the S3 document, reads EC2 and the Scheduler with a read-only IAM user, and renders panels from one in-process `ConsoleState`. Pure modules (`schedule.py`, `checks.py`, `freshness.py`, `views.py`) hold the logic; `poller.py` and `app.py` are thin I/O.

**Tech Stack:** Python 3.11 (fintech env), FastAPI 0.126, Starlette 0.50, Jinja2 3.1, httpx 0.28, boto3 1.43 with botocore Stubber, pandas_market_calendars, htmx 2.0.4 (vendored), systemd, Terraform (AWS provider), bash, PowerShell.

**Spec:** `docs/superpowers/specs/2026-10-08-homeguard-console-phase2a-design.md` (parent: `docs/superpowers/specs/2026-10-05-homeguard-console-design.md`)

## Global Constraints

- Work in the worktree `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\.worktrees\console-p2a` on branch `feat/console-p2a`, created from `main`. Never implement in the main working tree.
- Python: `PY=/c/Users/qwqw1/anaconda3/envs/fintech/python.exe`; run tests as `$PY -m pytest <paths> -q -p no:cacheprovider`.
- ASCII only in code, comments, logs, templates and docs. No emojis, no em dashes, no Unicode arrows.
- Logging: `from src.utils.logger import logger`, f-strings only (the logger does not take `%s` args), never `print()`.
- The uploader must import nothing beyond what the agent imports (`src.console_agent.status`, `src.utils.logger`, `src.settings`, `dotenv`, stdlib); never `src.trading`, never boto3. The agent's MemoryMax is 96M and the uploader's is 256M.
- The local app binds to 127.0.0.1 only; port 8765. Host header must be `127.0.0.1` or `localhost` (any port), else 400.
- 2a has no POST routes and grants no `ec2:StartInstances`/`ec2:StopInstances`.
- S3 object key: `console/latest/status.json`. Bucket name from `CONSOLE_SNAPSHOT_BUCKET` (instance .env and local .env) and Terraform variable `console_snapshot_bucket`.
- Uploader cadence: every 5 minutes (`OnBootSec=2min`, `OnUnitActiveSec=5min`) and at shutdown. Upload timeout 30 s; shutdown unit `TimeoutStopSec=60`.
- Poll cadence: agent every 10 s (5 s timeout); S3 every 60 s only while the agent is unreachable; EC2 every 30 s; schedules at startup and every 10 minutes.
- The four schedules: `homeguard-start-instance` `cron(0 8 ? * MON-FRI *)` and `homeguard-stop-instance` `cron(0 20 ? * MON-FRI *)` in America/New_York; `homeguard-start-instance-sunday` `cron(0 23 ? * SAT *)` and `homeguard-stop-instance-sunday` `cron(10 0 ? * SUN *)` in UTC.
- Thresholds: heartbeat older than 120 s is a warning; decision grace 5 minutes; drawdown (a percent in [-100, 0], negative by construction) caution at or below -10, warning at or below -20; homeguard-multi memory against 1G (1073741824 bytes), caution above 80%, warning above 95%; agent unreachable for over 2 minutes while running is a warning; power lock is the NYSE session plus or minus 15 minutes.
- Decision times: `{"ramp": ("15:55",), "omr": ("09:31", "15:50")}` (ET).
- Secrets: no account IDs, bucket names, tailnet hostnames or instance IDs in committed files; use `<YOUR_VALUE>` placeholders in examples.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk
  ```
- `docs/superpowers/` is matched by a `docs/*` ignore rule; plan and spec files are added with `git add -f`.
- Shell gotcha: Bash tool calls whose cwd is a deploy-branch worktree can fail the repo's PreToolUse hook; use `git -C <path>` from the main dir in that case.

## Review Focus

1. A missing S3 object reads as `AccessDenied`, not `NoSuchKey`, unless the reader also has `s3:ListBucket`; the operator should see "No snapshot yet" for a missing object and an error for real denials. Pinned by the `s3:ListBucket` grant in Task 3 and `test_s3_access_denied_is_an_error_not_no_snapshot` in Task 7.
2. The agent answering 403 (login mismatch) or 5xx must count as unreachable, fall back to S3 and show the HTTP error, not crash the poller. Pinned by `test_agent_403_falls_back_to_s3_and_reports_the_error` in Task 7.
3. Naive or unparseable decision timestamps must not crash the checks. Pinned by `test_naive_decision_timestamp_is_read_as_eastern` and `test_unparseable_decision_timestamp_reads_as_missed` in Task 6.
4. The DST change on 2026-11-01 must not shift the expected state: Monday 2026-11-02 07:59 EST stopped, 08:00 EST running, and the Saturday UTC window still lands on Saturday evening ET. Pinned by `test_expected_state_across_the_november_dst_change` in Task 5.
5. A reading timestamp slightly in the future (clock skew between machines) must not render a negative age. Pinned by `test_age_text_clamps_future_readings_to_zero` in Task 6.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/console_agent/upload.py` (create) | Build the status document and upload it with `aws s3 cp` |
| `infra/ec2/services/homeguard-console-upload.service` (create) | Oneshot periodic upload |
| `infra/ec2/services/homeguard-console-upload.timer` (create) | Every 5 minutes |
| `infra/ec2/services/homeguard-console-upload-shutdown.service` (create) | Upload in `ExecStop=` at shutdown |
| `infra/ec2/setup/install_console_upload.sh` (create) | Idempotent installer |
| `infra/terraform/console.tf` (create), `variables.tf`, `terraform.tfvars.example` (modify) | Bucket, instance upload grant, read-only IAM user |
| `tools/__init__.py`, `tools/console/__init__.py` (create) | Package markers |
| `tools/console/config.py` | `Settings` from the environment |
| `tools/console/decision_times.py` | Decision-times table |
| `tools/console/schedule.py` | Cron parsing, expected state, NYSE session, power lock, power status |
| `tools/console/freshness.py` | Freshness classes and age text |
| `tools/console/checks.py` | The eight exception checks |
| `tools/console/poller.py` | `ConsoleState`, `AgentClient`, `AwsClients`, `Poller` |
| `tools/console/views.py` | Template contexts from `ConsoleState` |
| `tools/console/app.py`, `tools/console/__main__.py` | FastAPI app and entry point |
| `tools/console/templates/`, `tools/console/static/` | Jinja templates, CSS, vendored htmx |
| `tools/console/requirements.txt`, `README.md`, `install_windows_task.ps1`, `install_macos_launchd.sh`, `com.homeguard.console.plist` | Install and run |
| `tests/console_agent/test_upload.py`, `tests/console_agent/test_upload_installer.py` | Uploader tests |
| `tests/console_app/` | Local app tests |

---

### Task 1: Snapshot uploader module

**Files:**
- Create: `src/console_agent/upload.py`
- Test: `tests/console_agent/test_upload.py`
- Modify: `src/console_agent/README.md` (add an "Uploader" section)

**Interfaces:**
- Consumes: `status.AgentConfig(repo_root, snapshot_dir, operator_login)`, `status.build_status(config, now) -> dict`, `status.read_latest_decision(latest_dir, strategy) -> dict | None`, `status.READ_ERRORS`; fixtures `agent_config` and `fake_systemd` from `tests/console_agent/conftest.py`.
- Produces: `upload.OBJECT_KEY = "console/latest/status.json"`; the document `{"uploaded_at": str, "reason": "periodic"|"shutdown", "status": <build_status dict>, "decisions": {strategy: dict | None}}`, which Task 7 reads.

- [ ] **Step 1: Create the worktree**

```bash
cd /c/Users/qwqw1/Dropbox/cs/github/Homeguard
git worktree add -b feat/console-p2a .worktrees/console-p2a main
cd .worktrees/console-p2a
```

- [ ] **Step 2: Write the failing tests**

`tests/console_agent/test_upload.py`:

```python
"""The uploader puts the agent's /status shape plus the latest decisions to S3 and fails loudly."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.console_agent import upload

NOW = datetime(2026, 10, 8, 0, 0, 5, tzinfo=timezone.utc)
BUCKET_ENV = {"CONSOLE_SNAPSHOT_BUCKET": "test-bucket"}


class FakeRun:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.uploaded = None

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        with open(args[3]) as handle:
            self.uploaded = json.load(handle)
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(args, 0, "", "")


def test_document_has_the_status_shape_and_full_decisions(agent_config, fake_systemd):
    document = upload.build_document(agent_config, NOW, "periodic")

    assert set(document) == {"uploaded_at", "reason", "status", "decisions"}
    assert document["uploaded_at"] == NOW.isoformat()
    assert document["reason"] == "periodic"
    assert set(document["status"]) == {"generated_at", "units", "strategies", "execution_lock", "errors"}
    assert document["decisions"]["ramp"]["decision_id"] == "ramp-20261005-1555"
    assert document["decisions"]["omr"] is None


def test_upload_copies_the_document_to_the_fixed_key(agent_config, fake_systemd):
    run = FakeRun()

    code = upload.run_upload(agent_config, BUCKET_ENV, "shutdown", NOW, run)

    assert code == 0
    args, kwargs = run.calls[0]
    assert args[:3] == ["aws", "s3", "cp"]
    assert args[4] == "s3://test-bucket/console/latest/status.json"
    assert kwargs["timeout"] == upload.UPLOAD_TIMEOUT_SECONDS
    assert kwargs["check"] is True
    assert run.uploaded["reason"] == "shutdown"


def test_missing_bucket_exits_non_zero_without_uploading(agent_config, fake_systemd):
    run = FakeRun()

    assert upload.run_upload(agent_config, {}, "periodic", NOW, run) == 1
    assert run.calls == []


def test_failed_copy_exits_non_zero(agent_config, fake_systemd):
    run = FakeRun(error=subprocess.CalledProcessError(1, ["aws"], stderr="AccessDenied"))

    assert upload.run_upload(agent_config, BUCKET_ENV, "periodic", NOW, run) == 1


def test_timed_out_copy_exits_non_zero(agent_config, fake_systemd):
    run = FakeRun(error=subprocess.TimeoutExpired(["aws"], 30))

    assert upload.run_upload(agent_config, BUCKET_ENV, "periodic", NOW, run) == 1


def test_missing_aws_cli_exits_non_zero(agent_config, fake_systemd):
    run = FakeRun(error=FileNotFoundError("aws"))

    assert upload.run_upload(agent_config, BUCKET_ENV, "periodic", NOW, run) == 1


def test_temporary_file_is_removed_after_upload(agent_config, fake_systemd):
    run = FakeRun()

    upload.run_upload(agent_config, BUCKET_ENV, "periodic", NOW, run)

    assert not Path(run.calls[0][0][3]).exists()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_agent/test_upload.py -q -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'upload'`

- [ ] **Step 4: Implement the uploader**

`src/console_agent/upload.py`:

```python
"""Uploads the console status document to S3: python -m src.console_agent.upload --reason periodic|shutdown.

The document is the agent's own /status shape plus the full latest decision per
strategy, so the local console renders one shape whichever source it read.
Uses the AWS CLI already on the instance, so the venv needs no boto3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from dotenv import load_dotenv

from src.console_agent import status
from src.settings import get_local_storage_dir
from src.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
BUCKET_ENV = "CONSOLE_SNAPSHOT_BUCKET"
OBJECT_KEY = "console/latest/status.json"
UPLOAD_TIMEOUT_SECONDS = 30
REASONS = ("periodic", "shutdown")


def build_document(config: status.AgentConfig, now: datetime, reason: str) -> dict:
    document = status.build_status(config, now)
    decisions = {}
    for name in document["strategies"]:
        try:
            decisions[name] = status.read_latest_decision(config.latest_dir, name)
        except status.READ_ERRORS:
            # build_status already put this read failure in document["errors"].
            decisions[name] = None
    return {"uploaded_at": now.isoformat(), "reason": reason, "status": document, "decisions": decisions}


def upload(document: dict, bucket: str, run: Callable = subprocess.run) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(document, handle, default=str)
        tmp_path = Path(handle.name)
    try:
        run(
            ["aws", "s3", "cp", str(tmp_path), f"s3://{bucket}/{OBJECT_KEY}", "--only-show-errors"],
            capture_output=True, text=True, timeout=UPLOAD_TIMEOUT_SECONDS, check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[console-upload] aws s3 cp failed with exit {e.returncode}: {(e.stderr or '').strip()}")
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"[console-upload] aws s3 cp did not complete: {e!r}")
    finally:
        tmp_path.unlink(missing_ok=True)
    return False


def run_upload(config: status.AgentConfig, environ: Mapping[str, str], reason: str, now: datetime,
               run: Callable = subprocess.run) -> int:
    bucket = environ.get(BUCKET_ENV, "").strip()
    if not bucket:
        logger.error(f"[console-upload] {BUCKET_ENV} is not set")
        return 1
    if not upload(build_document(config, now, reason), bucket, run):
        return 1
    logger.info(f"[console-upload] uploaded {reason} snapshot to s3://{bucket}/{OBJECT_KEY}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the Homeguard Console status document to S3")
    parser.add_argument("--reason", choices=REASONS, required=True)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    snapshot_dir = Path(get_local_storage_dir()) / "metrics_snapshots"
    config = status.AgentConfig(repo_root=REPO_ROOT, snapshot_dir=snapshot_dir, operator_login="")
    raise SystemExit(run_upload(config, os.environ, args.reason, datetime.now(timezone.utc)))


if __name__ == "__main__":
    main()
```

Note: `upload()` is called with `run` as a positional argument from `run_upload`; `FakeRun` reads the temp file inside the call, before the `finally` deletes it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_agent -q -p no:cacheprovider`
Expected: PASS (all console_agent tests, including the 7 new ones)

- [ ] **Step 6: Pin the import footprint**

Add to `tests/console_agent/test_upload.py`:

```python
def test_uploader_does_not_import_trading_or_boto3():
    import subprocess as sp
    import sys

    code = "import src.console_agent.upload, sys; print(any(m.startswith(('src.trading', 'boto3')) for m in sys.modules))"
    result = sp.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=str(upload.REPO_ROOT))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
```

Run: `$PY -m pytest tests/console_agent/test_upload.py -q -p no:cacheprovider`
Expected: PASS (8 tests)

- [ ] **Step 7: Document and commit**

Append to `src/console_agent/README.md`:

```markdown
## Uploader (Phase 2a)

`python -m src.console_agent.upload --reason periodic|shutdown` builds the same document as `GET /status`
plus the full latest decision per strategy and copies it to `s3://$CONSOLE_SNAPSHOT_BUCKET/console/latest/status.json`
with the AWS CLI (the instance role grants `s3:PutObject` on that prefix only). It exits non-zero on any failure.
The units `homeguard-console-upload.timer` (every 5 minutes) and `homeguard-console-upload-shutdown.service`
(at shutdown) run it; install them with `infra/ec2/setup/install_console_upload.sh`.
```

```bash
git add src/console_agent/upload.py src/console_agent/README.md tests/console_agent/test_upload.py
git commit -m "feat(console-agent): snapshot uploader for the local console

Builds the /status document plus the latest decision per strategy and
copies it to S3 with the AWS CLI; exits non-zero on any failure.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 2: Uploader units and installer

**Files:**
- Create: `infra/ec2/services/homeguard-console-upload.service`, `infra/ec2/services/homeguard-console-upload.timer`, `infra/ec2/services/homeguard-console-upload-shutdown.service`, `infra/ec2/setup/install_console_upload.sh`
- Test: `tests/console_agent/test_upload_installer.py`

**Interfaces:**
- Consumes: `python -m src.console_agent.upload --reason periodic|shutdown` from Task 1.
- Produces: unit names used by the rollout (Task 10).

- [ ] **Step 1: Write the failing installer and unit tests**

`tests/console_agent/test_upload_installer.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_agent/test_upload_installer.py -q -p no:cacheprovider`
Expected: FAIL (installer and unit files do not exist; `KeyError: 'Unit'` / non-zero bash exit without the message)

- [ ] **Step 3: Write the units**

`infra/ec2/services/homeguard-console-upload.service`:

```ini
[Unit]
Description=Homeguard Console snapshot upload (status document to S3)
After=network-online.target homeguard-console-agent.service
Wants=network-online.target

[Service]
Type=oneshot
User=ec2-user
WorkingDirectory=/home/ec2-user/Homeguard
Environment="PATH=/home/ec2-user/Homeguard/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ec2-user/Homeguard/venv/bin/python -m src.console_agent.upload --reason periodic
TimeoutStartSec=60
MemoryMax=256M
OOMScoreAdjust=500
StandardOutput=journal
StandardError=journal
SyslogIdentifier=homeguard-console-upload
```

`infra/ec2/services/homeguard-console-upload.timer`:

```ini
[Unit]
Description=Upload the Homeguard Console snapshot every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

`infra/ec2/services/homeguard-console-upload-shutdown.service`:

```ini
[Unit]
Description=Homeguard Console snapshot upload at shutdown
Wants=network-online.target
# systemd stops units in reverse start order, so ExecStop runs while all of these are still up.
After=network-online.target homeguard-multi.service homeguard-cscm.service homeguard-gateway.service homeguard-console-agent.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=ec2-user
WorkingDirectory=/home/ec2-user/Homeguard
Environment="PATH=/home/ec2-user/Homeguard/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/bin/true
ExecStop=/home/ec2-user/Homeguard/venv/bin/python -m src.console_agent.upload --reason shutdown
TimeoutStopSec=60
MemoryMax=256M
OOMScoreAdjust=500
StandardOutput=journal
StandardError=journal
SyslogIdentifier=homeguard-console-upload

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 4: Write the installer**

`infra/ec2/setup/install_console_upload.sh`:

```bash
#!/bin/bash
# Idempotent installer for the Homeguard Console snapshot uploader (Phase 2a).
# Run ON the instance as ec2-user from the repo root:
#   bash infra/ec2/setup/install_console_upload.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ec2-user/Homeguard}"
UNITS=(homeguard-console-upload.service homeguard-console-upload.timer homeguard-console-upload-shutdown.service)

BUCKET=$(sed -n 's/^CONSOLE_SNAPSHOT_BUCKET=//p' "$REPO_DIR/.env" 2>/dev/null | tail -1 | tr -d "\"' \r" || true)
if [ -z "$BUCKET" ] || [[ "$BUCKET" == "<"*">" ]]; then
    echo "[-] CONSOLE_SNAPSHOT_BUCKET is not set to a real bucket in $REPO_DIR/.env"
    echo "    Add: CONSOLE_SNAPSHOT_BUCKET=\"<your bucket>\""
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "[-] The aws CLI is not installed; the uploader needs it"
    exit 1
fi
aws --version

echo "[+] Installing ${UNITS[*]}"
for unit in "${UNITS[@]}"; do
    sudo cp "$REPO_DIR/infra/ec2/services/$unit" "/etc/systemd/system/$unit"
done
sudo systemd-analyze verify "${UNITS[@]/#//etc/systemd/system/}"
sudo systemctl daemon-reload
sudo systemctl enable --now homeguard-console-upload-shutdown.service homeguard-console-upload.timer

echo "[+] Running one upload now"
sudo systemctl start homeguard-console-upload.service
aws s3 ls "s3://$BUCKET/console/latest/status.json"
systemctl list-timers homeguard-console-upload.timer --no-pager
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_agent/test_upload_installer.py -q -p no:cacheprovider`
Expected: PASS (6 tests; the 4 installer cases skip only if no bash is on PATH)

- [ ] **Step 6: Commit**

```bash
git add infra/ec2/services/homeguard-console-upload.service infra/ec2/services/homeguard-console-upload.timer infra/ec2/services/homeguard-console-upload-shutdown.service infra/ec2/setup/install_console_upload.sh tests/console_agent/test_upload_installer.py
git commit -m "feat(infra): uploader units for the console snapshot

A 5-minute timer and a shutdown unit ordered after the trading units,
so its ExecStop upload runs while they are still up.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 3: Terraform bucket, upload grant and read-only user

**Files:**
- Create: `infra/terraform/console.tf`
- Modify: `infra/terraform/variables.tf` (append), `infra/terraform/terraform.tfvars.example` (append), `infra/terraform/README.md`, `docs/INFRASTRUCTURE_OVERVIEW.md`

**Interfaces:**
- Consumes: `aws_iam_role.ec2_cloudwatch[0]` (monitoring.tf, `count = var.enable_cloudwatch_agent ? 1 : 0`), `var.aws_region`.
- Produces: bucket `var.console_snapshot_bucket`, IAM user `homeguard-console`.

- [ ] **Step 1: Write console.tf**

`infra/terraform/console.tf`:

```hcl
# Homeguard Console Phase 2a: snapshot bucket, the instance's upload grant,
# and a read-only IAM user for the local console app.
# Access keys for the user are created by the operator with the CLI so they
# never enter Terraform state.

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "console_snapshots" {
  bucket = var.console_snapshot_bucket

  tags = {
    Name = "homeguard-console-snapshots"
  }
}

resource "aws_s3_bucket_public_access_block" "console_snapshots" {
  bucket                  = aws_s3_bucket.console_snapshots.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "console_snapshots" {
  bucket = aws_s3_bucket.console_snapshots.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_iam_role_policy" "console_snapshot_upload" {
  count = var.enable_cloudwatch_agent ? 1 : 0

  name = "homeguard-console-snapshot-upload"
  role = aws_iam_role.ec2_cloudwatch[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "s3:PutObject"
      Resource = "${aws_s3_bucket.console_snapshots.arn}/console/latest/*"
    }]
  })
}

resource "aws_iam_user" "console" {
  name = "homeguard-console"

  tags = {
    Name = "homeguard-console"
  }
}

resource "aws_iam_user_policy" "console_read" {
  name = "homeguard-console-read"
  user = aws_iam_user.console.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "DescribeInstance"
        Effect   = "Allow"
        Action   = "ec2:DescribeInstances"
        Resource = "*"
      },
      {
        Sid      = "ReadSchedules"
        Effect   = "Allow"
        Action   = "scheduler:GetSchedule"
        Resource = "*"
      },
      {
        Sid      = "ReadSnapshot"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${aws_s3_bucket.console_snapshots.arn}/console/latest/*"
      },
      {
        # Without ListBucket a missing object reads as AccessDenied instead of NoSuchKey.
        Sid       = "ListSnapshotPrefix"
        Effect    = "Allow"
        Action    = "s3:ListBucket"
        Resource  = aws_s3_bucket.console_snapshots.arn
        Condition = { StringLike = { "s3:prefix" = ["console/latest/*"] } }
      },
      {
        Sid    = "ReadSchedulerLambdaLogs"
        Effect = "Allow"
        Action = "logs:FilterLogEvents"
        Resource = [
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/homeguard-start-instance:*",
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/homeguard-stop-instance:*",
        ]
      },
    ]
  })
}

output "console_snapshot_bucket" {
  value = aws_s3_bucket.console_snapshots.bucket
}
```

Append to `infra/terraform/variables.tf`:

```hcl
variable "console_snapshot_bucket" {
  description = "Private S3 bucket that holds the Homeguard Console snapshot (console/latest/status.json)"
  type        = string
}
```

Append to `infra/terraform/terraform.tfvars.example`:

```hcl
# Homeguard Console snapshot bucket (globally unique name)
console_snapshot_bucket = "<YOUR_BUCKET_NAME>"
```

- [ ] **Step 2: Validate**

```bash
terraform -chdir=infra/terraform init -backend=false -input=false > /dev/null
terraform -chdir=infra/terraform fmt -check console.tf variables.tf
terraform -chdir=infra/terraform validate
```

Expected: `fmt` prints nothing and exits 0; `validate` prints `Success! The configuration is valid.` If `fmt -check` lists a file, run `terraform -chdir=infra/terraform fmt console.tf variables.tf` and re-check. Do not run `plan` or `apply` here; that is an operator step in Task 10.

- [ ] **Step 3: Update the infrastructure docs**

Append to `docs/INFRASTRUCTURE_OVERVIEW.md` (under its AWS resources section; match the surrounding heading level):

```markdown
### Homeguard Console (Phase 2a)

- **S3 bucket** `var.console_snapshot_bucket` (private, SSE-S3, public access blocked) holds one object,
  `console/latest/status.json`, written by `homeguard-console-upload.timer` every 5 minutes and by
  `homeguard-console-upload-shutdown.service` at shutdown.
- **Instance role** `homeguard-ec2-cloudwatch` has `s3:PutObject` on `console/latest/*` only
  (inline policy `homeguard-console-snapshot-upload`).
- **IAM user** `homeguard-console` (read-only): `ec2:DescribeInstances`, `scheduler:GetSchedule`,
  `s3:GetObject` and `s3:ListBucket` on the prefix, `logs:FilterLogEvents` on the two scheduler Lambda
  log groups. Its access keys are created with the CLI, not Terraform, and live in the `homeguard-console`
  profile on the operator's machines. Defined in `infra/terraform/console.tf`.
```

Append the same three bullets, prefixed by `## Homeguard Console resources (console.tf)`, to `infra/terraform/README.md`, plus:

```markdown
Apply only these resources:

    terraform plan -target=aws_s3_bucket.console_snapshots -target=aws_s3_bucket_public_access_block.console_snapshots \
      -target=aws_s3_bucket_server_side_encryption_configuration.console_snapshots \
      -target=aws_iam_role_policy.console_snapshot_upload -target=aws_iam_user.console -target=aws_iam_user_policy.console_read \
      -var 'ssh_allowed_cidrs=["<YOUR_IP>/32"]'

Then create the keys: `aws iam create-access-key --user-name homeguard-console` and store them with
`aws configure --profile homeguard-console` on each machine.
```

- [ ] **Step 4: Commit**

```bash
git add infra/terraform/console.tf infra/terraform/variables.tf infra/terraform/terraform.tfvars.example infra/terraform/README.md docs/INFRASTRUCTURE_OVERVIEW.md
git commit -m "feat(terraform): console snapshot bucket and read-only console user

Private bucket for console/latest/status.json, PutObject for the
instance role on that prefix, and a read-only homeguard-console user
(ListBucket included so a missing snapshot reads as NoSuchKey).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 4: Local app package, settings and decision times

**Files:**
- Create: `tools/__init__.py` (empty), `tools/console/__init__.py`, `tools/console/config.py`, `tools/console/decision_times.py`
- Test: `tests/console_app/__init__.py` (empty), `tests/console_app/test_config.py`, `tests/console_app/test_decision_times.py`

**Interfaces:**
- Produces: `Settings(instance_id: str, region: str, agent_url: str, snapshot_bucket: str, aws_profile: str = "homeguard-console", port: int = 8765)`; `settings_from_env(environ: Mapping[str, str]) -> Settings` (raises `ValueError` naming every missing variable); `DECISION_TIMES: dict[str, tuple[str, ...]]`.

- [ ] **Step 1: Write the failing tests**

`tests/console_app/test_config.py`:

```python
import pytest

from tools.console.config import Settings, settings_from_env

FULL_ENV = {
    "EC2_INSTANCE_ID": "i-0123456789abcdef0",
    "EC2_REGION": "us-east-1",
    "CONSOLE_AGENT_URL": "https://agent.example.ts.net:8443",
    "CONSOLE_SNAPSHOT_BUCKET": "bucket",
}


def test_settings_come_from_the_environment():
    settings = settings_from_env(FULL_ENV)

    assert settings == Settings("i-0123456789abcdef0", "us-east-1", "https://agent.example.ts.net:8443", "bucket")
    assert settings.aws_profile == "homeguard-console"
    assert settings.port == 8765


def test_profile_can_be_overridden():
    assert settings_from_env({**FULL_ENV, "CONSOLE_AWS_PROFILE": "other"}).aws_profile == "other"


@pytest.mark.parametrize("value", ["", "  ", "<YOUR_BUCKET_NAME>"], ids=["empty", "blank", "placeholder"])
def test_missing_or_placeholder_values_are_all_named(value):
    env = {**FULL_ENV, "CONSOLE_SNAPSHOT_BUCKET": value}
    env.pop("CONSOLE_AGENT_URL")

    with pytest.raises(ValueError, match="CONSOLE_AGENT_URL, CONSOLE_SNAPSHOT_BUCKET"):
        settings_from_env(env)
```

`tests/console_app/test_decision_times.py`:

```python
"""The console's decision times must match the times the adapters hardcode."""

import re
from pathlib import Path

import pytest

from tools.console.decision_times import DECISION_TIMES

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = {
    "ramp": REPO_ROOT / "src" / "trading" / "adapters" / "ramp_live_adapter.py",
    "omr": REPO_ROOT / "src" / "trading" / "adapters" / "omr_live_adapter.py",
}
_BEGIN_DECISION = re.compile(r'_begin_decision\("scheduled_\w+", schedule_time="(\d\d:\d\d)"\)')


@pytest.mark.parametrize("strategy", sorted(ADAPTERS))
def test_decision_times_match_the_adapter(strategy):
    adapter_times = _BEGIN_DECISION.findall(ADAPTERS[strategy].read_text(encoding="utf-8"))

    assert adapter_times, f"no scheduled decisions found in {ADAPTERS[strategy].name}"
    assert sorted(adapter_times) == sorted(DECISION_TIMES[strategy])


def test_every_strategy_with_times_has_an_adapter_check():
    assert set(DECISION_TIMES) == set(ADAPTERS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_app -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.console'`

- [ ] **Step 3: Implement**

`tools/__init__.py`: empty file.

`tools/console/__init__.py`:

```python
"""Homeguard Console: a read-only local operations console. Run with python -m tools.console."""
```

`tools/console/config.py`:

```python
"""Console settings from the repo .env and the environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_REQUIRED = {
    "instance_id": "EC2_INSTANCE_ID",
    "region": "EC2_REGION",
    "agent_url": "CONSOLE_AGENT_URL",
    "snapshot_bucket": "CONSOLE_SNAPSHOT_BUCKET",
}


@dataclass(frozen=True)
class Settings:
    instance_id: str
    region: str
    agent_url: str
    snapshot_bucket: str
    aws_profile: str = "homeguard-console"
    port: int = 8765


def settings_from_env(environ: Mapping[str, str]) -> Settings:
    values = {name: environ.get(variable, "").strip() for name, variable in _REQUIRED.items()}
    missing = [_REQUIRED[name] for name, value in values.items() if not value or value.startswith("<")]
    if missing:
        raise ValueError(f"missing or placeholder values in .env: {', '.join(missing)}")
    return Settings(**values, aws_profile=environ.get("CONSOLE_AWS_PROFILE", "homeguard-console"))
```

`tools/console/decision_times.py`:

```python
"""Scheduled decision times (ET). The adapters hardcode these; test_decision_times pins them."""

DECISION_TIMES: dict[str, tuple[str, ...]] = {
    "ramp": ("15:55",),
    "omr": ("09:31", "15:50"),
}
```

`tests/console_app/__init__.py`: empty file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_app -q -p no:cacheprovider`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tools/console/__init__.py tools/console/config.py tools/console/decision_times.py tests/console_app/__init__.py tests/console_app/test_config.py tests/console_app/test_decision_times.py
git commit -m "feat(console): local app settings and decision-times table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 5: Schedule, expected state and power status

**Files:**
- Create: `tools/console/schedule.py`
- Test: `tests/console_app/test_schedule.py`

**Interfaces:**
- Produces:
  - `EASTERN: ZoneInfo`, `SCHEDULE_ACTIONS: dict[str, str]` (schedule name to `"start"`/`"stop"`)
  - `CronSchedule(name, action, minute, hour, weekdays: frozenset[int], zone: ZoneInfo)` (frozen dataclass; weekday 0 = Monday)
  - `parse_cron(name: str, action: str, expression: str, timezone: str) -> CronSchedule` (raises `ValueError`)
  - `last_fire(schedule, now) -> datetime`, `next_fire(schedule, now) -> datetime`
  - `expected_state(schedules: list[CronSchedule], now) -> str | None` (`"running"`, `"stopped"`, or `None` with no schedules)
  - `next_event(schedules, now) -> tuple[str, datetime] | None` (action, time)
  - `fires_on(schedule, day: date) -> list[datetime]` (ET), `instance_window(schedules, day) -> tuple[datetime, datetime] | None`
  - `nyse_session(day: date) -> tuple[datetime, datetime] | None` (ET), `power_lock(day) -> tuple[datetime, datetime] | None`
  - `PowerStatus(level: str, text: str, code: str)`; `power_status(actual: str | None, expected: str | None, agent_down_for: timedelta | None) -> PowerStatus` with codes `unknown`, `agent_unreachable`, `missed_start`, `off_schedule`, `as_scheduled`, `transition`

- [ ] **Step 1: Write the failing tests**

`tests/console_app/test_schedule.py`:

```python
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tools.console.schedule import (
    EASTERN, SCHEDULE_ACTIONS, expected_state, instance_window, next_event, nyse_session,
    parse_cron, power_lock, power_status,
)

UTC = ZoneInfo("UTC")
REAL = [
    ("homeguard-start-instance", "cron(0 8 ? * MON-FRI *)", "America/New_York"),
    ("homeguard-stop-instance", "cron(0 20 ? * MON-FRI *)", "America/New_York"),
    ("homeguard-start-instance-sunday", "cron(0 23 ? * SAT *)", "UTC"),
    ("homeguard-stop-instance-sunday", "cron(10 0 ? * SUN *)", "UTC"),
]
SCHEDULES = [parse_cron(name, SCHEDULE_ACTIONS[name], expr, zone) for name, expr, zone in REAL]


def et(*args):
    return datetime(*args, tzinfo=EASTERN)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_the_four_real_expressions_parse():
    assert [s.weekdays for s in SCHEDULES] == [frozenset(range(5)), frozenset(range(5)), frozenset({5}), frozenset({6})]
    assert (SCHEDULES[3].hour, SCHEDULES[3].minute) == (0, 10)


@pytest.mark.parametrize("expression", [
    "cron(0 8 * * MON-FRI *)", "cron(0/5 8 ? * MON-FRI *)", "cron(0 8 ? * 2-6 *)",
    "cron(0 8 ? * FRI-MON *)", "rate(5 minutes)", "cron(0 8 ? JAN MON *)",
])
def test_unsupported_expressions_raise(expression):
    with pytest.raises(ValueError):
        parse_cron("x", "start", expression, "UTC")


@pytest.mark.parametrize("now,expected", [
    (et(2026, 10, 8, 7, 59), "stopped"),
    (et(2026, 10, 8, 8, 0), "running"),
    (et(2026, 10, 8, 19, 59), "running"),
    (et(2026, 10, 8, 20, 0), "stopped"),
    (utc(2026, 10, 10, 23, 30), "running"),
    (utc(2026, 10, 11, 0, 10), "stopped"),
    (et(2026, 10, 11, 12, 0), "stopped"),
    (et(2026, 11, 26, 12, 0), "running"),
], ids=["weekday-0759", "weekday-0800", "weekday-1959", "weekday-2000", "sat-2330utc", "sun-0010utc",
        "sunday-noon", "thanksgiving"])
def test_expected_state(now, expected):
    assert expected_state(SCHEDULES, now) == expected


def test_expected_state_across_the_november_dst_change():
    assert expected_state(SCHEDULES, et(2026, 11, 2, 7, 59)) == "stopped"
    assert expected_state(SCHEDULES, et(2026, 11, 2, 8, 0)) == "running"
    assert instance_window(SCHEDULES, date(2026, 10, 31)) == (et(2026, 10, 31, 19, 0), et(2026, 10, 31, 20, 10))
    assert instance_window(SCHEDULES, date(2026, 11, 7)) == (et(2026, 11, 7, 18, 0), et(2026, 11, 7, 19, 10))


def test_no_schedules_means_unknown_expected_state():
    assert expected_state([], et(2026, 10, 8, 12, 0)) is None
    assert next_event([], et(2026, 10, 8, 12, 0)) is None


def test_next_event_after_the_evening_stop_is_the_next_morning_start():
    action, when = next_event(SCHEDULES, et(2026, 10, 8, 20, 30))

    assert action == "start"
    assert when == et(2026, 10, 9, 8, 0)


def test_weekday_instance_window():
    assert instance_window(SCHEDULES, date(2026, 10, 8)) == (et(2026, 10, 8, 8, 0), et(2026, 10, 8, 20, 0))
    assert instance_window(SCHEDULES, date(2026, 10, 11)) is None


def test_session_and_power_lock_on_regular_early_close_and_holiday_days():
    assert nyse_session(date(2026, 10, 8)) == (et(2026, 10, 8, 9, 30), et(2026, 10, 8, 16, 0))
    assert power_lock(date(2026, 10, 8)) == (et(2026, 10, 8, 9, 15), et(2026, 10, 8, 16, 15))
    assert power_lock(date(2026, 11, 27))[1] == et(2026, 11, 27, 13, 15)
    assert nyse_session(date(2026, 11, 26)) is None
    assert power_lock(date(2026, 11, 26)) is None


@pytest.mark.parametrize("actual,expected,down_for,code,level", [
    ("stopped", "stopped", None, "as_scheduled", "normal"),
    ("running", "running", None, "as_scheduled", "normal"),
    ("stopped", "running", None, "missed_start", "warning"),
    ("running", "stopped", None, "off_schedule", "caution"),
    ("running", "running", timedelta(minutes=3), "agent_unreachable", "warning"),
    ("running", "stopped", timedelta(minutes=3), "agent_unreachable", "warning"),
    ("running", "running", timedelta(minutes=1), "as_scheduled", "normal"),
    ("pending", "running", None, "transition", "caution"),
    (None, "running", None, "unknown", "unknown"),
    ("running", None, None, "unknown", "unknown"),
])
def test_power_status(actual, expected, down_for, code, level):
    status = power_status(actual, expected, down_for)

    assert (status.code, status.level) == (code, level)
    assert status.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_app/test_schedule.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.console.schedule'`

- [ ] **Step 3: Implement**

`tools/console/schedule.py`:

```python
"""Expected instance state from the EventBridge schedules, and the NYSE session.

Only the cron forms the Homeguard schedules use are supported (fixed minute and
hour, '?' day of month, '*' month and year, named days of week); anything else
raises instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

EASTERN = ZoneInfo("America/New_York")
POWER_LOCK_BUFFER = timedelta(minutes=15)
AGENT_UNREACHABLE_AFTER = timedelta(minutes=2)
SCHEDULE_ACTIONS = {
    "homeguard-start-instance": "start",
    "homeguard-stop-instance": "stop",
    "homeguard-start-instance-sunday": "start",
    "homeguard-stop-instance-sunday": "stop",
}
_DAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
_CRON = re.compile(r"^cron\((\d{1,2}) (\d{1,2}) \? \* ([A-Z,\-]+) \*\)$")


@dataclass(frozen=True)
class CronSchedule:
    name: str
    action: str
    minute: int
    hour: int
    weekdays: frozenset[int]
    zone: ZoneInfo


@dataclass(frozen=True)
class PowerStatus:
    level: str
    text: str
    code: str


def parse_cron(name: str, action: str, expression: str, timezone: str) -> CronSchedule:
    match = _CRON.match(expression)
    if match is None:
        raise ValueError(f"{name}: unsupported schedule expression {expression!r}")
    minute, hour, days = match.groups()
    return CronSchedule(name, action, int(minute), int(hour), _parse_weekdays(name, days), ZoneInfo(timezone))


def _parse_weekdays(name: str, field: str) -> frozenset[int]:
    weekdays: set[int] = set()
    for part in field.split(","):
        first, _, last = part.partition("-")
        last = last or first
        if first not in _DAYS or last not in _DAYS or _DAYS[last] < _DAYS[first]:
            raise ValueError(f"{name}: unsupported day-of-week field {field!r}")
        weekdays.update(range(_DAYS[first], _DAYS[last] + 1))
    return frozenset(weekdays)


def _fire_on_local_day(schedule: CronSchedule, local_day: date) -> datetime | None:
    if local_day.weekday() not in schedule.weekdays:
        return None
    return datetime.combine(local_day, time(schedule.hour, schedule.minute), schedule.zone)


def last_fire(schedule: CronSchedule, now: datetime) -> datetime:
    local_now = now.astimezone(schedule.zone)
    for days_back in range(8):
        fire = _fire_on_local_day(schedule, local_now.date() - timedelta(days=days_back))
        if fire is not None and fire <= local_now:
            return fire
    raise ValueError(f"{schedule.name} has no fire time in the last week")


def next_fire(schedule: CronSchedule, now: datetime) -> datetime:
    local_now = now.astimezone(schedule.zone)
    for days_ahead in range(8):
        fire = _fire_on_local_day(schedule, local_now.date() + timedelta(days=days_ahead))
        if fire is not None and fire > local_now:
            return fire
    raise ValueError(f"{schedule.name} has no fire time in the next week")


def expected_state(schedules: list[CronSchedule], now: datetime) -> str | None:
    if not schedules:
        return None
    latest = max(schedules, key=lambda schedule: last_fire(schedule, now))
    return "running" if latest.action == "start" else "stopped"


def next_event(schedules: list[CronSchedule], now: datetime) -> tuple[str, datetime] | None:
    if not schedules:
        return None
    upcoming = min(schedules, key=lambda schedule: next_fire(schedule, now))
    return upcoming.action, next_fire(upcoming, now).astimezone(EASTERN)


def fires_on(schedule: CronSchedule, day: date) -> list[datetime]:
    """Fire times of a schedule that fall on the given Eastern calendar day."""
    fires = []
    for offset in (-1, 0, 1):
        fire = _fire_on_local_day(schedule, day + timedelta(days=offset))
        if fire is not None and fire.astimezone(EASTERN).date() == day:
            fires.append(fire.astimezone(EASTERN))
    return fires


def instance_window(schedules: list[CronSchedule], day: date) -> tuple[datetime, datetime] | None:
    starts = [fire for s in schedules if s.action == "start" for fire in fires_on(s, day)]
    stops = [fire for s in schedules if s.action == "stop" for fire in fires_on(s, day)]
    if not starts or not stops:
        return None
    return min(starts), max(stops)


@lru_cache(maxsize=16)
def nyse_session(day: date) -> tuple[datetime, datetime] | None:
    schedule = mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day)
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    return row["market_open"].to_pydatetime().astimezone(EASTERN), row["market_close"].to_pydatetime().astimezone(EASTERN)


def power_lock(day: date) -> tuple[datetime, datetime] | None:
    session = nyse_session(day)
    if session is None:
        return None
    return session[0] - POWER_LOCK_BUFFER, session[1] + POWER_LOCK_BUFFER


def power_status(actual: str | None, expected: str | None, agent_down_for: timedelta | None) -> PowerStatus:
    if actual is None or expected is None:
        return PowerStatus("unknown", "Instance state unknown", "unknown")
    if actual == "running" and agent_down_for is not None and agent_down_for > AGENT_UNREACHABLE_AFTER:
        return PowerStatus("warning", "Instance is up but the console cannot reach the agent", "agent_unreachable")
    if actual == "stopped" and expected == "running":
        return PowerStatus("warning", "Instance should be running", "missed_start")
    if actual == "running" and expected == "stopped":
        return PowerStatus("caution", "Instance is running outside its schedule", "off_schedule")
    if actual == expected:
        return PowerStatus("normal", f"Instance {actual}, as scheduled", "as_scheduled")
    return PowerStatus("caution", f"Instance {actual} (expected {expected})", "transition")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_app/test_schedule.py -q -p no:cacheprovider`
Expected: PASS (all cases). If `nyse_session` returns UTC-aware values, the `astimezone(EASTERN)` makes the equality with `et(...)` hold.

- [ ] **Step 5: Commit**

```bash
git add tools/console/schedule.py tests/console_app/test_schedule.py
git commit -m "feat(console): expected instance state from the four schedules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 6: Freshness and the eight exception checks

**Files:**
- Create: `tools/console/freshness.py`, `tools/console/checks.py`
- Test: `tests/console_app/test_freshness.py`, `tests/console_app/test_checks.py`

**Interfaces:**
- Consumes: `EASTERN`, `nyse_session` (Task 5); `DECISION_TIMES` (Task 4); the status document shape from `src/console_agent/status.py` (`units`: list of dicts with `unit`, `active_state`, `sub_state`, `memory_bytes`; `strategies`: name to dict with `units`, `snapshot` (`gauges`/`counters`: metric to label-key to value), `last_decision` (`timestamp`, `all_passed`, ...)).
- Produces:
  - `freshness.INSTANCE_OWNED`, `MARKET_OWNED`, `MEASURED`; `classify(kind: str, live: bool) -> str` (`live`, `frozen`, `drifting`, `unknown`); `age_text(as_of: datetime, now: datetime) -> str`
  - `checks.Check(name: str, level: str, detail: str)`; `run_checks(document: dict, live: bool, as_of: datetime, now: datetime) -> list[Check]` (always 8, in the order IB Gateway, Broker heartbeat, Market data stream, Decisions on schedule, Order rejects, Drawdown, Host memory, Metrics scrape)

- [ ] **Step 1: Write the failing tests**

`tests/console_app/test_freshness.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from tools.console import freshness

NOW = datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("kind,live,expected", [
    (freshness.INSTANCE_OWNED, True, "live"),
    (freshness.MARKET_OWNED, True, "live"),
    (freshness.MEASURED, True, "live"),
    (freshness.INSTANCE_OWNED, False, "frozen"),
    (freshness.MARKET_OWNED, False, "drifting"),
    (freshness.MEASURED, False, "unknown"),
])
def test_classify(kind, live, expected):
    assert freshness.classify(kind, live) == expected


@pytest.mark.parametrize("delta,text", [
    (timedelta(seconds=4), "4 s ago"),
    (timedelta(minutes=5, seconds=10), "5 min ago"),
    (timedelta(hours=12, minutes=3), "12 h 3 min ago"),
])
def test_age_text(delta, text):
    assert freshness.age_text(NOW - delta, NOW) == text


def test_age_text_clamps_future_readings_to_zero():
    assert freshness.age_text(NOW + timedelta(seconds=3), NOW) == "0 s ago"
```

`tests/console_app/test_checks.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from tools.console.checks import run_checks
from tools.console.schedule import EASTERN

GIB = 1 << 30


def et(*args):
    return datetime(*args, tzinfo=EASTERN)


def document(*, gateway="active", heartbeat_age=10.0, websocket=None, rejected=0, drawdown=None,
             memory=GIB // 2, decided_at="2026-10-08T15:55:04-04:00", now=et(2026, 10, 8, 12, 0), ramp_units=True):
    gauges = {"hg_broker_last_heartbeat_timestamp": {"{}": now.timestamp() - heartbeat_age}}
    if websocket is not None:
        gauges["hg_websocket_connected"] = {"{}": websocket}
    if drawdown is not None:
        gauges["hg_portfolio_drawdown_pct"] = {"{}": drawdown}
    return {
        "units": [
            {"unit": "homeguard-gateway.service", "active_state": gateway, "sub_state": "running", "memory_bytes": None},
            {"unit": "homeguard-multi.service", "active_state": "active", "sub_state": "running", "memory_bytes": memory},
        ],
        "strategies": {
            "ramp": {
                "units": ["homeguard-multi.service"] if ramp_units else [],
                "snapshot": {"gauges": gauges, "counters": {"hg_orders_rejected_total": {'{"reason": "x"}': rejected}}},
                "last_decision": {"timestamp": decided_at, "all_passed": True} if decided_at else None,
            },
            "mp": {"units": [], "snapshot": None, "last_decision": None},
        },
    }


def levels(doc, now, live=True):
    return {check.name: check.level for check in run_checks(doc, live, now, now)}


def check(doc, now, name, live=True):
    return next(c for c in run_checks(doc, live, now, now) if c.name == name)


NOON = et(2026, 10, 8, 12, 0)


def test_healthy_document_is_all_normal_except_scrape():
    result = levels(document(), NOON)

    assert len(result) == 8
    assert result.pop("Metrics scrape") == "unknown"
    assert set(result.values()) == {"normal"}


def test_gateway_not_active_is_a_warning():
    assert levels(document(gateway="failed"), NOON)["IB Gateway"] == "warning"


def test_gateway_missing_from_units_is_a_warning():
    doc = document()
    doc["units"] = doc["units"][1:]
    assert levels(doc, NOON)["IB Gateway"] == "warning"


@pytest.mark.parametrize("age,level", [(119, "normal"), (121, "warning")])
def test_heartbeat_age_boundary(age, level):
    assert levels(document(heartbeat_age=age), NOON)["Broker heartbeat"] == level


def test_absent_heartbeat_reads_as_not_reported():
    doc = document()
    doc["strategies"]["ramp"]["snapshot"]["gauges"].pop("hg_broker_last_heartbeat_timestamp")
    result = check(doc, NOON, "Broker heartbeat")
    assert (result.level, result.detail) == ("normal", "Not reported")


def test_stream_disconnected_in_session_is_a_caution_and_after_close_is_normal():
    assert levels(document(websocket=0), NOON)["Market data stream"] == "caution"
    assert levels(document(websocket=0), et(2026, 10, 8, 17, 0))["Market data stream"] == "normal"
    assert check(document(), NOON, "Market data stream").detail == "Not reported"


@pytest.mark.parametrize("value,level", [(-9.9, "normal"), (-10.0, "caution"), (-19.9, "caution"), (-20.0, "warning")])
def test_drawdown_uses_negative_bounds(value, level):
    assert levels(document(drawdown=value), NOON)["Drawdown"] == level


def test_positive_drawdown_value_never_alerts():
    assert levels(document(drawdown=25.0), NOON)["Drawdown"] == "normal"


@pytest.mark.parametrize("fraction,level", [(0.80, "normal"), (0.81, "caution"), (0.95, "caution"), (0.96, "warning")])
def test_memory_boundaries(fraction, level):
    assert levels(document(memory=int(GIB * fraction)), NOON)["Host memory"] == level


def test_rejects_since_process_start_are_a_caution_with_the_count():
    result = check(document(rejected=3), NOON, "Order rejects")
    assert result.level == "caution"
    assert "3" in result.detail


def test_missed_rebalance_after_grace_is_a_warning():
    doc = document(decided_at="2026-10-07T15:55:04-04:00")
    assert levels(doc, et(2026, 10, 8, 15, 59))["Decisions on schedule"] == "normal"
    assert levels(doc, et(2026, 10, 8, 16, 0, 1))["Decisions on schedule"] == "warning"


def test_strategy_without_a_unit_is_not_expected_to_decide():
    doc = document(decided_at=None, ramp_units=False)
    assert levels(doc, et(2026, 10, 8, 17, 0))["Decisions on schedule"] == "normal"


def test_no_decision_expectation_on_a_holiday_or_after_an_early_close():
    doc = document(decided_at="2026-10-01T15:55:04-04:00")
    assert levels(doc, et(2026, 11, 26, 17, 0))["Decisions on schedule"] == "normal"
    assert levels(doc, et(2026, 11, 27, 17, 0))["Decisions on schedule"] == "normal"


def test_naive_decision_timestamp_is_read_as_eastern():
    doc = document(decided_at="2026-10-08T15:55:04")
    assert levels(doc, et(2026, 10, 8, 17, 0))["Decisions on schedule"] == "normal"


def test_unparseable_decision_timestamp_reads_as_missed():
    doc = document(decided_at="not a time")
    assert levels(doc, et(2026, 10, 8, 17, 0))["Decisions on schedule"] == "warning"


def test_measured_checks_become_unknown_on_a_snapshot_and_keep_the_last_reading():
    as_of = et(2026, 10, 8, 20, 0)
    results = {c.name: c for c in run_checks(document(gateway="failed"), False, as_of, as_of + timedelta(hours=3))}

    for name in ("IB Gateway", "Broker heartbeat", "Market data stream", "Host memory"):
        assert results[name].level == "unknown"
        assert results[name].detail.startswith("Unknown since 20:00")
    assert "failed" in results["IB Gateway"].detail
    assert results["Order rejects"].level == "normal"
    assert results["Decisions on schedule"].level == "normal"


def test_document_without_strategies_or_units_still_returns_eight_checks():
    assert len(run_checks({}, True, NOON, NOON)) == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_app/test_freshness.py tests/console_app/test_checks.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.console.freshness'`

- [ ] **Step 3: Implement freshness**

`tools/console/freshness.py`:

```python
"""Freshness classes from the parent spec: what a value means once its reading is old."""
from __future__ import annotations

from datetime import datetime

INSTANCE_OWNED = "instance"  # process state, decisions, switches: still true as of the reading
MARKET_OWNED = "market"      # equity, P&L, drawdown: drift once prices move
MEASURED = "measured"        # gateway, heartbeat, stream, memory: only valid while measured

_STALE_CLASS = {INSTANCE_OWNED: "frozen", MARKET_OWNED: "drifting", MEASURED: "unknown"}


def classify(kind: str, live: bool) -> str:
    return "live" if live else _STALE_CLASS[kind]


def age_text(as_of: datetime, now: datetime) -> str:
    seconds = max(int((now - as_of).total_seconds()), 0)
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min ago"
```

- [ ] **Step 4: Implement the checks**

`tools/console/checks.py`:

```python
"""The eight exception checks. Pure: a status document, liveness and the time in, levels out."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from tools.console import freshness
from tools.console.decision_times import DECISION_TIMES
from tools.console.schedule import EASTERN, nyse_session

HEARTBEAT_MAX_AGE = timedelta(seconds=120)
DECISION_GRACE = timedelta(minutes=5)
DRAWDOWN_CAUTION_PCT = -10.0
DRAWDOWN_WARNING_PCT = -20.0
TRADING_UNIT_MEMORY_CAP_BYTES = 1 << 30  # homeguard-multi MemoryMax=1G
MEMORY_CAUTION = 0.80
MEMORY_WARNING = 0.95
GATEWAY_UNIT = "homeguard-gateway.service"
TRADING_UNIT = "homeguard-multi.service"
MEASURED_CHECKS = {"IB Gateway", "Broker heartbeat", "Market data stream", "Host memory"}


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str


def run_checks(document: dict, live: bool, as_of: datetime, now: datetime) -> list[Check]:
    units = document.get("units") or []
    strategies = document.get("strategies") or {}
    checks = [
        gateway_check(units),
        heartbeat_check(strategies, now),
        stream_check(strategies, now),
        decisions_check(strategies, now),
        rejects_check(strategies),
        drawdown_check(strategies),
        memory_check(units),
        Check("Metrics scrape", "unknown", "Measured from Phase 2b"),
    ]
    if freshness.classify(freshness.MEASURED, live) != "unknown":
        return checks
    return [_unknown_since(check, as_of) if check.name in MEASURED_CHECKS else check for check in checks]


def _unknown_since(check: Check, as_of: datetime) -> Check:
    return Check(check.name, "unknown", f"Unknown since {as_of.astimezone(EASTERN):%H:%M}; last reading: {check.detail}")


def _metric_values(strategies: dict, kind: str, name: str) -> list[float]:
    values: list[float] = []
    for entry in strategies.values():
        snapshot = entry.get("snapshot") or {}
        values.extend((snapshot.get(kind) or {}).get(name, {}).values())
    return values


def _unit(units: list[dict], name: str) -> dict | None:
    return next((unit for unit in units if unit.get("unit") == name), None)


def gateway_check(units: list[dict]) -> Check:
    unit = _unit(units, GATEWAY_UNIT)
    if unit is None:
        return Check("IB Gateway", "warning", f"{GATEWAY_UNIT} not reported")
    if unit.get("active_state") != "active":
        return Check("IB Gateway", "warning", f"{unit.get('active_state')} ({unit.get('sub_state')})")
    return Check("IB Gateway", "normal", "Running")


def heartbeat_check(strategies: dict, now: datetime) -> Check:
    beats = _metric_values(strategies, "gauges", "hg_broker_last_heartbeat_timestamp")
    if not beats:
        return Check("Broker heartbeat", "normal", "Not reported")
    age = now - datetime.fromtimestamp(max(beats), tz=timezone.utc)
    seconds = int(age.total_seconds())
    if age > HEARTBEAT_MAX_AGE:
        return Check("Broker heartbeat", "warning", f"Last heartbeat {seconds} s ago")
    return Check("Broker heartbeat", "normal", f"{seconds} s ago")


def stream_check(strategies: dict, now: datetime) -> Check:
    connected = _metric_values(strategies, "gauges", "hg_websocket_connected")
    if not connected:
        return Check("Market data stream", "normal", "Not reported")
    if min(connected) > 0:
        return Check("Market data stream", "normal", "Connected")
    session = nyse_session(now.astimezone(EASTERN).date())
    if session is not None and session[0] <= now < session[1]:
        return Check("Market data stream", "caution", "Disconnected during the session")
    return Check("Market data stream", "normal", "Disconnected (market closed)")


def _decided_at(entry: dict) -> datetime | None:
    decision = entry.get("last_decision") or {}
    try:
        decided = datetime.fromisoformat(decision["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    return decided if decided.tzinfo is not None else decided.replace(tzinfo=EASTERN)


def decisions_check(strategies: dict, now: datetime) -> Check:
    local_now = now.astimezone(EASTERN)
    session = nyse_session(local_now.date())
    if session is None:
        return Check("Decisions on schedule", "normal", "No session today")
    missed = []
    for name, entry in sorted(strategies.items()):
        if not entry.get("units"):
            continue
        for slot in DECISION_TIMES.get(name, ()):
            due = datetime.combine(local_now.date(), time.fromisoformat(slot), EASTERN)
            if not session[0] <= due < session[1] or local_now < due + DECISION_GRACE:
                continue
            decided = _decided_at(entry)
            if decided is None or decided < due:
                missed.append(f"{name} {slot}")
    if missed:
        return Check("Decisions on schedule", "warning", f"Missed: {', '.join(missed)}")
    return Check("Decisions on schedule", "normal", "On schedule")


def rejects_check(strategies: dict) -> Check:
    rejected = sum(_metric_values(strategies, "counters", "hg_orders_rejected_total"))
    if rejected > 0:
        return Check("Order rejects", "caution", f"{int(rejected)} rejected since the process started")
    return Check("Order rejects", "normal", "None")


def drawdown_check(strategies: dict) -> Check:
    drawdowns = _metric_values(strategies, "gauges", "hg_portfolio_drawdown_pct")
    if not drawdowns:
        return Check("Drawdown", "normal", "Not reported")
    worst = min(drawdowns)
    if worst <= DRAWDOWN_WARNING_PCT:
        return Check("Drawdown", "warning", f"{worst:.1f}%")
    if worst <= DRAWDOWN_CAUTION_PCT:
        return Check("Drawdown", "caution", f"{worst:.1f}%")
    return Check("Drawdown", "normal", f"{worst:.1f}%")


def memory_check(units: list[dict]) -> Check:
    unit = _unit(units, TRADING_UNIT)
    if unit is None or unit.get("memory_bytes") is None:
        return Check("Host memory", "normal", "Not reported")
    fraction = unit["memory_bytes"] / TRADING_UNIT_MEMORY_CAP_BYTES
    detail = f"{TRADING_UNIT} at {fraction:.0%} of 1G"
    if fraction > MEMORY_WARNING:
        return Check("Host memory", "warning", detail)
    if fraction > MEMORY_CAUTION:
        return Check("Host memory", "caution", detail)
    return Check("Host memory", "normal", detail)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_app/test_freshness.py tests/console_app/test_checks.py -q -p no:cacheprovider`
Expected: PASS (all cases)

- [ ] **Step 6: Commit**

```bash
git add tools/console/freshness.py tools/console/checks.py tests/console_app/test_freshness.py tests/console_app/test_checks.py
git commit -m "feat(console): freshness classes and the eight exception checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 7: Poller and ConsoleState

**Files:**
- Create: `tools/console/poller.py`
- Test: `tests/console_app/test_poller.py`

**Interfaces:**
- Consumes: `Settings` (Task 4); `SCHEDULE_ACTIONS`, `parse_cron`, `CronSchedule` (Task 5); the S3 document from Task 1 (`uploaded_at`, `reason`, `status`, `decisions`).
- Produces:
  - `ConsoleState` dataclass: `document: dict | None`, `source: str | None` (`"agent"`/`"s3"`), `as_of: datetime | None`, `reason: str | None`, `decisions: dict[str, dict | None]`, `instance_state: str | None`, `schedules: list[CronSchedule]`, `errors: dict[str, str]`, `agent_down_since: datetime | None`; property `live -> bool`.
  - `AgentClient(base_url: str, transport: httpx.BaseTransport | None = None)` with `status() -> dict`, `decision(strategy: str) -> dict`.
  - `AwsClients(ec2, scheduler, s3)`; `make_aws_clients(settings) -> AwsClients`.
  - `Poller(settings, agent, aws, clock: Callable[[], datetime])` with `state: ConsoleState`, `poll_agent(now)`, `poll_s3(now)`, `poll_ec2(now)`, `poll_schedules(now)`, `tick()`, `async run()`.
  - `SNAPSHOT_KEY = "console/latest/status.json"`.

The spec's integration test names a local `ThreadingHTTPServer` as the fake agent; `httpx.MockTransport` exercises the same `AgentClient` code (status codes, JSON, timeouts) without a socket, so the tests use it.

- [ ] **Step 1: Write the failing tests**

`tests/console_app/test_poller.py`:

```python
import io
import json
from datetime import datetime, timedelta, timezone

import boto3
import httpx
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from tools.console.config import Settings
from tools.console.poller import SNAPSHOT_KEY, AgentClient, AwsClients, Poller

SETTINGS = Settings("i-0123456789abcdef0", "us-east-1", "https://agent.test:8443", "bucket")
NOW = datetime(2026, 10, 8, 16, 0, tzinfo=timezone.utc)
STATUS = {
    "generated_at": NOW.isoformat(),
    "units": [],
    "strategies": {"ramp": {"units": ["homeguard-multi.service"], "snapshot": None,
                            "last_decision": {"decision_id": "ramp-1", "timestamp": NOW.isoformat()}}},
    "execution_lock": {"state": "free"},
    "errors": [],
}
DECISION = {"decision_id": "ramp-1", "preconditions": {"all_passed": True}}


class FakeAgent:
    def __init__(self):
        self.mode = "ok"
        self.decision_calls = 0

    def handler(self, request):
        if self.mode == "timeout":
            raise httpx.ConnectTimeout("timed out", request=request)
        if self.mode == "403":
            return httpx.Response(403, json={"error": "forbidden"})
        if request.url.path == "/status":
            return httpx.Response(200, json=STATUS)
        self.decision_calls += 1
        return httpx.Response(200, json=DECISION)


def client(name):
    return boto3.client(name, region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")


@pytest.fixture
def aws():
    clients = AwsClients(client("ec2"), client("scheduler"), client("s3"))
    stubs = {name: Stubber(getattr(clients, name)) for name in ("ec2", "scheduler", "s3")}
    for stub in stubs.values():
        stub.activate()
    yield clients, stubs
    for stub in stubs.values():
        stub.deactivate()


def make_poller(aws_clients, agent=None):
    agent = agent or FakeAgent()
    agent_client = AgentClient(SETTINGS.agent_url, transport=httpx.MockTransport(agent.handler))
    return Poller(SETTINGS, agent_client, aws_clients, clock=lambda: NOW), agent


def s3_body(uploaded_at, reason="shutdown"):
    data = json.dumps({"uploaded_at": uploaded_at.isoformat(), "reason": reason,
                       "status": STATUS, "decisions": {"ramp": DECISION}}).encode()
    return {"Body": StreamingBody(io.BytesIO(data), len(data))}


def test_live_agent_reading_fetches_the_decision_once(aws):
    poller, agent = make_poller(aws[0])

    poller.poll_agent(NOW)
    poller.poll_agent(NOW + timedelta(seconds=10))

    assert poller.state.source == "agent"
    assert poller.state.live is True
    assert poller.state.decisions["ramp"] == DECISION
    assert agent.decision_calls == 1


def test_agent_timeout_falls_back_to_s3_and_labels_the_source(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    agent.mode = "timeout"
    uploaded = NOW - timedelta(minutes=3)
    stubs["s3"].add_response("get_object", s3_body(uploaded), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_agent(NOW)
    poller.poll_s3(NOW)

    assert poller.state.source == "s3"
    assert poller.state.live is False
    assert poller.state.as_of == uploaded
    assert poller.state.reason == "shutdown"
    assert poller.state.agent_down_since == NOW
    assert "agent" in poller.state.errors


def test_agent_403_falls_back_to_s3_and_reports_the_error(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    agent.mode = "403"
    stubs["s3"].add_response("get_object", s3_body(NOW - timedelta(minutes=1)), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_agent(NOW)
    poller.poll_s3(NOW)

    assert "403" in poller.state.errors["agent"]
    assert poller.state.source == "s3"


def test_agent_recovery_returns_to_live(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    agent.mode = "timeout"
    stubs["s3"].add_response("get_object", s3_body(NOW - timedelta(minutes=3)), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})
    poller.poll_agent(NOW)
    poller.poll_s3(NOW)

    agent.mode = "ok"
    poller.poll_agent(NOW + timedelta(seconds=10))

    assert poller.state.source == "agent"
    assert poller.state.live is True
    assert poller.state.agent_down_since is None
    assert "agent" not in poller.state.errors


def test_older_s3_snapshot_does_not_replace_a_newer_agent_reading(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    poller.poll_agent(NOW)
    agent.mode = "timeout"
    poller.poll_agent(NOW + timedelta(seconds=10))
    stubs["s3"].add_response("get_object", s3_body(NOW - timedelta(minutes=5)), {"Bucket": "bucket", "Key": SNAPSHOT_KEY})

    poller.poll_s3(NOW + timedelta(seconds=10))

    assert poller.state.source == "agent"
    assert poller.state.live is False


def test_missing_snapshot_reads_as_no_snapshot_yet(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    stubs["s3"].add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)

    poller.poll_s3(NOW)

    assert poller.state.errors["s3"] == "No snapshot yet"
    assert poller.state.document is None


def test_s3_access_denied_is_an_error_not_no_snapshot(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    stubs["s3"].add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)

    poller.poll_s3(NOW)

    assert "AccessDenied" in poller.state.errors["s3"]


def test_ec2_state_and_failure(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    stubs["ec2"].add_response(
        "describe_instances",
        {"Reservations": [{"Instances": [{"InstanceId": SETTINGS.instance_id, "State": {"Code": 80, "Name": "stopped"}}]}]},
        {"InstanceIds": [SETTINGS.instance_id]},
    )
    stubs["ec2"].add_client_error("describe_instances", service_error_code="UnauthorizedOperation")

    poller.poll_ec2(NOW)
    assert poller.state.instance_state == "stopped"

    poller.poll_ec2(NOW)
    assert poller.state.instance_state is None
    assert "UnauthorizedOperation" in poller.state.errors["ec2"]


def add_schedules(stub, disabled=()):
    real = {
        "homeguard-start-instance": ("cron(0 8 ? * MON-FRI *)", "America/New_York"),
        "homeguard-stop-instance": ("cron(0 20 ? * MON-FRI *)", "America/New_York"),
        "homeguard-start-instance-sunday": ("cron(0 23 ? * SAT *)", "UTC"),
        "homeguard-stop-instance-sunday": ("cron(10 0 ? * SUN *)", "UTC"),
    }
    for name, (expression, zone) in real.items():
        stub.add_response(
            "get_schedule",
            {"Name": name, "ScheduleExpression": expression, "ScheduleExpressionTimezone": zone,
             "State": "DISABLED" if name in disabled else "ENABLED"},
            {"Name": name},
        )


def test_schedules_are_read_and_disabled_ones_skipped(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    add_schedules(stubs["scheduler"], disabled={"homeguard-start-instance-sunday"})

    poller.poll_schedules(NOW)

    assert [s.name for s in poller.state.schedules] == [
        "homeguard-start-instance", "homeguard-stop-instance", "homeguard-stop-instance-sunday"]


def test_a_failed_schedule_read_keeps_the_previous_schedules(aws):
    clients, stubs = aws
    poller, _ = make_poller(clients)
    add_schedules(stubs["scheduler"])
    poller.poll_schedules(NOW)
    stubs["scheduler"].add_client_error("get_schedule", service_error_code="AccessDeniedException")

    poller.poll_schedules(NOW + timedelta(minutes=10))

    assert len(poller.state.schedules) == 4
    assert "schedule:homeguard-start-instance" in poller.state.errors


def test_tick_polls_s3_only_while_the_agent_is_down(aws):
    clients, stubs = aws
    poller, agent = make_poller(clients)
    stubs["ec2"].add_response(
        "describe_instances",
        {"Reservations": [{"Instances": [{"InstanceId": SETTINGS.instance_id, "State": {"Code": 16, "Name": "running"}}]}]},
        {"InstanceIds": [SETTINGS.instance_id]},
    )
    add_schedules(stubs["scheduler"])

    poller.tick()

    # A poll_s3 call would hit the unstubbed S3 client and record an "s3" error.
    assert "s3" not in poller.state.errors
    stubs["ec2"].assert_no_pending_responses()
    stubs["scheduler"].assert_no_pending_responses()
    assert poller.state.instance_state == "running"
    assert len(poller.state.schedules) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_app/test_poller.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.console.poller'`

- [ ] **Step 3: Implement**

`tools/console/poller.py`:

```python
"""Keeps one ConsoleState current from the agent, S3, EC2 and the Scheduler.

Each source fails on its own: a failure is logged once, recorded in
state.errors, and the other sources keep updating.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import boto3
import httpx
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.logger import logger
from tools.console.config import Settings
from tools.console.schedule import SCHEDULE_ACTIONS, CronSchedule, parse_cron

SNAPSHOT_KEY = "console/latest/status.json"
AGENT_TIMEOUT_SECONDS = 5.0
AGENT_INTERVAL = timedelta(seconds=10)
S3_INTERVAL = timedelta(seconds=60)
EC2_INTERVAL = timedelta(seconds=30)
SCHEDULE_INTERVAL = timedelta(minutes=10)
AWS_ERRORS = (BotoCoreError, ClientError)
AGENT_ERRORS = (httpx.HTTPError, ValueError)


@dataclass
class ConsoleState:
    document: dict | None = None
    source: str | None = None
    as_of: datetime | None = None
    reason: str | None = None
    decisions: dict = field(default_factory=dict)
    instance_state: str | None = None
    schedules: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    agent_down_since: datetime | None = None

    @property
    def live(self) -> bool:
        return self.source == "agent" and self.agent_down_since is None


class AgentClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url, timeout=AGENT_TIMEOUT_SECONDS, transport=transport)

    def status(self) -> dict:
        response = self._http.get("/status")
        response.raise_for_status()
        return response.json()

    def decision(self, strategy: str) -> dict:
        response = self._http.get("/decisions", params={"strategy": strategy})
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class AwsClients:
    ec2: object
    scheduler: object
    s3: object


def make_aws_clients(settings: Settings) -> AwsClients:
    session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.region)
    config = BotoConfig(retries={"mode": "standard", "max_attempts": 3}, connect_timeout=5, read_timeout=10)
    return AwsClients(*(session.client(name, config=config) for name in ("ec2", "scheduler", "s3")))


def _decision_id(document: dict | None, strategy: str) -> str | None:
    entry = ((document or {}).get("strategies") or {}).get(strategy) or {}
    return (entry.get("last_decision") or {}).get("decision_id")


class Poller:
    def __init__(self, settings: Settings, agent: AgentClient, aws: AwsClients, clock: Callable[[], datetime]):
        self.settings = settings
        self.agent = agent
        self.aws = aws
        self.clock = clock
        self.state = ConsoleState()
        self._last_run: dict[str, datetime] = {}

    def _fail(self, source: str, error: object) -> None:
        if source not in self.state.errors:
            logger.warning(f"[console] {source} poll failed: {error!r}")
        self.state.errors[source] = repr(error)

    def poll_agent(self, now: datetime) -> None:
        try:
            document = self.agent.status()
        except AGENT_ERRORS as e:
            self._fail("agent", e)
            self.state.agent_down_since = self.state.agent_down_since or now
            return
        previous = self.state.document if self.state.source == "agent" else None
        self._refresh_decisions(document, previous)
        self.state.document, self.state.source, self.state.as_of, self.state.reason = document, "agent", now, None
        self.state.agent_down_since = None
        self.state.errors.pop("agent", None)

    def _refresh_decisions(self, document: dict, previous: dict | None) -> None:
        for name in (document.get("strategies") or {}):
            latest = _decision_id(document, name)
            if latest is None:
                self.state.decisions[name] = None
                continue
            if latest == _decision_id(previous, name) and self.state.decisions.get(name) is not None:
                continue
            try:
                self.state.decisions[name] = self.agent.decision(name)
                self.state.errors.pop(f"decision:{name}", None)
            except AGENT_ERRORS as e:
                self._fail(f"decision:{name}", e)

    def poll_s3(self, now: datetime) -> None:
        try:
            body = self.aws.s3.get_object(Bucket=self.settings.snapshot_bucket, Key=SNAPSHOT_KEY)["Body"].read()
            snapshot = json.loads(body)
            uploaded_at = datetime.fromisoformat(snapshot["uploaded_at"])
            document = snapshot["status"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                self.state.errors["s3"] = "No snapshot yet"
            else:
                self._fail("s3", e)
            return
        except (BotoCoreError, ValueError, KeyError, TypeError) as e:
            self._fail("s3", e)
            return
        self.state.errors.pop("s3", None)
        if self.state.as_of is not None and self.state.as_of >= uploaded_at:
            return
        self.state.document, self.state.source, self.state.as_of = document, "s3", uploaded_at
        self.state.reason = snapshot.get("reason")
        self.state.decisions = snapshot.get("decisions") or {}

    def poll_ec2(self, now: datetime) -> None:
        try:
            reservations = self.aws.ec2.describe_instances(InstanceIds=[self.settings.instance_id])["Reservations"]
            self.state.instance_state = reservations[0]["Instances"][0]["State"]["Name"]
        except AWS_ERRORS + (IndexError, KeyError) as e:
            self._fail("ec2", e)
            self.state.instance_state = None
            return
        self.state.errors.pop("ec2", None)

    def poll_schedules(self, now: datetime) -> None:
        schedules: list[CronSchedule] = []
        failed = False
        for name, action in SCHEDULE_ACTIONS.items():
            try:
                entry = self.aws.scheduler.get_schedule(Name=name)
                if entry["State"] == "ENABLED":
                    schedules.append(parse_cron(name, action, entry["ScheduleExpression"],
                                                entry.get("ScheduleExpressionTimezone", "UTC")))
                self.state.errors.pop(f"schedule:{name}", None)
            except AWS_ERRORS + (KeyError, ValueError) as e:
                self._fail(f"schedule:{name}", e)
                failed = True
        # A partial list would compute the wrong expected state, so keep the last full one.
        if not failed:
            self.state.schedules = schedules

    def _due(self, name: str, interval: timedelta, now: datetime) -> bool:
        last = self._last_run.get(name)
        if last is not None and now - last < interval:
            return False
        self._last_run[name] = now
        return True

    def tick(self) -> None:
        now = self.clock()
        self.poll_agent(now)
        if self.state.agent_down_since is not None and self._due("s3", S3_INTERVAL, now):
            self.poll_s3(now)
        if self._due("ec2", EC2_INTERVAL, now):
            self.poll_ec2(now)
        if self._due("schedules", SCHEDULE_INTERVAL, now):
            self.poll_schedules(now)

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
                self.state.errors.pop("poller", None)
            except Exception as e:
                # Keep the console up and say so; one bad tick must not stop every panel.
                logger.error(f"[console] poller tick failed: {e!r}")
                self.state.errors["poller"] = repr(e)
            await asyncio.sleep(AGENT_INTERVAL.total_seconds())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_app/test_poller.py -q -p no:cacheprovider`
Expected: PASS (all tests). If Stubber rejects a response shape (`ParamValidationError` on an output field), remove only that field from the stub response, never from the poller.

- [ ] **Step 5: Commit**

```bash
git add tools/console/poller.py tests/console_app/test_poller.py
git commit -m "feat(console): poller with agent, S3, EC2 and Scheduler sources

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 8: Views, templates and the FastAPI app

**Files:**
- Create: `tools/console/views.py`, `tools/console/app.py`, `tools/console/__main__.py`, `tools/console/templates/index.html`, `tools/console/templates/panels/{header,schedule,exceptions,strategies,account,host,gates}.html`, `tools/console/static/console.css`, `tools/console/static/htmx.min.js`
- Test: `tests/console_app/test_views.py`, `tests/console_app/test_app.py`

**Interfaces:**
- Consumes: `ConsoleState` (Task 7); `run_checks` (Task 6); `freshness` (Task 6); schedule functions and `PowerStatus.code` (Task 5); `DECISION_TIMES` (Task 4); `src.console_agent.status.summarize_decision`, `READ_ERRORS`.
- Produces: `views.PANELS`, `views.page_context(state, now, region) -> dict`, `views.gates_context(state, strategy) -> dict | None`; `app.create_app(poller, region: str, clock, start_polling: bool = True) -> FastAPI`.

- [ ] **Step 1: Vendor htmx**

```bash
curl -sSfL https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js -o tools/console/static/htmx.min.js
head -c 60 tools/console/static/htmx.min.js
```

Expected: a minified JavaScript header (starts with `var htmx=` or `(function`), file about 50 KB.

- [ ] **Step 2: Write the failing view tests**

`tests/console_app/test_views.py`:

```python
from datetime import datetime, timedelta, timezone

from tools.console.poller import ConsoleState
from tools.console.schedule import EASTERN, SCHEDULE_ACTIONS, parse_cron
from tools.console.views import gates_context, page_context

NOW = datetime(2026, 10, 8, 16, 30, tzinfo=timezone.utc)  # 12:30 ET
SCHEDULES = [
    parse_cron("homeguard-start-instance", "start", "cron(0 8 ? * MON-FRI *)", "America/New_York"),
    parse_cron("homeguard-stop-instance", "stop", "cron(0 20 ? * MON-FRI *)", "America/New_York"),
]
DOCUMENT = {
    "units": [{"unit": "homeguard-multi.service", "active_state": "active", "sub_state": "running",
               "restarts": 0, "memory_bytes": 1 << 29, "active_since": "Thu 2026-10-08 08:01:12 EDT"}],
    "strategies": {
        "ramp": {"enabled": False, "variant": "v11", "units": ["homeguard-multi.service"],
                 "snapshot": {"gauges": {"hg_portfolio_equity_usd": {"{}": 101234.0},
                                         "hg_strategy_positions_count": {'{"strategy": "ramp"}': 26.0}}},
                 "last_decision": {"timestamp": "2026-10-07T15:55:04-04:00", "all_passed": False}},
        "mp": {"enabled": True, "variant": "v01", "units": [], "snapshot": None, "last_decision": None},
    },
    "errors": [{"source": "snapshot:cscm", "error": "FileNotFoundError()"}],
}
RECORD = {"decision_id": "ramp-1", "timestamp": "2026-10-07T15:55:04-04:00",
          "trigger": {"kind": "scheduled_rebalance", "schedule_time": "15:55"},
          "preconditions": {"all_passed": False, "strategy_enabled": {"passed": False, "error": "disabled"}, "extra": {}}}


def live_state(**changes):
    state = ConsoleState(document=DOCUMENT, source="agent", as_of=NOW - timedelta(seconds=4),
                         instance_state="running", schedules=SCHEDULES, decisions={"ramp": RECORD})
    for name, value in changes.items():
        setattr(state, name, value)
    return state


def test_live_header_badge_and_error_strip():
    header = page_context(live_state(), NOW, "us-east-1")["header"]

    assert header["badge"] == "Live from agent, 4 s ago"
    assert header["level"] == "normal"
    assert "snapshot:cscm: FileNotFoundError()" in header["errors"]


def test_s3_header_badge_names_the_upload_and_reason():
    uploaded = datetime(2026, 10, 8, 0, 0, 12, tzinfo=timezone.utc)
    state = live_state(source="s3", as_of=uploaded, reason="shutdown", agent_down_since=NOW)

    header = page_context(state, NOW, "us-east-1")["header"]

    assert header["badge"].startswith("Snapshot from S3, taken 2026-10-07 20:00:12 ET (shutdown)")
    assert header["level"] == "caution"


def test_no_document_shows_no_status_and_no_checks():
    context = page_context(ConsoleState(), NOW, "us-east-1")

    assert context["header"]["badge"] == "No status yet"
    assert context["checks"] is None


def test_rail_has_instance_lock_and_session_bands_and_a_ramp_tick():
    rail = page_context(live_state(), NOW, "us-east-1")["rail"]

    assert [band["cls"] for band in rail["bands"]] == ["instance", "lock", "session"]
    assert [tick["label"] for tick in rail["ticks"]] == ["ramp 15:55"]
    assert rail["now_x"] is not None


def test_missed_start_power_tile_links_the_start_lambda_log():
    state = live_state(instance_state="stopped", source="s3", agent_down_since=NOW)

    power = page_context(state, NOW, "us-east-1")["power"]

    assert power["level"] == "warning"
    assert "homeguard-start-instance" in power["log_url"]


def test_strategy_rows_flag_a_switch_on_with_no_unit():
    rows = {row["name"]: row for row in page_context(live_state(), NOW, "us-east-1")["strategies"]}

    assert rows["mp"]["caution"] is True
    assert rows["mp"]["process"] == "no unit"
    assert rows["ramp"]["caution"] is False
    assert rows["ramp"]["decided_at"] == "10-07 15:55"


def test_account_values_drift_on_the_snapshot_and_positions_freeze():
    state = live_state(source="s3", agent_down_since=NOW, as_of=NOW - timedelta(hours=1))

    rows = page_context(state, NOW, "us-east-1")["account"]

    cells = {cell["label"]: cell for cell in rows[0]["cells"]}
    assert cells["Equity"]["value"] == "$101,234"
    assert cells["Equity"]["cls"] == "drifting"
    assert cells["Positions"]["cls"] == "frozen"
    assert rows[0]["age"] == "1 h 0 min ago"


def test_gates_context_summarizes_the_full_record():
    context = gates_context(live_state(), "ramp")

    assert context["gates"]["strategy_enabled"] == {"passed": False, "error": "disabled"}
    assert gates_context(live_state(), "nope") is None
```

- [ ] **Step 3: Run the view tests to verify they fail**

Run: `$PY -m pytest tests/console_app/test_views.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.console.views'`

- [ ] **Step 4: Implement views**

`tools/console/views.py`:

```python
"""Template contexts from the ConsoleState. Pure: state and time in, plain dicts out."""
from __future__ import annotations

from datetime import date, datetime, time

from src.console_agent.status import READ_ERRORS, summarize_decision
from tools.console import freshness
from tools.console.checks import run_checks
from tools.console.decision_times import DECISION_TIMES
from tools.console.poller import ConsoleState
from tools.console.schedule import EASTERN, expected_state, instance_window, next_event, nyse_session, power_lock, power_status

PANELS = ("header", "schedule", "exceptions", "strategies", "account", "host")
RAIL_START_HOUR = 6
RAIL_END_HOUR = 22
RAIL_WIDTH = 1000
START_LAMBDA_LOG_GROUP = "/aws/lambda/homeguard-start-instance"
ACCOUNT_GAUGES = (
    ("Equity", "hg_portfolio_equity_usd", "${:,.0f}", freshness.MARKET_OWNED),
    ("Day P&L", "hg_portfolio_day_pnl_usd", "${:+,.0f}", freshness.MARKET_OWNED),
    ("Drawdown", "hg_portfolio_drawdown_pct", "{:.1f}%", freshness.MARKET_OWNED),
    ("Positions", "hg_strategy_positions_count", "{:.0f}", freshness.INSTANCE_OWNED),
)


def page_context(state: ConsoleState, now: datetime, region: str) -> dict:
    document = state.document or {}
    return {
        "header": header_context(state, now),
        "rail": rail_context(state, now),
        "power": power_context(state, now, region),
        "checks": run_checks(document, state.live, state.as_of, now) if state.document else None,
        "strategies": strategy_rows(state),
        "account": account_rows(state, now),
        "units": document.get("units") or [],
        "live": state.live,
    }


def header_context(state: ConsoleState, now: datetime) -> dict:
    errors = [f"{source}: {error}" for source, error in sorted(state.errors.items())]
    errors += [f"{e.get('source')}: {e.get('error')}" for e in (state.document or {}).get("errors") or []]
    if state.document is None:
        return {"badge": "No status yet", "level": "unknown", "errors": errors}
    age = freshness.age_text(state.as_of, now)
    if state.live:
        return {"badge": f"Live from agent, {age}", "level": "normal", "errors": errors}
    if state.source == "s3":
        taken = state.as_of.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")
        reason = f" ({state.reason})" if state.reason else ""
        return {"badge": f"Snapshot from S3, taken {taken}{reason}, {age}", "level": "caution", "errors": errors}
    return {"badge": f"Agent unreachable; last agent reading {age}", "level": "caution", "errors": errors}


def _rail_x(moment: datetime, day: date) -> float:
    start = datetime.combine(day, time(RAIL_START_HOUR), EASTERN)
    end = datetime.combine(day, time(RAIL_END_HOUR), EASTERN)
    fraction = (moment - start) / (end - start)
    return round(RAIL_WIDTH * min(max(fraction, 0.0), 1.0), 1)


def _band(cls: str, label: str, span: tuple[datetime, datetime], y: int, day: date) -> dict:
    x = _rail_x(span[0], day)
    return {"cls": cls, "label": f"{label} {span[0]:%H:%M}-{span[1]:%H:%M}", "x": x,
            "w": round(_rail_x(span[1], day) - x, 1), "y": y, "h": 10}


def rail_context(state: ConsoleState, now: datetime) -> dict:
    local_now = now.astimezone(EASTERN)
    day = local_now.date()
    bands = []
    for cls, label, span, y in (
        ("instance", "Instance window", instance_window(state.schedules, day), 12),
        ("lock", "Power lock", power_lock(day), 26),
        ("session", "NYSE session", nyse_session(day), 40),
    ):
        if span is not None:
            bands.append(_band(cls, label, span, y, day))
    ticks = []
    strategies = (state.document or {}).get("strategies") or {}
    if nyse_session(day) is not None:
        for name, entry in sorted(strategies.items()):
            if entry.get("units"):
                for slot in DECISION_TIMES.get(name, ()):
                    moment = datetime.combine(day, time.fromisoformat(slot), EASTERN)
                    ticks.append({"x": _rail_x(moment, day), "label": f"{name} {slot}"})
    hours = [{"x": _rail_x(datetime.combine(day, time(hour), EASTERN), day), "label": f"{hour:02d}"}
             for hour in range(RAIL_START_HOUR, RAIL_END_HOUR + 1, 2)]
    rail_start = datetime.combine(day, time(RAIL_START_HOUR), EASTERN)
    rail_end = datetime.combine(day, time(RAIL_END_HOUR), EASTERN)
    now_x = _rail_x(local_now, day) if rail_start <= local_now <= rail_end else None
    return {"bands": bands, "ticks": ticks, "hours": hours, "now_x": now_x}


def power_context(state: ConsoleState, now: datetime, region: str) -> dict:
    down_for = now - state.agent_down_since if state.agent_down_since else None
    status = power_status(state.instance_state, expected_state(state.schedules, now), down_for)
    upcoming = next_event(state.schedules, now)
    log_url = None
    if status.code == "missed_start":
        group = START_LAMBDA_LOG_GROUP.replace("/", "$252F")
        log_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/{group}"
    return {
        "level": status.level,
        "text": status.text,
        "next": f"Next scheduled {upcoming[0]}: {upcoming[1]:%a %H:%M} ET" if upcoming else None,
        "log_url": log_url,
    }


def _short_time(timestamp: str | None) -> str:
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return "-"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=EASTERN)
    return moment.astimezone(EASTERN).strftime("%m-%d %H:%M")


def strategy_rows(state: ConsoleState) -> list[dict]:
    rows = []
    for name, entry in sorted(((state.document or {}).get("strategies") or {}).items()):
        decision = entry.get("last_decision") or {}
        rows.append({
            "name": name,
            "enabled": entry.get("enabled"),
            "process": ", ".join(entry.get("units") or []) or "no unit",
            "variant": entry.get("variant"),
            "decided_at": _short_time(decision.get("timestamp")),
            "passed": decision.get("all_passed"),
            "caution": bool(entry.get("enabled")) and not entry.get("units"),
            "has_gates": state.decisions.get(name) is not None,
        })
    return rows


def _first_value(values: dict | None, template: str) -> str:
    if not values:
        return "-"
    return template.format(next(iter(values.values())))


def account_rows(state: ConsoleState, now: datetime) -> list[dict]:
    age = "" if state.live or state.as_of is None else freshness.age_text(state.as_of, now)
    rows = []
    for name, entry in sorted(((state.document or {}).get("strategies") or {}).items()):
        gauges = (entry.get("snapshot") or {}).get("gauges") or {}
        if not gauges:
            continue
        cells = [{"label": label, "value": _first_value(gauges.get(metric), template),
                  "cls": freshness.classify(kind, state.live)}
                 for label, metric, template, kind in ACCOUNT_GAUGES]
        rows.append({"name": name, "cells": cells, "age": age})
    return rows


def gates_context(state: ConsoleState, strategy: str) -> dict | None:
    record = state.decisions.get(strategy)
    if record is None:
        return None
    try:
        summary = summarize_decision(record)
    except READ_ERRORS as e:
        return {"strategy": strategy, "summary": None, "gates": {}, "error": repr(e)}
    return {"strategy": strategy, "summary": summary, "gates": summary["gates"], "error": None}
```

- [ ] **Step 5: Run the view tests to verify they pass**

Run: `$PY -m pytest tests/console_app/test_views.py -q -p no:cacheprovider`
Expected: PASS (8 tests)

- [ ] **Step 6: Write the templates and CSS**

`tools/console/templates/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Homeguard Console</title>
<link rel="stylesheet" href="/static/console.css">
<script src="/static/htmx.min.js"></script>
</head>
<body>
<header id="header" hx-get="/panels/header" hx-trigger="every 10s">{% include "panels/header.html" %}</header>
<main>
  <section class="day" aria-labelledby="day-h" hx-get="/panels/schedule" hx-trigger="every 10s">{% include "panels/schedule.html" %}</section>
  <section aria-labelledby="annun-h" hx-get="/panels/exceptions" hx-trigger="every 10s">{% include "panels/exceptions.html" %}</section>
  <section aria-labelledby="strats-h" hx-get="/panels/strategies" hx-trigger="every 10s">{% include "panels/strategies.html" %}</section>
  <section id="gates" aria-live="polite"></section>
  <section aria-labelledby="pf-h" hx-get="/panels/account" hx-trigger="every 10s">{% include "panels/account.html" %}</section>
  <section aria-labelledby="host-h" hx-get="/panels/host" hx-trigger="every 10s">{% include "panels/host.html" %}</section>
</main>
</body>
</html>
```

`tools/console/templates/panels/header.html`:

```html
<h1>Homeguard</h1>
<p class="badge {{ header.level }}"><span class="level">{{ header.level|upper }}</span> {{ header.badge }}</p>
{% if header.errors %}<ul class="errors">{% for error in header.errors %}<li>{{ error }}</li>{% endfor %}</ul>{% endif %}
```

`tools/console/templates/panels/schedule.html`:

```html
<h2 id="day-h">Today's schedule</h2>
<div class="rail-row">
  <svg class="rail" viewBox="0 0 1000 64" role="img" aria-label="Instance window, power lock, NYSE session and decisions for today">
    {% for hour in rail.hours %}<text class="hour" x="{{ hour.x }}" y="8">{{ hour.label }}</text>{% endfor %}
    {% for band in rail.bands %}<rect class="band {{ band.cls }}" x="{{ band.x }}" y="{{ band.y }}" width="{{ band.w }}" height="{{ band.h }}"><title>{{ band.label }}</title></rect>{% endfor %}
    {% for tick in rail.ticks %}<line class="tick" x1="{{ tick.x }}" x2="{{ tick.x }}" y1="10" y2="52"/><text class="tick-label" x="{{ tick.x }}" y="62">{{ tick.label }}</text>{% endfor %}
    {% if rail.now_x is not none %}<line class="now" x1="{{ rail.now_x }}" x2="{{ rail.now_x }}" y1="0" y2="64"/>{% endif %}
  </svg>
  <div class="tile {{ power.level }}">
    <strong>Power</strong><span class="level">{{ power.level|upper }}</span>
    <span>{{ power.text }}</span>
    {% if power.next %}<small>{{ power.next }}</small>{% endif %}
    {% if power.log_url %}<a href="{{ power.log_url }}" target="_blank" rel="noopener">Start Lambda log</a>{% endif %}
  </div>
</div>
<p class="legend"><span class="key instance"></span>Instance window <span class="key lock"></span>Power lock <span class="key session"></span>NYSE session</p>
```

`tools/console/templates/panels/exceptions.html`:

```html
<h2 id="annun-h">Exceptions</h2>
{% if checks is none %}
<p class="muted">No snapshot yet</p>
{% else %}
<div class="tiles">
  {% for check in checks %}
  <div class="tile {{ check.level }}"><strong>{{ check.name }}</strong><span class="level">{{ check.level|upper }}</span><small>{{ check.detail }}</small></div>
  {% endfor %}
</div>
{% endif %}
```

`tools/console/templates/panels/strategies.html`:

```html
<h2 id="strats-h">Strategies</h2>
<table>
  <thead><tr><th>Strategy</th><th>Switch</th><th>Process</th><th>Variant</th><th>Last decision</th><th>Gates</th></tr></thead>
  <tbody>
  {% for row in strategies %}
    <tr class="{{ 'caution' if row.caution else '' }} {{ '' if live else 'frozen' }}">
      <td>{{ row.name }}</td>
      <td>{{ 'on' if row.enabled else 'off' }}</td>
      <td>{{ row.process }}{% if row.caution %} <span class="level">CAUTION: switch on, no unit</span>{% endif %}</td>
      <td>{{ row.variant or '-' }}</td>
      <td>{{ row.decided_at }} {% if row.passed is true %}passed{% elif row.passed is false %}not passed{% endif %}</td>
      <td>{% if row.has_gates %}<button hx-get="/panels/gates/{{ row.name }}" hx-target="#gates">Show gates</button>{% endif %}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
```

`tools/console/templates/panels/gates.html`:

```html
<h2>Gates for {{ strategy }}{% if summary %} at {{ summary.timestamp }}{% endif %}</h2>
{% if error %}<p class="tile warning">Could not read the decision record: {{ error }}</p>{% endif %}
<table>
  <thead><tr><th>Gate</th><th>Result</th><th>Error</th></tr></thead>
  <tbody>
  {% for name, gate in gates.items() %}
    <tr class="{{ '' if gate.passed else 'caution' }}"><td>{{ name }}</td><td>{{ 'passed' if gate.passed else 'FAILED' }}</td><td>{{ gate.error or '' }}</td></tr>
  {% endfor %}
  </tbody>
</table>
```

`tools/console/templates/panels/account.html`:

```html
<h2 id="pf-h">Account</h2>
{% if not account %}<p class="muted">No account readings</p>{% endif %}
{% for row in account %}
<div class="account-row">
  <strong>{{ row.name }}</strong>
  {% for cell in row.cells %}<span class="cell {{ cell.cls }}">{{ cell.label }} {{ cell.value }}</span>{% endfor %}
  {% if row.age %}<small class="muted">as of {{ row.age }}; prices have moved since</small>{% endif %}
</div>
{% endfor %}
```

`tools/console/templates/panels/host.html`:

```html
<h2 id="host-h">Host</h2>
<table class="{{ '' if live else 'frozen' }}">
  <thead><tr><th>Unit</th><th>State</th><th>Restarts</th><th>Memory</th><th>Since</th></tr></thead>
  <tbody>
  {% for unit in units %}
    <tr class="{{ '' if unit.active_state == 'active' else 'caution' }}">
      <td>{{ unit.unit }}</td><td>{{ unit.active_state }} ({{ unit.sub_state }})</td>
      <td>{{ unit.restarts if unit.restarts is not none else '-' }}</td>
      <td>{{ '%.0f MB'|format(unit.memory_bytes / 1048576) if unit.memory_bytes else '-' }}</td>
      <td>{{ unit.active_since or '-' }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
```

`tools/console/static/console.css`:

```css
:root {
  --bg: #1e1e1e;
  --panel: #262626;
  --line: #3a3a3a;
  --text: #e0e0e0;
  --muted: #a0a0a0;
  --caution: #b45309;
  --warning: #b91c1c;
  --instance: #2563eb;
  --lock: #7c3aed;
  --session: #15803d;
  --hatch: #333333;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, sans-serif; }
h1 { margin: 0 0 4px; font-size: 20px; }
h2 { margin: 20px 0 8px; font-size: 16px; }
section { max-width: 1200px; }
.muted, small { color: var(--muted); }
.badge { display: inline-block; margin: 4px 0; padding: 4px 8px; border-radius: 4px; background: var(--panel); }
.badge.caution, .tile.caution, tr.caution td { background: var(--caution); color: #ffffff; }
.badge.warning, .tile.warning { background: var(--warning); color: #ffffff; }
.badge.unknown, .tile.unknown {
  background: repeating-linear-gradient(45deg, var(--panel), var(--panel) 6px, var(--hatch) 6px, var(--hatch) 12px);
  color: var(--text);
}
.level { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; margin-right: 6px; }
.errors { margin: 4px 0; padding-left: 18px; color: #fca5a5; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
.tile { display: flex; flex-direction: column; gap: 2px; padding: 10px; border-radius: 6px; background: var(--panel); }
.tile small { color: inherit; opacity: 0.85; }
.rail-row { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 12px; align-items: start; }
.rail { width: 100%; height: auto; background: var(--panel); border-radius: 6px; }
.rail .hour, .rail .tick-label { fill: var(--muted); font-size: 9px; text-anchor: middle; }
.band.instance, .key.instance { fill: var(--instance); background: var(--instance); }
.band.lock, .key.lock { fill: var(--lock); background: var(--lock); }
.band.session, .key.session { fill: var(--session); background: var(--session); }
.rail .tick { stroke: #f5f5f5; stroke-width: 2; }
.rail .now { stroke: #facc15; stroke-width: 2; }
.legend { color: var(--muted); font-size: 12px; }
.key { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px; border-radius: 2px; }
table { width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 6px; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
.frozen td, .frozen { color: var(--muted); }
.account-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; padding: 8px; background: var(--panel); border-radius: 6px; margin-bottom: 6px; }
.cell.drifting { color: var(--muted); font-style: italic; }
.cell.frozen { color: var(--muted); }
button { background: var(--instance); color: #ffffff; border: 0; border-radius: 4px; padding: 3px 8px; cursor: pointer; }
@media (max-width: 720px) { .rail-row { grid-template-columns: 1fr; } }
```

- [ ] **Step 7: Write the failing app tests**

`tests/console_app/test_app.py`:

```python
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from tests.console_app.test_views import DOCUMENT, RECORD, SCHEDULES
from tools.console.app import create_app
from tools.console.poller import ConsoleState

NOW = datetime(2026, 10, 8, 16, 30, tzinfo=timezone.utc)


class StubPoller:
    def __init__(self, state):
        self.state = state


def make_client(state, host="127.0.0.1:8765"):
    app = create_app(StubPoller(state), "us-east-1", clock=lambda: NOW, start_polling=False)
    return TestClient(app, base_url=f"http://{host}")


def live_state():
    return ConsoleState(document=DOCUMENT, source="agent", as_of=NOW - timedelta(seconds=4),
                        instance_state="running", schedules=SCHEDULES, decisions={"ramp": RECORD})


def test_page_renders_every_panel():
    response = make_client(live_state()).get("/")

    assert response.status_code == 200
    for heading in ("Today's schedule", "Exceptions", "Strategies", "Account", "Host"):
        assert heading in response.text
    assert "Live from agent, 4 s ago" in response.text


def test_each_panel_fragment_renders():
    client = make_client(live_state())
    for panel in ("header", "schedule", "exceptions", "strategies", "account", "host"):
        assert client.get(f"/panels/{panel}").status_code == 200


def test_unknown_panel_is_404():
    assert make_client(live_state()).get("/panels/nope").status_code == 404


def test_gates_fragment():
    client = make_client(live_state())

    assert "strategy_enabled" in client.get("/panels/gates/ramp").text
    assert client.get("/panels/gates/mp").status_code == 404


def test_empty_state_renders_no_snapshot_yet():
    response = make_client(ConsoleState()).get("/")

    assert response.status_code == 200
    assert "No snapshot yet" in response.text
    assert "No status yet" in response.text


def test_snapshot_state_renders_unknown_tiles():
    state = live_state()
    state.source, state.agent_down_since, state.reason = "s3", NOW, "shutdown"

    response = make_client(state).get("/panels/exceptions")

    assert "tile unknown" in response.text
    assert "Unknown since" in response.text


def test_foreign_host_is_rejected():
    assert make_client(live_state(), host="evil.example:8765").get("/").status_code == 400


def test_localhost_is_accepted():
    assert make_client(live_state(), host="localhost:8765").get("/").status_code == 200


def test_post_is_not_allowed():
    assert make_client(live_state()).post("/").status_code == 405
```

- [ ] **Step 8: Run the app tests to verify they fail**

Run: `$PY -m pytest tests/console_app/test_app.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.console.app'`

- [ ] **Step 9: Implement the app and entry point**

`tools/console/app.py`:

```python
"""FastAPI app: the page, the htmx panel fragments, and a Host check (127.0.0.1 only)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tools.console import views

HERE = Path(__file__).resolve().parent


def create_app(poller, region: str, clock: Callable[[], datetime], start_polling: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(poller.run()) if start_polling else None
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # Defeats DNS rebinding: a page on another origin cannot reach the app under its own hostname.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    def render(request: Request, template: str, context: dict) -> HTMLResponse:
        return templates.TemplateResponse(request, template, context)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return render(request, "index.html", views.page_context(poller.state, clock(), region))

    @app.get("/panels/{panel}", response_class=HTMLResponse)
    def panel(request: Request, panel: str) -> HTMLResponse:
        if panel not in views.PANELS:
            raise HTTPException(status_code=404)
        return render(request, f"panels/{panel}.html", views.page_context(poller.state, clock(), region))

    @app.get("/panels/gates/{strategy}", response_class=HTMLResponse)
    def gates(request: Request, strategy: str) -> HTMLResponse:
        context = views.gates_context(poller.state, strategy)
        if context is None:
            raise HTTPException(status_code=404)
        return render(request, "panels/gates.html", context)

    return app
```

`tools/console/__main__.py`:

```python
"""Entry point: python -m tools.console (run from the repo root)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from botocore.exceptions import ProfileNotFound
from dotenv import load_dotenv

from src.utils.logger import logger
from tools.console.app import create_app
from tools.console.config import settings_from_env
from tools.console.poller import AgentClient, Poller, make_aws_clients

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    try:
        settings = settings_from_env(os.environ)
        aws = make_aws_clients(settings)
    except (ValueError, ProfileNotFound) as e:
        logger.error(f"[console] refusing to start: {e}")
        raise SystemExit(1)
    poller = Poller(settings, AgentClient(settings.agent_url), aws, utc_now)
    logger.info(f"[console] serving on http://127.0.0.1:{settings.port}")
    uvicorn.run(create_app(poller, settings.region, utc_now), host="127.0.0.1", port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run all console app tests**

Run: `$PY -m pytest tests/console_app -q -p no:cacheprovider`
Expected: PASS (every test in tests/console_app)

- [ ] **Step 11: Commit**

```bash
git add tools/console/views.py tools/console/app.py tools/console/__main__.py tools/console/templates tools/console/static tests/console_app/test_views.py tests/console_app/test_app.py
git commit -m "feat(console): panels, templates and the local FastAPI app

Schedule rail, power tile, exceptions, strategies with gates, account
and host panels, rendered from one ConsoleState; Host check rejects
anything but 127.0.0.1 and localhost.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 9: Install, autostart and docs

**Files:**
- Create: `tools/console/requirements.txt`, `tools/console/README.md`, `tools/console/install_windows_task.ps1`, `tools/console/install_macos_launchd.sh`, `tools/console/com.homeguard.console.plist`
- Modify: `.env.example`, `docs/architecture/ARCHITECTURE_OVERVIEW.md`, `CLAUDE.md` (architecture block)
- Test: `tests/console_app/test_install_files.py`

- [ ] **Step 1: Write the failing test**

`tests/console_app/test_install_files.py`:

```python
import plistlib
from pathlib import Path

CONSOLE = Path(__file__).resolve().parents[2] / "tools" / "console"
REPO_ROOT = CONSOLE.parents[1]


def test_launchd_plist_runs_the_console_module():
    plist = plistlib.loads((CONSOLE / "com.homeguard.console.plist").read_bytes())

    assert plist["Label"] == "com.homeguard.console"
    assert plist["ProgramArguments"][1:] == ["-m", "tools.console"]
    assert plist["RunAtLoad"] is True


def test_requirements_list_the_app_dependencies():
    names = {line.split("==")[0].split(">=")[0].strip() for line in (CONSOLE / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")}

    assert {"fastapi", "uvicorn", "jinja2", "httpx", "boto3", "python-dotenv", "pandas_market_calendars"} <= names


def test_env_example_has_placeholders_for_the_console():
    text = (REPO_ROOT / ".env.example").read_text()

    assert "CONSOLE_AGENT_URL=<YOUR_" in text
    assert "CONSOLE_SNAPSHOT_BUCKET=<YOUR_" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/console_app/test_install_files.py -q -p no:cacheprovider`
Expected: FAIL with `FileNotFoundError` for the plist

- [ ] **Step 3: Write the files**

`tools/console/requirements.txt`:

```text
# Homeguard Console local app (python -m tools.console). The fintech env already has these.
fastapi>=0.110
uvicorn>=0.30
jinja2>=3.1
httpx>=0.27
boto3>=1.34
python-dotenv>=1.0
pandas_market_calendars>=4.4
pyyaml>=6.0
```

`tools/console/com.homeguard.console.plist` (install_macos_launchd.sh fills the placeholders):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.homeguard.console</string>
  <key>ProgramArguments</key>
  <array>
    <string>__PYTHON__</string>
    <string>-m</string>
    <string>tools.console</string>
  </array>
  <key>WorkingDirectory</key>
  <string>__REPO__</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>__HOME__/Library/Logs/homeguard-console.log</string>
  <key>StandardErrorPath</key>
  <string>__HOME__/Library/Logs/homeguard-console.log</string>
</dict>
</plist>
```

`tools/console/install_macos_launchd.sh`:

```bash
#!/bin/bash
# Installs the Homeguard Console as a launchd agent that starts at login.
#   bash tools/console/install_macos_launchd.sh /path/to/env/bin/python
set -euo pipefail

PYTHON="${1:?usage: install_macos_launchd.sh /path/to/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.homeguard.console.plist"

"$PYTHON" -c "import fastapi, uvicorn, jinja2, httpx, boto3" || {
    echo "[-] Missing dependencies; run: $PYTHON -m pip install -r $REPO/tools/console/requirements.txt"
    exit 1
}
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s|__PYTHON__|$PYTHON|g" -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" \
    "$REPO/tools/console/com.homeguard.console.plist" > "$PLIST"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "[+] Homeguard Console installed; open http://127.0.0.1:8765 (log: ~/Library/Logs/homeguard-console.log)"
```

`tools/console/install_windows_task.ps1`:

```powershell
# Registers a logon task that runs the Homeguard Console on http://127.0.0.1:8765.
#   powershell -File tools\console\install_windows_task.ps1 -Python C:\path\to\env\pythonw.exe
param([Parameter(Mandatory = $true)][string]$Python)
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
& $Python -c "import fastapi, uvicorn, jinja2, httpx, boto3"
if ($LASTEXITCODE -ne 0) { throw "Missing dependencies; run: $Python -m pip install -r $repo\tools\console\requirements.txt" }

$action = New-ScheduledTaskAction -Execute $Python -Argument "-m tools.console" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "HomeguardConsole" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Homeguard Console (read-only) on 127.0.0.1:8765" -Force | Out-Null
Start-ScheduledTask -TaskName "HomeguardConsole"
Write-Output "[+] HomeguardConsole task registered and started; open http://127.0.0.1:8765"
```

Append to `.env.example`:

```bash
# Homeguard Console local app (tools/console). Agent URL is the tailnet https URL of the agent on :8443.
CONSOLE_AGENT_URL=<YOUR_TAILNET_AGENT_URL>
CONSOLE_SNAPSHOT_BUCKET=<YOUR_BUCKET_NAME>
CONSOLE_AWS_PROFILE=homeguard-console
```

`tools/console/README.md`:

```markdown
# Homeguard Console (local app)

Read-only operations console for the EC2 trading instance. It polls the console agent over the
tailnet every 10 s, falls back to the S3 snapshot (`console/latest/status.json`) when the agent is
unreachable, and reads EC2 and the four EventBridge schedules with the read-only `homeguard-console`
AWS profile. Design: `docs/superpowers/specs/2026-10-08-homeguard-console-phase2a-design.md`.

## Setup

1. Add `CONSOLE_AGENT_URL`, `CONSOLE_SNAPSHOT_BUCKET` (and optionally `CONSOLE_AWS_PROFILE`) to the repo
   `.env`; `EC2_INSTANCE_ID` and `EC2_REGION` are already there.
2. `aws configure --profile homeguard-console` with the keys from `aws iam create-access-key --user-name homeguard-console`.
3. Run once by hand: `python -m tools.console`, then open http://127.0.0.1:8765.
4. Start at login:
   - Windows: `powershell -File tools\console\install_windows_task.ps1 -Python <env>\pythonw.exe`
   - macOS: `bash tools/console/install_macos_launchd.sh <env>/bin/python`

## Modules

| Module | Role |
| --- | --- |
| `poller.py` | ConsoleState from the agent, S3, EC2 and the Scheduler |
| `schedule.py` | Cron parsing, expected instance state, NYSE session and power lock |
| `checks.py` | The eight exception checks |
| `freshness.py` | Live, frozen, drifting and unknown classes |
| `views.py` | Template contexts |
| `app.py` | FastAPI app and Host check |
| `decision_times.py` | Decision times, pinned to the adapters by a test |
```

Add a `tools/console/` line to the architecture block in `CLAUDE.md` under `### src/ packages` (immediately after the `console_agent/` line):

```text
(tools/console/  Local read-only console app: FastAPI + htmx on 127.0.0.1:8765, agent + S3 + EC2 sources)
```

and the same one-line entry, with a sentence on the S3 snapshot path, to `docs/architecture/ARCHITECTURE_OVERVIEW.md` next to its console agent entry.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/console_app tests/console_agent -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Smoke-run the entry point's fail-fast path**

```bash
CONSOLE_AGENT_URL=https://agent.invalid:8443 CONSOLE_SNAPSHOT_BUCKET=none EC2_INSTANCE_ID=i-0 EC2_REGION=us-east-1 \
  CONSOLE_AWS_PROFILE=does-not-exist timeout 20 $PY -m tools.console; echo "exit=$?"
```

Expected: a `refusing to start:` error naming the `does-not-exist` profile and `exit=1`, with no traceback. (Variables already set in the environment take precedence over the repo .env, since `load_dotenv` does not override.) The served path is exercised in Task 10 Step 6 with the real profile.

- [ ] **Step 6: Commit**

```bash
git add tools/console/requirements.txt tools/console/README.md tools/console/install_windows_task.ps1 tools/console/install_macos_launchd.sh tools/console/com.homeguard.console.plist .env.example docs/architecture/ARCHITECTURE_OVERVIEW.md CLAUDE.md tests/console_app/test_install_files.py
git commit -m "feat(console): start-at-login installers, requirements and docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GrABMv5NQvWYtGp3DnrKmk"
```

---

### Task 10: Whole-branch verification, merge and rollout

Steps marked [operator] wait for the operator's explicit go-ahead in the conversation.

- [ ] **Step 1: Full suite**

Run: `$PY -m pytest tests/trading tests/monitoring tests/console_agent tests/console_app -q -p no:cacheprovider`
Expected: PASS except the known flaky Alpaca-network tests; any other failure blocks the merge.

- [ ] **Step 2: Final review, then merge to main**

Run the whole-branch review per the execution skill, fix Critical and Important findings with failing-first tests, then fast-forward main from the main dir: `git -C /c/Users/qwqw1/Dropbox/cs/github/Homeguard merge --ff-only feat/console-p2a` and push main (standing permission).

- [ ] **Step 3: [operator] Apply Terraform**

Set `console_snapshot_bucket` in the local `infra/terraform/terraform.tfvars`, run the targeted plan from `infra/terraform/README.md` with `-var 'ssh_allowed_cidrs=["<YOUR_IP>/32"]'` (the operator's current IP, as in the local terraform.tfvars), show the plan, and apply only after approval. Expected: 6 resources to add, 0 to change, 0 to destroy (plus the data source).

- [ ] **Step 4: [operator] Keys and profile**

The operator runs `aws iam create-access-key --user-name homeguard-console` and `aws configure --profile homeguard-console` on the Windows PC and the Mac. Verify: `aws --profile homeguard-console ec2 describe-instances --instance-ids "$EC2_INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name'` and `aws --profile homeguard-console scheduler get-schedule --name homeguard-start-instance --query ScheduleExpression` both succeed (exit gate item 1; repeat for the other three schedules).

- [ ] **Step 5: [operator] Instance install**

Cherry-pick the Task 1 and Task 2 commits onto `ramp-phase4-turnover-regime-research` in a separate worktree, run `tests/console_agent` there, push, then on the instance: `git pull --ff-only`, add `CONSOLE_SNAPSHOT_BUCKET` to `~/Homeguard/.env`, and `bash infra/ec2/setup/install_console_upload.sh`. Expected: `aws --version` prints, `systemd-analyze verify` is silent, one upload lands (`aws s3 ls` shows `status.json`), and the timer is listed. Outside market hours only.

- [ ] **Step 6: Run the console**

On the Windows PC: add the console values to `.env`, run `python -m tools.console`, open http://127.0.0.1:8765, confirm "Live from agent" while the instance is up; then install the logon task. On the Mac: same with launchd.

- [ ] **Step 7: Exit gate**

After the next 20:00 ET scheduled stop: the console shows "Snapshot from S3, taken ... (shutdown)" with `uploaded_at` within a minute of 20:00, unknown tiles for IB Gateway, Broker heartbeat, Market data stream and Host memory, and the power tile reads "Instance stopped, as scheduled". The missed start warning is covered by `test_missed_start_power_tile_links_the_start_lambda_log` and `test_power_status`.

- [ ] **Step 8: Session log**

Write `docs/progress/20261008_CONSOLE_PHASE2A.md` in the CLAUDE.md format (summary, changes, commits, remaining work including Phase 2b, validation), commit and push.
