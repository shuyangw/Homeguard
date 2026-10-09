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


def test_uploader_does_not_import_trading_or_boto3():
    import subprocess as sp
    import sys

    code = "import src.console_agent.upload, sys; print(any(m.startswith(('src.trading', 'boto3')) for m in sys.modules))"
    result = sp.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=str(upload.REPO_ROOT))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_aws_cli_times_out_on_its_own_before_the_process_is_killed(agent_config, fake_systemd):
    run = FakeRun()

    upload.run_upload(agent_config, BUCKET_ENV, "shutdown", NOW, run)

    args, kwargs = run.calls[0]
    connect = int(args[args.index("--cli-connect-timeout") + 1])
    read = int(args[args.index("--cli-read-timeout") + 1])
    assert connect + read < kwargs["timeout"] < 60
