import json

import pytest

from src.utils.run_status import RunStatus


def _read(tmp_path, name):
    files = list(tmp_path.glob(f"{name}_*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


def test_clean_run_records_done(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.run_status._STATUS_DIR", tmp_path)
    with RunStatus("clean", meta={"x": 1}) as st:
        st.heartbeat(note="mid")
    d = _read(tmp_path, "clean")
    assert d["status"] == "DONE"
    assert d["meta"]["x"] == 1
    assert d["note"] == "mid"
    assert d["started_at"] and d["ended_at"]


def test_exception_records_error_and_reraises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.run_status._STATUS_DIR", tmp_path)
    with pytest.raises(ValueError):
        with RunStatus("boom"):
            raise ValueError("kaboom")
    d = _read(tmp_path, "boom")
    assert d["status"] == "ERROR"
    assert "kaboom" in d["error"]


def test_killed_run_leaves_stale_running(tmp_path, monkeypatch):
    # A SIGKILL'd run never reaches __exit__; the file must remain RUNNING with a
    # heartbeat so we can tell it died (and roughly when) rather than guessing.
    monkeypatch.setattr("src.utils.run_status._STATUS_DIR", tmp_path)
    st = RunStatus("killed")
    st.__enter__()
    try:
        d = _read(tmp_path, "killed")
        assert d["status"] == "RUNNING"
        assert d["started_at"] and d["heartbeat_at"]
    finally:
        st._stop.set()  # stop the heartbeat thread (simulated kill cleanup)
