"""Shared fixtures for decision_log tests."""
import pytest


@pytest.fixture
def tmp_decisions_dir(tmp_path, monkeypatch):
    """Redirect decision_log path resolution to a temp directory.

    Patches paths.decisions_dir and paths.latest_dir so tests don't write
    to the real data/trading/decisions/.
    """
    decisions = tmp_path / "decisions"
    latest = decisions / "_latest"
    decisions.mkdir(parents=True)
    latest.mkdir()

    from src.trading.decision_log import paths as paths_mod
    monkeypatch.setattr(paths_mod, "decisions_dir", lambda: decisions)
    monkeypatch.setattr(paths_mod, "latest_dir", lambda: latest)
    return decisions
