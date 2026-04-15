"""Tests for v1 -> v2 state file migration (broker tagging)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.trading.state.strategy_state_manager import StrategyStateManager


@pytest.fixture
def temp_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_file = tmp / "strategy_positions.json"
        toggle_file = tmp / "strategy_toggle.yaml"
        toggle_file.write_text(
            "strategies:\n"
            "  omr:\n    enabled: true\n"
            "  mp:\n    enabled: true\n"
            "  ramp:\n    enabled: true\n"
        )
        yield state_file, toggle_file


def _write_v1_state(state_file: Path) -> None:
    """Seed a v1-format state file with untagged positions."""
    v1 = {
        "version": 1,
        "last_updated": "2026-04-14T10:00:00-04:00",
        "execution_lock": None,
        "strategies": {
            "omr": {
                "positions": {
                    "TQQQ": {
                        "qty": 100,
                        "entry_price": 52.30,
                        "entry_time": "2026-04-14T15:50:00-04:00",
                        "order_id": "abc123",
                    },
                    "SOXL": {
                        "qty": 50,
                        "entry_price": 28.10,
                        "entry_time": "2026-04-14T15:50:00-04:00",
                        "order_id": "abc124",
                    },
                },
                "last_execution": "2026-04-14T15:50:00-04:00",
            },
            "mp": {
                "positions": {
                    "PLTR": {
                        "qty": 200,
                        "entry_price": 18.00,
                        "entry_time": "2026-04-13T10:00:00-04:00",
                        "order_id": "def456",
                    }
                },
                "last_execution": "2026-04-13T10:00:00-04:00",
            },
        },
    }
    state_file.write_text(json.dumps(v1))


def test_migration_bumps_version_and_tags_positions(temp_paths):
    state_file, toggle_file = temp_paths
    _write_v1_state(state_file)

    StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    # Re-read from disk -- migration must have persisted.
    on_disk = json.loads(state_file.read_text())
    assert on_disk["version"] == 2
    omr = on_disk["strategies"]["omr"]["positions"]
    mp = on_disk["strategies"]["mp"]["positions"]
    assert omr["TQQQ"]["broker"] == "alpaca"
    assert omr["SOXL"]["broker"] == "alpaca"
    assert mp["PLTR"]["broker"] == "alpaca"


def test_migration_noop_on_v2_file(temp_paths):
    state_file, toggle_file = temp_paths
    v2 = {
        "version": 2,
        "last_updated": "2026-04-14T10:00:00-04:00",
        "execution_lock": None,
        "strategies": {
            "omr": {
                "positions": {
                    "TQQQ": {
                        "qty": 100,
                        "entry_price": 52.30,
                        "entry_time": "2026-04-14T15:50:00-04:00",
                        "order_id": "abc123",
                        "broker": "ibkr",
                    }
                },
                "last_execution": None,
            }
        },
    }
    state_file.write_text(json.dumps(v2))

    StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    on_disk = json.loads(state_file.read_text())
    assert on_disk["version"] == 2
    # The ibkr tag must be preserved -- migration must NOT stamp alpaca.
    assert on_disk["strategies"]["omr"]["positions"]["TQQQ"]["broker"] == "ibkr"


def test_migration_handles_malformed_strategy_entry(temp_paths):
    state_file, toggle_file = temp_paths
    v1 = {
        "version": 1,
        "strategies": {
            "omr": {
                "positions": {
                    "TQQQ": {"qty": 10, "entry_price": 50.0, "entry_time": "t", "order_id": "x"},
                },
                "last_execution": None,
            },
            "broken": None,  # malformed entry
        },
    }
    state_file.write_text(json.dumps(v1))
    StrategyStateManager(state_file=state_file, toggle_file=toggle_file)
    on_disk = json.loads(state_file.read_text())
    assert on_disk["version"] == 2
    assert on_disk["strategies"]["omr"]["positions"]["TQQQ"]["broker"] == "alpaca"
    # broken entry preserved as-is; no crash
    assert "broken" in on_disk["strategies"]


def test_migration_handles_empty_strategies(temp_paths):
    state_file, toggle_file = temp_paths
    v1 = {
        "version": 1,
        "strategies": {
            "omr": {"positions": {}, "last_execution": None},
        },
    }
    state_file.write_text(json.dumps(v1))

    StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    on_disk = json.loads(state_file.read_text())
    assert on_disk["version"] == 2


def test_add_position_requires_broker(temp_paths):
    state_file, toggle_file = temp_paths
    mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    with pytest.raises(ValueError, match="broker"):
        mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x')


def test_add_position_stores_broker_tag(temp_paths):
    state_file, toggle_file = temp_paths
    mgr = StrategyStateManager(state_file=state_file, toggle_file=toggle_file)

    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    positions = mgr.get_positions('omr')
    assert positions['TQQQ']['broker'] == 'alpaca'
