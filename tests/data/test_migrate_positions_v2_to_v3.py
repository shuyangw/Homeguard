"""Tests for the v2 -> v3 strategy_positions.json migration."""
import json
import sys
from pathlib import Path

import pytest

# Migration script lives under scripts/data/; add to path for import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from migrate_positions_v2_to_v3 import migrate_state, V3_VERSION  # noqa: E402


# A synthetic v2 file: pre-futures schema. Top-level is dict of
# strategy_name -> {positions: {symbol: PositionInfo}, ...}.
V2_FIXTURE = {
    "strategies": {
        "ramp": {
            "positions": {
                "AAPL": {"qty": 100, "entry_price": 180.50, "entry_time": "2026-04-15T15:55:00Z", "order_id": "abc123"},
                "MSFT": {"qty": 50, "entry_price": 410.20, "entry_time": "2026-04-15T15:55:01Z", "order_id": "abc124"},
            },
            "last_execution": "2026-04-15T15:55:30Z",
        },
        "omr": {
            "positions": {
                "TQQQ": {"qty": -25, "entry_price": 65.10, "entry_time": "2026-04-14T15:55:00Z", "order_id": "xyz999"},
            },
            "last_execution": "2026-04-14T15:55:30Z",
        },
    }
}


def test_migrate_adds_version_field():
    """v3 output has version: 3 at top level."""
    out = migrate_state(V2_FIXTURE)
    assert out["version"] == V3_VERSION == 3


def test_migrate_preserves_strategy_structure():
    """The strategies/positions structure is preserved."""
    out = migrate_state(V2_FIXTURE)
    assert "ramp" in out["strategies"]
    assert "omr" in out["strategies"]
    assert "AAPL" in out["strategies"]["ramp"]["positions"]
    assert "TQQQ" in out["strategies"]["omr"]["positions"]


def test_migrate_adds_null_contract_month_to_existing_positions():
    """Stock and options positions in v2 get contract_month=null in v3.
    This marks them as non-futures so loader code can branch correctly."""
    out = migrate_state(V2_FIXTURE)
    aapl = out["strategies"]["ramp"]["positions"]["AAPL"]
    assert aapl["contract_month"] is None
    assert aapl["raw_symbol"] is None
    msft = out["strategies"]["ramp"]["positions"]["MSFT"]
    assert msft["contract_month"] is None


def test_migrate_preserves_existing_position_fields():
    """qty / entry_price / entry_time / order_id pass through unchanged."""
    out = migrate_state(V2_FIXTURE)
    aapl = out["strategies"]["ramp"]["positions"]["AAPL"]
    assert aapl["qty"] == 100
    assert aapl["entry_price"] == 180.50
    assert aapl["entry_time"] == "2026-04-15T15:55:00Z"
    assert aapl["order_id"] == "abc123"


def test_migrate_idempotent_on_already_v3():
    """Migrating a v3 file returns it unchanged."""
    out_once = migrate_state(V2_FIXTURE)
    out_twice = migrate_state(out_once)
    assert out_once == out_twice


def test_migrate_writes_file_atomically(tmp_path):
    """File-level migration: read v2 from disk, write v3 to disk, verify content."""
    from migrate_positions_v2_to_v3 import migrate_file

    src = tmp_path / "v2.json"
    src.write_text(json.dumps(V2_FIXTURE))
    out = tmp_path / "v3.json"
    migrate_file(src, out)

    written = json.loads(out.read_text())
    assert written["version"] == 3
    assert written["strategies"]["ramp"]["positions"]["AAPL"]["contract_month"] is None
