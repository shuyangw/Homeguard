"""Tests for AuditLog."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.trading.futures.audit_log import AuditEntry, AuditLog


def test_log_appends_to_daily_file(tmp_path):
    """Each entry appends to YYYYMMDD-rotated file as JSONL."""
    log = AuditLog(log_dir=tmp_path)
    entry = AuditEntry(
        timestamp=datetime(2026, 5, 11, 14, 0, tzinfo=timezone.utc),
        event_type="submit",
        strategy="adaptation_d",
        broker="ibkr_futures",
        raw_symbol="MESH6",
        contract_month="202603",
        side="BUY",
        quantity=2,
        order_type="LIMIT",
        limit_price=5300.0,
        fill_price=None,
        fill_quantity=None,
        ibkr_order_id=12345,
        ibkr_perm_id=67890,
        ibkr_exec_id=None,
        error_message=None,
    )
    log.log(entry)

    path = tmp_path / "audit_20260511.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "submit"
    assert parsed["raw_symbol"] == "MESH6"
    assert parsed["quantity"] == 2
    assert parsed["ibkr_order_id"] == 12345


def test_multiple_entries_same_day_append(tmp_path):
    log = AuditLog(log_dir=tmp_path)
    base = datetime(2026, 5, 11, 14, 0, tzinfo=timezone.utc)
    for i in range(3):
        log.log(AuditEntry(
            timestamp=base, event_type="submit",
            strategy="x", broker="ibkr_futures",
            raw_symbol="MESH6", contract_month="202603",
            side="BUY", quantity=i + 1, order_type="LIMIT",
            limit_price=5300.0,
            fill_price=None, fill_quantity=None,
            ibkr_order_id=i, ibkr_perm_id=None, ibkr_exec_id=None,
            error_message=None,
        ))
    path = tmp_path / "audit_20260511.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[0])["quantity"] == 1
    assert json.loads(lines[2])["quantity"] == 3


def test_different_days_different_files(tmp_path):
    log = AuditLog(log_dir=tmp_path)
    e1 = AuditEntry(
        timestamp=datetime(2026, 5, 11, 14, 0, tzinfo=timezone.utc),
        event_type="submit", strategy="x", broker="ibkr_futures",
        raw_symbol="MESH6", contract_month="202603",
        side="BUY", quantity=1, order_type="MARKET",
        limit_price=None, fill_price=None, fill_quantity=None,
        ibkr_order_id=1, ibkr_perm_id=None, ibkr_exec_id=None,
        error_message=None,
    )
    e2 = AuditEntry(
        timestamp=datetime(2026, 5, 12, 14, 0, tzinfo=timezone.utc),
        event_type="submit", strategy="x", broker="ibkr_futures",
        raw_symbol="MESH6", contract_month="202603",
        side="BUY", quantity=1, order_type="MARKET",
        limit_price=None, fill_price=None, fill_quantity=None,
        ibkr_order_id=2, ibkr_perm_id=None, ibkr_exec_id=None,
        error_message=None,
    )
    log.log(e1)
    log.log(e2)
    assert (tmp_path / "audit_20260511.jsonl").exists()
    assert (tmp_path / "audit_20260512.jsonl").exists()


def test_log_submission_helper(tmp_path):
    """Helper method takes a ResolvedOrder-like obj + ibkr_response dict."""
    log = AuditLog(log_dir=tmp_path)
    from types import SimpleNamespace
    order = SimpleNamespace(
        strategy_intent="ES.v.0",
        strategy="adaptation_d",
        raw_symbol="ESM4",
        contract_month="202406",
        side=SimpleNamespace(value="BUY"),
        quantity=2,
        order_type=SimpleNamespace(value="LIMIT"),
        limit_price=5300.0,
    )
    log.log_submission(order, ibkr_response={"orderId": 999, "permId": 1234})
    paths = list(tmp_path.glob("audit_*.jsonl"))
    assert len(paths) == 1
    entry = json.loads(paths[0].read_text().strip())
    assert entry["event_type"] == "submit"
    assert entry["ibkr_order_id"] == 999
    assert entry["ibkr_perm_id"] == 1234
    assert entry["raw_symbol"] == "ESM4"
    assert entry["extras"]["resolved_from"] == "ES.v.0"


def test_log_fill_helper(tmp_path):
    log = AuditLog(log_dir=tmp_path)
    from types import SimpleNamespace
    fill = SimpleNamespace(
        timestamp=datetime(2026, 5, 11, 14, 5, tzinfo=timezone.utc),
        strategy_tag="adaptation_d",
        symbol="ESM4",
        contract_month="202406",
        side="BUY",
        quantity=2,
        fill_price=5300.25,
        fill_quantity=2,
        order_id=999,
        exec_id="EXEC-abc",
    )
    log.log_fill(fill)
    paths = list(tmp_path.glob("audit_*.jsonl"))
    entry = json.loads(paths[0].read_text().strip())
    assert entry["event_type"] == "fill"
    assert entry["fill_price"] == 5300.25
    assert entry["fill_quantity"] == 2
    assert entry["ibkr_exec_id"] == "EXEC-abc"


def test_log_cancel_and_reject(tmp_path):
    log = AuditLog(log_dir=tmp_path)
    log.log_cancel(
        timestamp=datetime(2026, 5, 11, 14, 10, tzinfo=timezone.utc),
        strategy="adaptation_d", raw_symbol="ESM4", contract_month="202406",
        ibkr_order_id=999,
    )
    log.log_reject(
        timestamp=datetime(2026, 5, 11, 14, 11, tzinfo=timezone.utc),
        strategy="adaptation_d", raw_symbol="ESM4", contract_month="202406",
        ibkr_order_id=1000, error_message="margin shortfall",
    )
    path = tmp_path / "audit_20260511.jsonl"
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "cancel"
    assert json.loads(lines[1])["event_type"] == "reject"
    assert json.loads(lines[1])["error_message"] == "margin shortfall"


def test_jsonl_each_line_valid(tmp_path):
    """Every line in the file must be a valid JSON object."""
    log = AuditLog(log_dir=tmp_path)
    for i in range(5):
        log.log(AuditEntry(
            timestamp=datetime(2026, 5, 11, 14, i, tzinfo=timezone.utc),
            event_type="submit", strategy="x", broker="ibkr_futures",
            raw_symbol="MESH6", contract_month="202603",
            side="BUY", quantity=1, order_type="MARKET",
            limit_price=None, fill_price=None, fill_quantity=None,
            ibkr_order_id=i, ibkr_perm_id=None, ibkr_exec_id=None,
            error_message=None,
        ))
    path = tmp_path / "audit_20260511.jsonl"
    for line in path.read_text().strip().split("\n"):
        json.loads(line)  # raises if invalid
