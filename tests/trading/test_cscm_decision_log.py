"""Integration: CSCM rebalance writes a DecisionRecord via module helpers."""
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

from src.trading.decision_log.reader import latest
from src.trading.decision_log.record import DecisionRecord


def test_cscm_rebalance_emits_decision_record(tmp_decisions_dir):
    """The CSCM rebalance entry point writes a record with cscm-specific extras."""
    from scripts.trading.run_cscm_live import _maybe_emit_decision

    risk_signals = MagicMock(
        regime="bullish", btc_price=77234.40, btc_sma=70932.83,
        is_rebalance_day=True, reduce_exposure=False, exposure_pct=1.0,
        top_symbols=["BTC/USD", "ETH/USD"],
        momentum_scores={"BTC/USD": 0.06, "ETH/USD": 0.07},
    )
    target_positions = {"BTC/USD": 0.5, "ETH/USD": 0.5}
    executed = [
        ("BTC/USD", "buy", "filled", 77234.40, "DEMO-1", None),
        ("ETH/USD", "buy", "skipped", None, None, "broker init failed"),
    ]
    account = {"portfolio_value": 100000.0, "cash": 50000.0, "buying_power": 50000.0}

    _maybe_emit_decision(
        risk_signals=risk_signals,
        target_positions=target_positions,
        executed=executed,
        account=account,
        broker_name="coinbase",
        initial_capital=100000.0,
    )

    rec = latest("cscm")
    assert rec is not None
    assert rec.strategy == "cscm"
    assert rec.inputs.regime == "bullish"
    assert rec.inputs.extra["btc_price"] == 77234.40
    assert rec.inputs.extra["is_rebalance_day"] is True
    assert len(rec.executions) == 2
    assert rec.executions[1].status == "skipped"
    assert rec.executions[1].reason == "broker init failed"
