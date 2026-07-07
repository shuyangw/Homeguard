import pytest
from src.backtesting.engine import futures_backtest as fb

_SLICE = {
    "strategy": {"name": "FuturesCarry", "universe": ["GC", "CL"]},
    "dates": {"start": "2022-01-03", "end": "2022-03-31"},
    "backtest": {"initial_capital": 1_000_000, "vol_target_per_instrument": 0.20,
                 "rebalance": "weekly", "cost_mult": 1.0},
    "pre_registration": {
        "construction": "test fixture -- append_run registration gating check",
        "expected_sign": "long_short",
        "hypothesis": "not a real trial; validates register=True/False routing to append_run only",
    },
}


def test_register_false_skips_append_run(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("append_run must NOT be called when register=False")
    monkeypatch.setattr("src.experiments.append_run", _boom)
    res = fb.run_futures_backtest(_SLICE, register=False)
    assert res["run_id"] is None


def test_register_true_calls_append_run(monkeypatch):
    called = {}
    def _fake(**kwargs):
        called["yes"] = True
        return "rid-123"
    monkeypatch.setattr("src.experiments.append_run", _fake)
    res = fb.run_futures_backtest(_SLICE)  # default register=True
    assert called.get("yes") and res["run_id"] == "rid-123"
