"""Regression test: the `backtest.idm` flag must thread end-to-end through
the carver walk-forward driver (_config_to_kwargs -> _run_window -> the
per-window backtest config), not be silently dropped as the driver rebuilds
its per-window config internally.
"""
from datetime import date

import scripts.backtest_scripts.run_carver_walkforward as wf


def test_config_to_kwargs_reads_idm():
    assert wf._config_to_kwargs({
        "strategy": {"universe": ["ES"]},
        "dates": {"start": "2010-01-01", "end": "2011-01-01"},
        "backtest": {"idm": True},
    })["idm"] is True
    assert wf._config_to_kwargs({
        "strategy": {"universe": ["ES"]},
        "dates": {"start": "2010-01-01", "end": "2011-01-01"},
        "backtest": {},
    })["idm"] is False


def test_run_window_threads_idm_into_backtest_config(monkeypatch):
    captured = {}

    def fake_backtest(config, register=True):
        captured["idm"] = config["backtest"].get("idm", "MISSING")
        return {"equity_curve": [1.0, 1.0], "trades": None}

    monkeypatch.setattr(wf, "run_futures_backtest", fake_backtest)
    wf._run_window(
        ["ES"], date(2010, 1, 1), date(2010, 6, 1), 1_000_000, 0.20,
        cost_mult=1.0, strategy_name="FuturesCarryXS", idm=True, register=False,
    )
    assert captured["idm"] is True


def test_run_window_idm_defaults_false(monkeypatch):
    captured = {}

    def fake_backtest(config, register=True):
        captured["idm"] = config["backtest"].get("idm", False)
        return {"equity_curve": [1.0, 1.0], "trades": None}

    monkeypatch.setattr(wf, "run_futures_backtest", fake_backtest)
    wf._run_window(
        ["ES"], date(2010, 1, 1), date(2010, 6, 1), 1_000_000, 0.20,
        cost_mult=1.0, strategy_name="FuturesCarry", register=False,
    )
    assert captured["idm"] is False
