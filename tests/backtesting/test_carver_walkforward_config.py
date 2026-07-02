import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "wf", "scripts/backtest_scripts/run_carver_walkforward.py")
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)


def test_config_to_kwargs_extracts_params():
    cfg = {
        "asset_class": "futures",
        "strategy": {"universe": ["ES", "GC", "6E"]},
        "dates": {"start": "2010-06-07", "end": "2026-02-20"},
        "backtest": {"initial_capital": 10_000_000,
                     "vol_target_per_instrument": 0.20,
                     "rebalance": "weekly", "cost_mult": 1.0},
    }
    kw = wf._config_to_kwargs(cfg)
    assert kw["universe"] == ["ES", "GC", "6E"]
    assert kw["capital"] == 10_000_000
    assert kw["vol_target"] == 0.20
    assert kw["start"] == "2010-06-07"
    assert kw["end"] == "2026-02-20"


def test_report_interpolates_actual_capital_and_count(tmp_path):
    # Minimal fake result covering everything _write_readiness_report reads.
    result = {
        "oos_sharpe": 0.3, "psr": 1.0, "dsr": 1.0, "pbo": 0.25,
        "oos_sharpe_1_5x_cost": 0.2, "n_windows": 2, "n_oos_days": 500,
        "window_sharpes": [0.3, 0.4], "trial_count": 1,
        "skew": -0.2, "kurtosis_pearson": 5.0,
        "universe": ["ES", "GC", "6E"], "window_universes": [["ES", "GC"], ["ES", "GC", "6E"]],
        "window_start": __import__("datetime").date(2013, 6, 7),
        "window_end": __import__("datetime").date(2026, 2, 20),
        "capital": 10_000_000, "vol_target": 0.20,
    }
    out = tmp_path / "BROAD.md"
    wf._write_readiness_report(result, train_months=36, test_months=12,
                               step_months=12, start="2010-06-07", end="2026-02-20",
                               report_path=str(out))
    text = out.read_text()
    assert "$10,000,000" in text          # actual capital, not the default
    assert "12-instrument" not in text    # stale micro prose removed
    assert "0.20" in text                 # actual vol target
    # Regression guard: the stale alarmist tail-stats prose must never reappear
    # (it was fixed in the baseline file once but not the generator; this locks it).
    for stale in ("extreme skew", "far outside", "not four digits", "DONE_WITH_CONCERNS"):
        assert stale not in text, f"stale tail-stats prose resurfaced: {stale!r}"
    assert "Note: tail statistics" in text
