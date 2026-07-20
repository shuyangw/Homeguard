import pandas as pd
from pathlib import Path
from src.backtesting.engine.fill_sink import FillSink


def test_route_result_to_sink_writes_window(tmp_path, monkeypatch):
    # Build a fake res and exercise ONLY the sink-routing branch via the helper.
    from src.backtesting.engine import fx_backtest

    class Res:
        trades = pd.DataFrame({"date": ["2011-01-03"], "pair": ["EURUSD"], "units": [1.0]})
        equity_curve = pd.Series([1.0, 1.01], name="equity")
        leverage_utilization = pd.Series([0.2, 0.3])

    sink = FillSink("FxSeatbelt", "rid", {}, root=tmp_path)
    fx_backtest._route_fills(Res(), sink, window=3)
    assert (sink.run_dir / "w03_trades.csv.gz").exists()
    assert (sink.run_dir / "w03_leverage_utilization.csv.gz").exists()


def test_route_result_to_sink_tags_cfg_hash(tmp_path):
    from src.backtesting.engine import fx_backtest

    class Res:
        trades = pd.DataFrame({"date": ["2011-01-03"], "pair": ["EURUSD"], "units": [1.0]})
        equity_curve = pd.Series([1.0, 1.01], name="equity")
        leverage_utilization = pd.Series([0.2, 0.3])

    sink = FillSink("FxSeatbelt", "rid", {}, root=tmp_path)
    fx_backtest._route_fills(Res(), sink, window=3, cfg_hash="c1x")
    assert (sink.run_dir / "w03_c1x_trades.csv.gz").exists()
    assert (sink.run_dir / "w03_c1x_leverage_utilization.csv.gz").exists()
