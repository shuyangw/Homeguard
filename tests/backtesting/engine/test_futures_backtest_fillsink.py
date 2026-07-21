import pandas as pd
from src.backtesting.engine.fill_sink import FillSink
from src.backtesting.engine import futures_backtest


def test_futures_route_fills_writes_margin_extra(tmp_path):
    class Res:
        trades = pd.DataFrame({"date": ["2017-01-03"], "symbol": ["CL"], "units": [1.0]})
        equity_curve = pd.Series([1.0, 1.02], name="equity")
        margin_utilization = pd.Series([0.4, 0.5])

    sink = FillSink("FuturesCarry", "rid", {}, root=tmp_path)
    futures_backtest._route_fills(Res(), sink, window=2)
    assert (sink.run_dir / "w02_trades.csv.gz").exists()
    assert (sink.run_dir / "w02_margin_utilization.csv.gz").exists()


def test_futures_route_fills_tags_cfg_hash(tmp_path):
    class Res:
        trades = pd.DataFrame({"date": ["2017-01-03"], "symbol": ["CL"], "units": [1.0]})
        equity_curve = pd.Series([1.0, 1.02], name="equity")
        margin_utilization = pd.Series([0.4, 0.5])

    sink = FillSink("FuturesCarry", "rid", {}, root=tmp_path)
    futures_backtest._route_fills(Res(), sink, window=2, cfg_hash="c1x")
    assert (sink.run_dir / "w02_c1x_trades.csv.gz").exists()
    assert (sink.run_dir / "w02_c1x_margin_utilization.csv.gz").exists()
