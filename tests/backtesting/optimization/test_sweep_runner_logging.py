import pandas as pd
from src.backtesting.engine.fill_sink import FillSink


def test_sweep_writes_per_symbol_via_sink(tmp_path):
    sink = FillSink("SweepDemo", "rid", {"kind": "sweep"}, root=tmp_path)

    class FakePortfolio:
        trades = [
            {"type": "entry", "timestamp": "2020-01-02", "price": 10.0, "shares": 5},
            {"type": "exit", "timestamp": "2020-01-05", "price": 11.0, "shares": 5,
             "pnl": 5.0, "pnl_pct": 0.1, "exit_reason": "target"},
        ]

    for sym in ("AAPL", "MSFT"):
        sink.write_portfolio(FakePortfolio(), window=0, cfg_hash=sym, symbol=sym)
    sink.finalize()
    assert (sink.run_dir / "w00_AAPL_trades.csv.gz").exists()
    assert (sink.run_dir / "w00_MSFT_trades.csv.gz").exists()
    assert (sink.run_dir / "manifest.csv").exists()
