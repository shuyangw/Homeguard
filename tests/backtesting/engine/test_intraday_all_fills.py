import importlib.util
import sys
from pathlib import Path

MOD = Path("scripts/backtest_scripts/run_fx_london_breakout_walkforward.py")


def _load():
    spec = importlib.util.spec_from_file_location("lb_runner", MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lb_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_trade_log_includes_exits():
    from src.backtesting.engine.intraday_order_engine import Fill
    import pandas as pd
    mod = _load()
    fills = [
        Fill("o1", pd.Timestamp("2011-01-21 09:30", tz="UTC"), 1.5, 1.0, "buy"),
        Fill("o1x", pd.Timestamp("2011-01-21 15:00", tz="UTC"), 1.6, 1.0, "sell"),
    ]
    df = mod.build_trade_log("GBPUSD", fills, day_r=0.9)
    assert len(df) == 2  # entry AND exit both present
    assert set(df["side"]) == {"buy", "sell"}
