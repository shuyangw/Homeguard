"""Pillar correlation: run single-shot full-period backtests of two configs and
correlate their daily returns on common dates. Supplementary to the standalone
walk-forward Sharpe (the OOS metric used for the inclusion bar).

`run_futures_backtest` returns `equity_curve` as a plain list (via
`res.equity_curve.tolist()`), which drops the date index -- there is no
`dates` key in that dict. To recover per-day dates, `_run` calls it with
`log_trades=True` (writing `output/backtests/futures/<strategy>/<start>_to_<end>/
equity.csv`, which retains the date index) and reads dates back from that CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.utils.logger import get_logger


def daily_return_correlation(equity_a, equity_b, dates_a, dates_b) -> float:
    sa = pd.Series(equity_a, index=pd.DatetimeIndex(dates_a)).pct_change()
    sb = pd.Series(equity_b, index=pd.DatetimeIndex(dates_b)).pct_change()
    joined = pd.concat([sa, sb], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _run(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    res = run_futures_backtest(cfg, register=False, log_trades=True)
    equity_df = pd.read_csv(Path(res["trade_log_dir"]) / "equity.csv", parse_dates=["date"])
    dates = [d.date() for d in equity_df["date"]]
    return equity_df["equity"].tolist(), dates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    args = p.parse_args()
    ea, da = _run(args.a)
    eb, db = _run(args.b)
    rho = daily_return_correlation(ea, eb, da, db)
    get_logger(__name__).info(f"[pillar_correlation] rho({args.a} , {args.b}) = {rho:.4f}")


if __name__ == "__main__":
    main()
