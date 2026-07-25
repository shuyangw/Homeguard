"""Spread backtest assembly: wire a spread strategy's book through the
FxSpreadPortfolioSimulator to a daily return series.

run_spread_backtest loads the daily FX panel, builds the per-pair quote->USD
conversion panel, instantiates the named spread strategy, produces its per-date
spread book + trailing spread-vol map, runs the beta-weighted spread simulator,
and returns the equity curve's daily pct-change series. This is the assembly the
strategy-lead walk-forward calls per OOS window; the statistical gate is NOT run
here.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.backtesting.data.fx_backtest_loader import (
    build_quote_usd_panel,
    load_fx_daily_panel,
)
from src.backtesting.engine.fx_backtest import _cost_fn_factory
from src.backtesting.engine.fx_spread_simulator import FxSpreadPortfolioSimulator
from src.strategies.advanced.fx_audnzd_pairs import AudNzdPairs, AudNzdPairsKalman
from src.strategies.advanced.fx_coint_scanner import CointScanner
from src.strategies.advanced.fx_vol_ratio_pair import VolRatioPair
from src.utils import logger
from src.utils.run_status import RunStatus

_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_LEVERAGE_CAP = 4.0


def _conversion_legs(universe: list[str]) -> list[str]:
    # build_quote_usd_panel converts each pair's QUOTE currency to USD via a
    # {quote}USD or USD{quote} leg. Some traded universes (e.g. VolRatioPair's
    # EURNOK/EURSEK) omit those legs, so load them alongside the traded pairs;
    # load_fx_daily_panel skips any with no cache data. Only the traded pairs
    # drive the close panel and strategy -- these are conversion-only.
    legs: list[str] = []
    for pair in universe:
        quote = pair[3:]
        if quote != "USD":
            legs.extend([f"{quote}USD", f"USD{quote}"])
    return legs


_VOL_RATIO_DECLARED_SETS = (("EURNOK", "EURSEK"), ("AUDUSD", "NZDUSD"),
                            ("XAUUSD", "XAGUSD"))


def _vol_ratio_coupled_sets(present: list[str]) -> tuple:
    present_set = set(present)
    surviving = []
    for leg_a, leg_b in _VOL_RATIO_DECLARED_SETS:
        if leg_a in present_set and leg_b in present_set:
            surviving.append((leg_a, leg_b))
        else:
            logger.warning(f"[fx_spread_backtest] VolRatioPair dropping coupled "
                           f"set ({leg_a}, {leg_b}): leg absent from cache")
    return tuple(surviving)


def _make_strategy(strategy_name: str, universe: list[str]):
    if strategy_name == "AudNzdPairs":
        return AudNzdPairs()
    if strategy_name == "AudNzdPairsKalman":
        return AudNzdPairsKalman()
    if strategy_name == "CointScanner":
        return CointScanner(universe)
    if strategy_name == "VolRatioPair":
        return VolRatioPair(coupled_sets=_vol_ratio_coupled_sets(universe))
    raise ValueError(f"unknown spread strategy: {strategy_name}")


def _write_trade_log(res, strategy_name: str, start: date, end: date) -> str:
    out = Path("output") / "backtests" / "fx" / strategy_name / f"{start}_to_{end}"
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity_curve.rename("equity").to_frame().to_csv(out / "equity.csv", index_label="date")
    logger.info(f"[fx_spread_backtest] wrote trade log ({len(res.trades)} fills) to {out}")
    return str(out)


def run_spread_backtest(strategy_name: str, universe: list[str], start: date,
                        end: date, vol_target: float = 0.10,
                        rebalance: str = "weekly",
                        initial_capital: float = _DEFAULT_CAPITAL,
                        leverage_cap: float = _DEFAULT_LEVERAGE_CAP,
                        cost_mult: float = 1.0,
                        log_trades: bool = False) -> pd.Series:
    with RunStatus("fx_spread_backtest",
                   meta={"strategy": strategy_name, "start": str(start),
                         "end": str(end), "rebalance": rebalance,
                         "cost_mult": cost_mult}):
        load_universe = list(dict.fromkeys(universe + _conversion_legs(universe)))
        panel = load_fx_daily_panel(load_universe, start, end)
        panel_pairs = {c[0] for c in panel.columns}
        present = [p for p in universe if p in panel_pairs]
        close = panel.xs("close", axis=1, level=1)[present]
        quote_usd = build_quote_usd_panel(panel, present)

        strategy = _make_strategy(strategy_name, present)
        book, sigma = strategy.spread_book(close)

        sim = FxSpreadPortfolioSimulator(initial_capital, _cost_fn_factory(),
                                         rebalance=rebalance,
                                         cost_mult=cost_mult,
                                         leverage_cap=leverage_cap)
        res = sim.run_spreads(close, book, sigma, quote_usd, vol_target)
        if log_trades:
            _write_trade_log(res, strategy_name, start, end)
    return res.equity_curve.pct_change(fill_method=None).dropna()


def _as_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an FX spread backtest.")
    parser.add_argument("--strategy", required=True,
                        choices=["AudNzdPairs", "AudNzdPairsKalman", "CointScanner", "VolRatioPair"])
    parser.add_argument("--universe", required=True,
                        help="comma-separated pair list")
    parser.add_argument("--start", required=True, type=_as_date)
    parser.add_argument("--end", required=True, type=_as_date)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--rebalance", default="weekly")
    parser.add_argument("--log-trades", action="store_true",
                        help="persist trades.csv/equity.csv under output/backtests/fx/")
    args = parser.parse_args()

    universe = [p.strip() for p in args.universe.split(",") if p.strip()]
    returns = run_spread_backtest(args.strategy, universe, args.start, args.end,
                                  vol_target=args.vol_target,
                                  rebalance=args.rebalance,
                                  log_trades=args.log_trades)
    logger.info(f"[fx_spread_backtest] {args.strategy}: {len(returns)} daily "
                f"returns, cumulative={float((1 + returns).prod() - 1):.4f}")


if __name__ == "__main__":
    main()
