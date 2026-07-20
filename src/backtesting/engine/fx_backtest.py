"""Config-driven spot-FX backtest orchestration.

Assembles: daily FX panel -> USD-conversion + rate-diff panels -> strategy
forecast -> close-to-close vol -> FxSpotPortfolioSimulator (carry-accruing) ->
standard report -> experiment registry. The FX counterpart to
futures_backtest.py; kept separate because spot-FX PnL/carry/notional math does
not fit the futures contract/margin abstractions.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

from src.backtesting.costs.fx import fx_round_trip_usd
from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel, build_quote_usd_panel
from src.backtesting.engine.fx_spot_portfolio_simulator import FxSpotPortfolioSimulator
from src.backtesting.reporting.standard_report import StandardReportGenerator
from src.backtesting.utils.idm_weights import compute_div_mult
from src.data.fx.clusters import fx_cluster_for
from src.data.fx_rates import load_fx_rate_panel, build_rate_diff_panel, currencies_for_pairs
from src.features.volatility import close_to_close_rv
from src.strategies.registry import get_strategy_class
from src.utils import logger

_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20
_DEFAULT_REBALANCE = "weekly"
_DEFAULT_LEVERAGE_CAP = 10.0


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


_METALS_BASES = {"XAU", "XAG"}


def _tier_for_pair(pair: str) -> str:
    if pair[:3] in _METALS_BASES or pair[3:] in _METALS_BASES:
        return "major"  # metals use the bps path; tier is irrelevant
    if "USD" in (pair[:3], pair[3:]):
        return "major"
    return "minor"


def _cost_fn_factory(session: str = "ny"):
    def cost_fn(pair, units_traded, price, quote_to_usd):
        return fx_round_trip_usd(pair, units_traded, price, quote_to_usd,
                                 tier=_tier_for_pair(pair), session=session)
    return cost_fn


def _route_fills(res, fill_sink, window, cfg_hash=None):
    extras = {"leverage_utilization": res.leverage_utilization.rename(
        "leverage_utilization").reset_index()}
    fill_sink.write_window(res.trades, window, cfg_hash=cfg_hash, extras=extras)


def run_fx_backtest(config: Dict[str, Any], register: bool = True,
                    log_trades: bool = False, fill_sink=None,
                    window=None, fill_cfg_hash=None) -> Dict[str, Any]:
    strat_cfg = config.get("strategy", {})
    dates_cfg = config.get("dates", {})
    bt = config.get("backtest", {})

    universe = list(strat_cfg["universe"])
    start = _as_date(dates_cfg["start"])
    end = _as_date(dates_cfg["end"])
    capital = float(bt.get("initial_capital", _DEFAULT_CAPITAL))
    vol_target = float(bt.get("vol_target_per_instrument", _DEFAULT_VOL_TARGET))
    rebalance = bt.get("rebalance", _DEFAULT_REBALANCE)
    cost_mult = float(bt.get("cost_mult", 1.0))
    leverage_cap = float(bt.get("leverage_cap", _DEFAULT_LEVERAGE_CAP))
    use_idm = bool(bt.get("idm", False))
    idm_cap = bt.get("idm_cap", None)

    strategy_name = strat_cfg.get("name", "FxTrend")

    panel = load_fx_daily_panel(universe, start, end)
    present = [p for p in universe if p in {c[0] for c in panel.columns}]
    close = panel.xs("close", axis=1, level=1)[present]

    strategy = get_strategy_class(strategy_name)(present, **strat_cfg.get("params", {}))

    quote_usd = build_quote_usd_panel(panel, present)
    rate_panel = load_fx_rate_panel(currencies_for_pairs(present), close.index)
    rate_diff = build_rate_diff_panel(present, rate_panel)

    forecasts = strategy.forecast_panel(close)[present]
    returns = close.pct_change(fill_method=None)
    daily_vol = returns.apply(lambda col: close_to_close_rv(col, 25, annualization_factor=1), axis=0)

    div_mult = compute_div_mult(present, per_instrument_cap=idm_cap,
                                cluster_fn=fx_cluster_for) if use_idm else 1.0

    sim = FxSpotPortfolioSimulator(capital, _cost_fn_factory(), rebalance=rebalance,
                                   cost_mult=cost_mult, leverage_cap=leverage_cap)
    res = sim.run_sized(close, forecasts, daily_vol, vol_target, quote_usd, rate_diff, div_mult)

    report = StandardReportGenerator().generate_report(
        res.equity_curve, strategy_name, present, str(start), str(end), capital)

    run_id = None
    if register:
        try:
            from src.experiments import append_run
            run_id = append_run(
                strategy_name=strategy_name, agent_name="fx-harness",
                metrics=report["overall_metrics"], asset_class="fx",
                data_frequency="daily", params=config,
                window_start=start, window_end=end)
        except Exception as e:
            logger.error(f"[fx_backtest] registry append_run failed (non-fatal): {e}")

    if fill_sink is not None:
        _route_fills(res, fill_sink, window if window is not None else 0, fill_cfg_hash)
        trade_log_dir = str(fill_sink.run_dir)
    else:
        trade_log_dir = _write_trade_log(res, strategy_name, start, end) if log_trades else None
    return {
        "n_days": len(res.equity_curve),
        "metrics": report["overall_metrics"],
        "equity_curve": res.equity_curve.tolist(),
        "run_id": run_id,
        "trade_log_dir": trade_log_dir,
    }


def _write_trade_log(res, strategy_name: str, start, end) -> str:
    out = Path("output") / "backtests" / "fx" / strategy_name / f"{start}_to_{end}"
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity_curve.rename("equity").to_frame().to_csv(out / "equity.csv", index_label="date")
    res.leverage_utilization.rename("leverage_utilization").to_frame().to_csv(
        out / "leverage_utilization.csv", index_label="date")
    logger.info(f"[fx_backtest] wrote trade log ({len(res.trades)} fills) to {out}")
    return str(out)
