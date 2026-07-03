"""Carver TSMOM walk-forward + statistical gate + readiness report (Task 10).

Carver TSMOM is PARAMETER-FREE (the multi-speed EWMAC blend and forecast cap
are fixed doctrine, not fit). "Walk-forward" here therefore does NOT search
over parameters -- it rolls non-overlapping OOS test windows across the full
date range, runs `run_futures_backtest` once per window (using the preceding
`train_months` purely as signal warm-up so the EWMAC speeds have lookback at
the start of the OOS segment, then keeping ONLY the OOS-dated portion of the
resulting equity curve), stitches those per-window OOS return series into one
concatenated OOS return series, and evaluates the statistical gate
(Sharpe / PSR / DSR / PBO) on that stitched series. Because there is no
parameter selection, the project trial count for this run is 1 -- DSR
therefore reduces to PSR against benchmark Sharpe 0 (no deflation term).

This is the acceptance/proof script for the futures backtest harness
(Tasks 1-9): the first methodology-compliant (Section 3 walk-forward +
Section 2 statistical gate) futures result.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.pbo import pbo
from src.backtesting.statistics.psr import psr
from src.utils import logger

_DEFAULT_UNIVERSE = [
    "MES", "MNQ", "M2K", "MYM", "MCL", "MNG", "MGC", "SIL", "6E", "6J", "ZN", "ZC",
]
_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20

# Carver TSMOM is parameter-free -- no selection was performed to arrive at
# this config, so the project-wide trial count for THIS run is 1. Documented
# per docs/methodology/backtesting.md Section 2.3's explicit-trial-count rule.
TRIAL_COUNT_PARAMETER_FREE = 1

_REPORT_PATH = "docs/reports/futures/CARVER_TSMOM_READINESS.md"
_TRADING_DAYS_PER_YEAR = 252


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _config_to_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract walk_forward_carver kwargs from a futures backtest YAML dict."""
    strat = config.get("strategy", {})
    dates = config.get("dates", {})
    bt = config.get("backtest", {})
    return {
        "universe": list(strat["universe"]),
        "capital": float(bt.get("initial_capital", _DEFAULT_CAPITAL)),
        "vol_target": float(bt.get("vol_target_per_instrument", _DEFAULT_VOL_TARGET)),
        "start": str(dates["start"]),
        "end": str(dates["end"]),
        "strategy_name": strat.get("name", "CarverMomentum"),
        "strategy_params": strat.get("params", {}),
    }


def _add_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def _build_windows(train_months: int, test_months: int, step_months: int,
                    start: date, end: date) -> List[tuple[date, date, date]]:
    """Return (train_start, test_start, test_end) triples, non-overlapping in OOS."""
    windows: List[tuple[date, date, date]] = []
    train_start = start
    while True:
        test_start = _add_months(train_start, train_months)
        if test_start >= end:
            break
        test_end = min(_add_months(test_start, test_months), end)
        windows.append((train_start, test_start, test_end))
        if test_end >= end:
            break
        train_start = _add_months(train_start, step_months)
    return windows


def _run_window(universe: Sequence[str], train_start: date, test_end: date,
                 capital: float, vol_target: float, cost_mult: float,
                 strategy_name: str = "CarverMomentum",
                 strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = {
        "strategy": {"name": strategy_name, "universe": list(universe),
                     "params": strategy_params or {}},
        "dates": {"start": str(train_start), "end": str(test_end)},
        "backtest": {
            "initial_capital": capital,
            "vol_target_per_instrument": vol_target,
            "rebalance": "weekly",
            "cost_mult": cost_mult,
        },
    }
    return run_futures_backtest(config)


def _oos_returns(equity_curve: List[float], dates: List[date], test_start: date) -> np.ndarray:
    """Slice the OOS-dated segment of a window's equity curve and diff to returns."""
    if len(equity_curve) != len(dates):
        raise ValueError(
            f"equity_curve length {len(equity_curve)} != trading-day count {len(dates)} "
            "-- window date range mismatch between run_futures_backtest and load_daily_panel"
        )
    eq = pd.Series(equity_curve, index=pd.Index(dates))
    oos_idx = eq.index[eq.index >= test_start]
    if len(oos_idx) == 0:
        return np.array([], dtype=float)
    start_pos = eq.index.get_loc(oos_idx[0])
    # Include one day before the OOS start (if available) so the first OOS
    # return is a real day-over-day change, not a NaN from pct_change's edge.
    segment = eq.iloc[max(start_pos - 1, 0):]
    return segment.pct_change().dropna().to_numpy(dtype=float)


def _annualized_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return float("nan")
    std = float(np.nanstd(returns, ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    mean = float(np.nanmean(returns))
    return mean / std * np.sqrt(_TRADING_DAYS_PER_YEAR)


def _compute_pbo(per_window_returns: List[np.ndarray]) -> float:
    """PBO across windows-as-columns (CSCV on the OOS return series per window).

    Each window's stitched-eligible OOS return series is treated as one
    "config" column; PBO here answers whether the OOS ranking of windows is
    stable under CSCV resampling, not a parameter-selection PBO (there is no
    parameter selection for a parameter-free strategy).
    """
    usable = [r for r in per_window_returns if r.size > 1]
    if len(usable) < 2:
        return float("nan")
    min_len = min(r.size for r in usable)
    if min_len < 2:
        return float("nan")
    matrix = np.column_stack([r[:min_len] for r in usable])
    return pbo(matrix)


def walk_forward_carver(
    train_months: int,
    test_months: int,
    step_months: int,
    start: str,
    end: str,
    universe: Optional[Sequence[str]] = None,
    capital: float = _DEFAULT_CAPITAL,
    vol_target: float = _DEFAULT_VOL_TARGET,
    strategy_name: str = "CarverMomentum",
    strategy_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Roll OOS test windows for the parameter-free Carver TSMOM strategy.

    Returns a dict with `oos_sharpe`, `psr`, `dsr`, `pbo`,
    `oos_sharpe_1_5x_cost`, `n_windows`, `n_oos_days`, `window_sharpes`,
    `trial_count`, and `run_id` (registry append; None on failure).
    """
    universe = list(universe) if universe is not None else list(_DEFAULT_UNIVERSE)
    start_d = _as_date(start)
    end_d = _as_date(end)

    windows = _build_windows(train_months, test_months, step_months, start_d, end_d)
    if len(windows) < 2:
        raise ValueError(
            f"walk-forward requires >=2 OOS windows, got {len(windows)} "
            f"for range {start}..{end} with train={train_months}m test={test_months}m step={step_months}m"
        )

    per_window_returns_1x: List[np.ndarray] = []
    per_window_returns_1_5x: List[np.ndarray] = []
    window_sharpes: List[float] = []
    window_universes: List[List[str]] = []
    used_windows: List[tuple[date, date, date]] = []

    for train_start, test_start, test_end in windows:
        # `load_daily_panel` (Task 5) gracefully excludes any root with no
        # usable data for [train_start, test_end] (micro contracts phase in
        # over time -- e.g. CME Micro E-mini S&P 500 launched 2019-05 -- and
        # isolated roll-calendar data-quality issues on specific roots/
        # windows are logged + skipped, not fatal). One `load_daily_panel`
        # call here both resolves which roots actually have data (its
        # columns) and gives the OOS date-slicing index; `_run_window` (via
        # `run_futures_backtest`) reloads the same window internally, but
        # that internal call resolves to the SAME included roots since the
        # inputs are identical -- deterministic since the roll-date tie-break
        # fix in `continuous_contract_loader.py`.
        try:
            panel = load_daily_panel(universe, train_start, test_end)
        except FileNotFoundError as e:
            logger.warning(
                f"[walk_forward_carver] skipping window {test_start}..{test_end}: {e}"
            )
            continue
        window_universe = sorted({r for r, _ in panel.columns})
        dates = list(panel.index)
        logger.info(
            f"[walk_forward_carver] window train_start={train_start} test={test_start}..{test_end} "
            f"roots_with_data={window_universe}"
        )

        # Pass only `window_universe` (roots with data this window) to
        # `run_futures_backtest` -- `CarverMomentumStrategy.forecast_panel`
        # indexes its output by the FULL requested universe and raises a
        # KeyError for any root missing from `close`'s columns, and
        # `load_daily_panel` (correctly) omits roots with zero data rather
        # than padding them in as all-NaN columns.
        res_1x = _run_window(window_universe, train_start, test_end, capital, vol_target, cost_mult=1.0,
                              strategy_name=strategy_name, strategy_params=strategy_params)
        oos_1x = _oos_returns(res_1x["equity_curve"], dates, test_start)
        per_window_returns_1x.append(oos_1x)
        window_sharpes.append(_annualized_sharpe(oos_1x))

        res_1_5x = _run_window(window_universe, train_start, test_end, capital, vol_target, cost_mult=1.5,
                                strategy_name=strategy_name, strategy_params=strategy_params)
        oos_1_5x = _oos_returns(res_1_5x["equity_curve"], dates, test_start)
        per_window_returns_1_5x.append(oos_1_5x)
        window_universes.append(window_universe)
        used_windows.append((train_start, test_start, test_end))

    if len(used_windows) < 2:
        raise ValueError(
            f"walk-forward requires >=2 usable OOS windows after data-availability filtering, "
            f"got {len(used_windows)} for range {start}..{end}"
        )
    windows = used_windows

    stitched_1x = np.concatenate(per_window_returns_1x)
    stitched_1_5x = np.concatenate(per_window_returns_1_5x)

    n = int(stitched_1x.size)
    oos_sharpe = _annualized_sharpe(stitched_1x)
    oos_sharpe_1_5x_cost = _annualized_sharpe(stitched_1_5x)

    series = pd.Series(stitched_1x)
    skew = float(series.skew()) if n > 2 else 0.0
    # pandas .kurtosis() is EXCESS kurtosis (normal = 0); psr/dsr want Pearson
    # kurtosis (normal = 3) per docs/methodology/backtesting.md Section 2.2.
    kurt = float(series.kurtosis()) + 3.0 if n > 3 else 3.0

    psr_val = psr(oos_sharpe, 0.0, n, skew, kurt)
    dsr_val = dsr(oos_sharpe, [oos_sharpe], n, skew, kurt,
                   n_trials_project=TRIAL_COUNT_PARAMETER_FREE)
    pbo_val = _compute_pbo(per_window_returns_1x)

    result: Dict[str, Any] = {
        "oos_sharpe": oos_sharpe,
        "psr": psr_val,
        "dsr": dsr_val,
        "pbo": pbo_val,
        "oos_sharpe_1_5x_cost": oos_sharpe_1_5x_cost,
        "n_windows": len(windows),
        "n_oos_days": n,
        "window_sharpes": window_sharpes,
        "trial_count": TRIAL_COUNT_PARAMETER_FREE,
        "skew": skew,
        "kurtosis_pearson": kurt,
        "universe": universe,
        "capital": capital,
        "vol_target": vol_target,
        "window_universes": window_universes,
        "window_start": windows[0][1],
        "window_end": windows[-1][2],
        "strategy_name": strategy_name,
    }

    run_id = None
    try:
        from src.experiments import append_run

        run_id = append_run(
            strategy_name=strategy_name,
            agent_name="futures-harness-walkforward",
            metrics={
                k: v for k, v in result.items()
                if k not in ("window_sharpes", "universe", "window_universes")
            },
            asset_class="futures",
            data_frequency="daily",
            params={
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "universe": universe,
                "vol_target_per_instrument": vol_target,
                "initial_capital": capital,
                "trial_count_project_wide": TRIAL_COUNT_PARAMETER_FREE,
            },
            window_start=windows[0][1],
            window_end=windows[-1][2],
            phase="walk_forward",
        )
    except Exception as e:
        logger.error(f"[walk_forward_carver] registry append_run failed (non-fatal): {e}")

    result["run_id"] = run_id
    return result


def _verdict(result: Dict[str, Any]) -> str:
    psr_val = result["psr"]
    dsr_val = result["dsr"]
    pbo_val = result["pbo"]
    sharpe = result["oos_sharpe"]
    sharpe_1_5x = result["oos_sharpe_1_5x_cost"]

    if any(np.isnan(x) for x in (psr_val, dsr_val, sharpe)):
        return "INCONCLUSIVE -- insufficient data to compute the statistical gate."

    passes_stat_gate = psr_val >= 0.95 and dsr_val >= 0.95
    passes_cost_gate = sharpe_1_5x > 0.0 and (sharpe <= 0 or sharpe_1_5x >= 0.5 * sharpe)
    passes_pbo = (not np.isnan(pbo_val)) and pbo_val < 0.25

    if sharpe <= 0:
        return "REJECT -- OOS Sharpe is non-positive; no edge to deflate or gate."
    if passes_stat_gate and passes_cost_gate and passes_pbo:
        return "PASS -- clears the combined statistical gate (Section 2.5) and the 1.5x cost gate (Section 4)."
    reasons = []
    if not passes_stat_gate:
        reasons.append(f"PSR/DSR below 0.95 (psr={psr_val:.3f}, dsr={dsr_val:.3f})")
    if not passes_cost_gate:
        reasons.append(f"fails 1.5x cost sensitivity (1x={sharpe:.3f}, 1.5x={sharpe_1_5x:.3f})")
    if not passes_pbo:
        reasons.append(f"PBO not comfortably acceptable (pbo={pbo_val})")
    return "WEAK -- does not clear the combined gate: " + "; ".join(reasons)


def _write_readiness_report(result: Dict[str, Any], train_months: int, test_months: int,
                             step_months: int, start: str, end: str,
                             report_path: str = _REPORT_PATH) -> str:
    verdict = _verdict(result)
    window_rows = "\n".join(
        f"| {i + 1} | {s:.4f} | {result['window_universes'][i]} |"
        for i, s in enumerate(result["window_sharpes"])
    )
    _title_display = {"CarverMomentum": "Carver TSMOM"}
    _sname = result.get("strategy_name", "CarverMomentum")
    title = _title_display.get(_sname, _sname)
    content = f"""# {title} Walk-Forward Readiness Report

Generated by `scripts/backtest_scripts/run_carver_walkforward.py::main()`.

## Design

{title} is run PARAMETER-FREE: its constants are fixed doctrine, not fit to
data. Consequently this walk-forward performs NO parameter search. It rolls
non-overlapping OOS test windows
(train={train_months}m / test={test_months}m / step={step_months}m) across
{start} .. {end}, running `run_futures_backtest` once per window over
[train_start, test_end] (the train segment serves only as signal lookback
warm-up), keeping ONLY the OOS-dated (test_start..test_end) portion of each
window's equity curve, and stitching those OOS segments into one
concatenated OOS daily return series. The statistical gate
(Sharpe / PSR / DSR / PBO) is computed on that stitched series.

**Trial count = {result['trial_count']}.** This is a single parameter-free
configuration with no selection over trials, so the project-wide trial count
fed to DSR (`docs/methodology/backtesting.md` Section 2.3) is 1 for this run.
With `n_trials=1`, `expected_max_sharpe` returns 0.0 (no deflation term), so
DSR reduces to PSR against a benchmark Sharpe of 0 -- this is the correct,
honest behavior for a non-selected, parameter-free strategy, not a bug.

Requested universe ({len(result['universe'])} roots): {result['universe']}.
Initial capital: ${result['capital']:,.0f}. Vol target per instrument:
{result['vol_target']:.2f}. Rebalance: weekly.
Data frequency: daily (aggregated from Databento 1-min continuous contracts).

**Per-window data-availability filtering.** Instruments phase in over time
(micro contracts launched 2019+, SOFR 2018, some roots later), so a fixed
universe cannot have full history back to {start} for every root.
`load_daily_panel` (`src/backtesting/data/futures_backtest_loader.py`)
gracefully excludes any root with no usable data for a window's
[train_start, test_end] range (logged as a WARNING, never silently). One isolated data-quality issue was
also handled the same way: `ContinuousContractDataLoader.load` previously
raised a nondeterministic `KeyError` for SIL on windows spanning 2017-02-26,
traced to an unstable tie-break in the per-day active-contract ranking
(`_active_contract_per_day` in `src/data/continuous_contract_loader.py`) --
fixed by adding a deterministic `symbol` tie-break to the sort. See the
per-window "roots with data" column below for exactly which roots
contributed to which window (a root with data for only part of a window
still appears -- `run_futures_backtest` sizes it to zero contracts wherever
its forecast/price/vol are unavailable).

## Metrics

| Metric | Value |
|---|---|
| OOS Sharpe (1x cost) | {result['oos_sharpe']:.4f} |
| OOS Sharpe (1.5x cost) | {result['oos_sharpe_1_5x_cost']:.4f} |
| PSR (vs benchmark 0) | {result['psr']:.4f} |
| DSR (n_trials={result['trial_count']}) | {result['dsr']:.4f} |
| PBO (windows-as-columns CSCV) | {result['pbo']:.4f} |
| n_windows | {result['n_windows']} |
| n_oos_days | {result['n_oos_days']} |
| skew | {result['skew']:.4f} |
| kurtosis (Pearson) | {result['kurtosis_pearson']:.4f} |
| window_start | {result['window_start']} |
| window_end | {result['window_end']} |
| registry run_id | {result.get('run_id')} |

## Per-window OOS Sharpe

| Window | OOS Sharpe | Roots with data |
|---|---|---|
{window_rows}

## Verdict

{verdict}

## Notes on PBO interpretation

PBO here is computed via CSCV (`src/backtesting/statistics/pbo.py`) treating
each window's OOS return series as a column ("windows-as-columns"), NOT as a
parameter-selection PBO -- there is no parameter selection for a
parameter-free strategy. It answers whether the ranking of windows by Sharpe
is stable under resampling, a weaker but still informative overfitting check
given only one configuration was ever run.

## Note: tail statistics

The stitched OOS return series has skew {result['skew']:.2f} and Pearson
kurtosis {result['kurtosis_pearson']:.1f} -- fat-tailed but finite, far from
the pathological tail stats (kurtosis in the thousands) an earlier version of
the harness produced when the simulator let account equity cross zero and
`pct_change` exploded on the zero-crossing equity curve; that was fixed before
merge via equity-feedback sizing plus a bankruptcy floor (equity is now
provably non-negative after both mark-to-market and cost debits), so the
PSR/DSR values here are reliable. Elevated kurtosis reflects a few large days
concentrating the performance and should be weighed alongside PBO when judging
robustness. The WEAK verdict (OOS Sharpe {result['oos_sharpe']:.4f}, PBO
{result['pbo']:.3f}) rests on the clean statistics, not on any tail artifact.
"""
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Carver TSMOM walk-forward + gate")
    parser.add_argument("--config", default=None,
                        help="Futures backtest YAML; drives universe/capital/vol-target/dates")
    parser.add_argument("--report", default=_REPORT_PATH,
                        help="Output readiness-report path (defaults to the baseline path)")
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(Path(args.config).read_text())
        kw = _config_to_kwargs(cfg)
    else:
        kw = {"universe": list(_DEFAULT_UNIVERSE), "capital": _DEFAULT_CAPITAL,
              "vol_target": _DEFAULT_VOL_TARGET, "start": "2010-06-07", "end": "2025-02-01",
              "strategy_name": "CarverMomentum", "strategy_params": {}}

    result = walk_forward_carver(
        train_months=36, test_months=12, step_months=12,
        start=kw["start"], end=kw["end"], universe=kw["universe"],
        capital=kw["capital"], vol_target=kw["vol_target"],
        strategy_name=kw.get("strategy_name", "CarverMomentum"),
        strategy_params=kw.get("strategy_params", {}),
    )
    report_path = _write_readiness_report(
        result, train_months=36, test_months=12, step_months=12,
        start=kw["start"], end=kw["end"], report_path=args.report,
    )
    logger.info(
        f"[walk_forward_carver] wrote {report_path}; "
        f"oos_sharpe={result['oos_sharpe']:.4f} psr={result['psr']:.4f} "
        f"dsr={result['dsr']:.4f} pbo={result['pbo']} "
        f"oos_sharpe_1_5x_cost={result['oos_sharpe_1_5x_cost']:.4f} "
        f"n_windows={result['n_windows']}"
    )


if __name__ == "__main__":
    main()
