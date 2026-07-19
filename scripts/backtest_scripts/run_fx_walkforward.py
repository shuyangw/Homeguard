"""Spot-FX walk-forward + statistical gate + readiness report.

FX trend and value are PARAMETER-FREE (fixed forecast scalars/speeds), so this
rolls non-overlapping OOS windows, runs run_fx_backtest once per window per cost
leg (1x and 1.5x), stitches the OOS-dated return series, and evaluates the
Sharpe/PSR/DSR/PBO gate. Trial count = 1.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
from src.backtesting.engine.fx_backtest import run_fx_backtest
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr
from src.backtesting.walkforward_common import (
    _annualized_sharpe,
    _as_date,
    _build_windows,
    _compute_pbo,
    _oos_returns,
    _verdict as _verdict_fx,
    get_campaign_trial_distribution,
)
from src.utils import logger

_DEFAULT_UNIVERSE = ["EURUSD", "USDJPY", "USDCHF", "EURJPY", "EURCHF", "CHFJPY", "XAUUSD", "XAGUSD"]
_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20
_REPORT_PATH = "docs/reports/fx/FX_WALK_FORWARD.md"


def _config_to_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract walk_forward_fx kwargs from an FX backtest YAML dict."""
    strat = config.get("strategy", {})
    dates = config.get("dates", {})
    bt = config.get("backtest", {})
    return {
        "universe": list(strat["universe"]),
        "capital": float(bt.get("initial_capital", _DEFAULT_CAPITAL)),
        "vol_target": float(bt.get("vol_target_per_instrument", _DEFAULT_VOL_TARGET)),
        "start": str(dates["start"]),
        "end": str(dates["end"]),
        "strategy_name": strat.get("name", "FxTrend"),
        "tier": bt.get("tier", "major"),
        "idm": bool(bt.get("idm", False)),
        "idm_cap": bt.get("idm_cap", None),
    }


def _run_window_fx(universe: Sequence[str], train_start: date, test_end: date,
                    capital: float, vol_target: float, cost_mult: float,
                    strategy_name: str, tier: str, idm: bool,
                    idm_cap: Optional[float]) -> Dict[str, Any]:
    config = {
        "asset_class": "fx",
        "strategy": {"name": strategy_name, "universe": list(universe), "params": {}},
        "dates": {"start": str(train_start), "end": str(test_end)},
        "backtest": {"initial_capital": capital, "vol_target_per_instrument": vol_target,
                     "rebalance": "weekly", "cost_mult": cost_mult, "leverage_cap": 10.0,
                     "tier": tier, "idm": idm, "idm_cap": idm_cap},
    }
    return run_fx_backtest(config, register=False, log_trades=False)


def process_window(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Top-level (picklable) per-window worker: runs every cost leg in
    spec["cost_mults"], register=False."""
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    try:
        panel = load_fx_daily_panel(universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[fx_walk_forward] skipping window {test_start}..{test_end}: {e}")
        return None
    window_universe = sorted({p for p, _ in panel.columns})
    dates = list(panel.index)
    oos_by_cost: Dict[float, np.ndarray] = {}
    for cost_mult in spec["cost_mults"]:
        res = _run_window_fx(window_universe, train_start, test_end, spec["capital"],
                              spec["vol_target"], cost_mult, spec["strategy_name"], spec["tier"],
                              spec["idm"], spec["idm_cap"])
        oos_by_cost[cost_mult] = _oos_returns(res["equity_curve"], dates, test_start)
    return {
        "train_start": train_start, "test_start": test_start, "test_end": test_end,
        "window_universe": window_universe,
        "oos_by_cost": oos_by_cost,
    }


def walk_forward_fx(
    train_months: int,
    test_months: int,
    step_months: int,
    start: str,
    end: str,
    universe: Optional[Sequence[str]] = None,
    capital: float = _DEFAULT_CAPITAL,
    vol_target: float = _DEFAULT_VOL_TARGET,
    strategy_name: str = "FxTrend",
    tier: str = "major",
    idm: bool = False,
    idm_cap: Optional[float] = None,
    max_workers: Optional[int] = None,
    cost_mults: Sequence[float] = (1.0, 1.5),
) -> Dict[str, Any]:
    """Roll OOS test windows for a parameter-free FX strategy (trend/value).

    Returns a dict with `oos_sharpe`, `psr`, `dsr`, `pbo`,
    `oos_sharpe_1_5x_cost`, `oos_sharpe_by_cost`, `n_windows`, `n_oos_days`,
    `window_sharpes`, `trial_count`, and `run_id` (registry append; None on
    failure). `oos_sharpe` and `oos_sharpe_1_5x_cost` are always the 1.0x and
    1.5x legs respectively (requires 1.0 and 1.5 to be present in
    `cost_mults`, the default). PSR/DSR/PBO/skew/kurtosis are always computed
    on the 1.0x leg only, regardless of what other legs are requested.
    """
    universe = list(universe) if universe is not None else list(_DEFAULT_UNIVERSE)
    start_d = _as_date(start)
    end_d = _as_date(end)
    cost_mults = list(cost_mults)

    windows = _build_windows(train_months, test_months, step_months, start_d, end_d)
    if len(windows) < 2:
        raise ValueError(
            f"walk-forward requires >=2 OOS windows, got {len(windows)} "
            f"for range {start}..{end} with train={train_months}m test={test_months}m step={step_months}m"
        )

    per_window_returns_by_cost: Dict[float, List[np.ndarray]] = {c: [] for c in cost_mults}
    window_sharpes: List[float] = []
    window_universes: List[List[str]] = []
    used_windows: List[tuple[date, date, date]] = []

    specs = [
        {"universe": universe, "train_start": ts, "test_start": tst, "test_end": te,
         "capital": capital, "vol_target": vol_target, "strategy_name": strategy_name,
         "tier": tier, "idm": idm, "idm_cap": idm_cap, "cost_mults": cost_mults}
        for (ts, tst, te) in windows
    ]
    from src.backtesting.parallel import parallel_map
    results = parallel_map(process_window, specs, max_workers=max_workers)
    for r in results:
        if r is None:
            continue
        for c in cost_mults:
            per_window_returns_by_cost[c].append(r["oos_by_cost"][c])
        window_sharpes.append(_annualized_sharpe(r["oos_by_cost"][1.0]))
        window_universes.append(r["window_universe"])
        used_windows.append((r["train_start"], r["test_start"], r["test_end"]))

    if len(used_windows) < 2:
        raise ValueError(
            f"walk-forward requires >=2 usable OOS windows after data-availability filtering, "
            f"got {len(used_windows)} for range {start}..{end}"
        )
    windows = used_windows

    stitched_by_cost: Dict[float, np.ndarray] = {
        c: np.concatenate(per_window_returns_by_cost[c]) for c in cost_mults
    }
    oos_sharpe_by_cost: Dict[float, float] = {
        c: _annualized_sharpe(stitched_by_cost[c]) for c in cost_mults
    }

    stitched_1x = stitched_by_cost[1.0]
    n = int(stitched_1x.size)
    oos_sharpe = oos_sharpe_by_cost[1.0]
    oos_sharpe_1_5x_cost = oos_sharpe_by_cost[1.5]

    series = pd.Series(stitched_1x)
    skew = float(series.skew()) if n > 2 else 0.0
    # pandas .kurtosis() is EXCESS kurtosis (normal = 0); psr/dsr want Pearson
    # kurtosis (normal = 3) per docs/methodology/backtesting.md Section 2.2.
    kurt = float(series.kurtosis()) + 3.0 if n > 3 else 3.0

    psr_val = psr(oos_sharpe, 0.0, n, skew, kurt)
    # Gate 0.1/0.2: deflate against the real, growing project-wide
    # trial-Sharpe distribution (mirrors gate_return_stream), not a
    # single-element list.
    n_trials, trial_sharpes = get_campaign_trial_distribution()
    dsr_val = dsr(oos_sharpe, trial_sharpes, n, skew, kurt,
                   n_trials_project=n_trials)
    pbo_val = _compute_pbo(per_window_returns_by_cost[1.0])

    result: Dict[str, Any] = {
        "oos_sharpe": oos_sharpe,
        "psr": psr_val,
        "dsr": dsr_val,
        "pbo": pbo_val,
        "oos_sharpe_1_5x_cost": oos_sharpe_1_5x_cost,
        "oos_sharpe_by_cost": oos_sharpe_by_cost,
        "n_windows": len(windows),
        "n_oos_days": n,
        "window_sharpes": window_sharpes,
        "trial_count": n_trials,
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
            agent_name="fx-harness-walkforward",
            metrics={
                k: v for k, v in result.items()
                if k not in ("window_sharpes", "universe", "window_universes")
            },
            asset_class="fx",
            data_frequency="daily",
            params={
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "universe": universe,
                "vol_target_per_instrument": vol_target,
                "initial_capital": capital,
                "tier": tier,
                "trial_count_project_wide": n_trials,
            },
            window_start=windows[0][1],
            window_end=windows[-1][2],
            phase="walk_forward",
        )
    except Exception as e:
        logger.error(f"[walk_forward_fx] registry append_run failed (non-fatal): {e}")

    result["run_id"] = run_id
    return result


def _write_readiness_report(result: Dict[str, Any], train_months: int, test_months: int,
                             step_months: int, start: str, end: str,
                             report_path: str = _REPORT_PATH) -> str:
    verdict = _verdict_fx(result)
    window_rows = "\n".join(
        f"| {i + 1} | {s:.4f} | {result['window_universes'][i]} |"
        for i, s in enumerate(result["window_sharpes"])
    )
    _title_display = {"FxTrend": "FX Trend", "FxValue": "FX Value"}
    _sname = result.get("strategy_name", "FxTrend")
    title = _title_display.get(_sname, _sname)
    content = f"""# {title} Walk-Forward Readiness Report

Generated by `scripts/backtest_scripts/run_fx_walkforward.py::main()`.

## Design

{title} is run PARAMETER-FREE: its constants are fixed doctrine, not fit to
data. Consequently this walk-forward performs NO parameter search. It rolls
non-overlapping OOS test windows
(train={train_months}m / test={test_months}m / step={step_months}m) across
{start} .. {end}, running `run_fx_backtest` once per window over
[train_start, test_end] (the train segment serves only as signal lookback
warm-up), keeping ONLY the OOS-dated (test_start..test_end) portion of each
window's equity curve, and stitching those OOS segments into one
concatenated OOS daily return series. The statistical gate
(Sharpe / PSR / DSR / PBO) is computed on that stitched series.

**Trial count = {result['trial_count']}.** {title} itself is a single
parameter-free configuration with no in-run parameter search, but per Gate 0
DSR is deflated using the real, growing PROJECT-WIDE trial-Sharpe
distribution across the whole futures/FX campaign
(`docs/methodology/backtesting.md` Section 2.3 / 9.4), sourced via
`src.backtesting.walkforward_common.get_campaign_trial_distribution()`.

Requested universe ({len(result['universe'])} pairs): {result['universe']}.
Initial capital: ${result['capital']:,.0f}. Vol target per instrument:
{result['vol_target']:.2f}. Rebalance: weekly.
Data frequency: daily spot FX.

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

| Window | OOS Sharpe | Pairs with data |
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
"""
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="FX trend/value walk-forward + gate")
    parser.add_argument("--config", default=None,
                        help="FX backtest YAML; drives universe/capital/vol-target/dates")
    parser.add_argument("--report", default=_REPORT_PATH,
                        help="Output readiness-report path (defaults to the baseline path)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Max worker processes for the per-window map (default: auto-parallel)")
    parser.add_argument("--json", default=None,
                        help="Optional path to dump the gate metrics as JSON")
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(Path(args.config).read_text())
        kw = _config_to_kwargs(cfg)
    else:
        kw = {"universe": list(_DEFAULT_UNIVERSE), "capital": _DEFAULT_CAPITAL,
              "vol_target": _DEFAULT_VOL_TARGET, "start": "2011-01-01", "end": "2025-12-31",
              "strategy_name": "FxTrend", "tier": "major", "idm": True, "idm_cap": 2.5}

    # Run-status logging survives a SIGKILL: if this run is killed, the status
    # file is frozen at RUNNING with the last heartbeat, so a stale RUNNING file
    # tells us it died (and roughly when) instead of leaving us to guess.
    from src.utils.run_status import RunStatus

    _meta = {"strategy": kw.get("strategy_name", "FxTrend"),
             "config": args.config, "jobs": args.jobs,
             "start": kw["start"], "end": kw["end"], "n_pairs": len(kw["universe"]),
             "idm": kw.get("idm", False), "idm_cap": kw.get("idm_cap")}
    with RunStatus("fx_walkforward", meta=_meta) as st:
        result = walk_forward_fx(
            train_months=args.train_months, test_months=args.test_months, step_months=args.step_months,
            start=kw["start"], end=kw["end"], universe=kw["universe"],
            capital=kw["capital"], vol_target=kw["vol_target"],
            strategy_name=kw.get("strategy_name", "FxTrend"),
            tier=kw.get("tier", "major"),
            idm=kw.get("idm", False),
            idm_cap=kw.get("idm_cap"),
            max_workers=args.jobs,
        )
        st.heartbeat(note=f"gate computed: oos_sharpe={result['oos_sharpe']:.4f} n_windows={result['n_windows']}")
        report_path = _write_readiness_report(
            result, train_months=args.train_months, test_months=args.test_months,
            step_months=args.step_months, start=kw["start"], end=kw["end"], report_path=args.report,
        )
    logger.info(
        f"[walk_forward_fx] wrote {report_path}; "
        f"oos_sharpe={result['oos_sharpe']:.4f} psr={result['psr']:.4f} "
        f"dsr={result['dsr']:.4f} pbo={result['pbo']} "
        f"oos_sharpe_1_5x_cost={result['oos_sharpe_1_5x_cost']:.4f} "
        f"n_windows={result['n_windows']}"
    )

    if args.json:
        import json
        keys = ("oos_sharpe", "psr", "dsr", "pbo", "oos_sharpe_1_5x_cost", "n_windows")
        Path(args.json).write_text(json.dumps({k: result[k] for k in keys}))


if __name__ == "__main__":
    main()
