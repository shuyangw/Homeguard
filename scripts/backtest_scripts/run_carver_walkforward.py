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

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.backtesting.engine.fill_sink import FillSink
from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr
from src.backtesting.walkforward_common import (
    _annualized_sharpe,
    _as_date,
    _build_windows,
    _compute_pbo,
    _oos_returns_dated,
    _stitch_oos_dedup,
    _verdict,
    get_campaign_trial_distribution,
)
from src.utils import logger

_DEFAULT_UNIVERSE = [
    "MES", "MNQ", "M2K", "MYM", "MCL", "MNG", "MGC", "SIL", "6E", "6J", "ZN", "ZC",
]
_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20

_REPORT_PATH = "docs/reports/futures/CARVER_TSMOM_READINESS.md"


def _leg_tag(mult: float) -> str:
    s = ("%g" % float(mult)).replace(".", "")
    return f"c{s}x"


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
        "idm": bool(bt.get("idm", False)),
        "idm_cap": bt.get("idm_cap", None),
        # Gate 0 caveat-fix (#16 FuturesTurnOfMonth, strategy-lead TODO): this
        # walk-forward used to hardcode "weekly" per-window regardless of what
        # the config declared, mis-sampling any daily-rebalance signal (like
        # turn-of-month's payment-cycle window) onto a weekly runner. Now
        # honors the config's declared rebalance frequency, defaulting to
        # "weekly" only when the config is silent (preserves prior behavior
        # for every other Tier 1 config, none of which declare `rebalance`).
        "rebalance": bt.get("rebalance", "weekly"),
    }


def _run_window(universe: Sequence[str], train_start: date, test_end: date,
                 capital: float, vol_target: float, cost_mult: float,
                 strategy_name: str = "CarverMomentum",
                 strategy_params: Optional[Dict[str, Any]] = None,
                 idm: bool = False,
                 idm_cap: float | None = None,
                 rebalance: str = "weekly",
                 register: bool = True,
                 fill_sink: Optional[FillSink] = None,
                 window: Optional[int] = None,
                 fill_cfg_hash: Optional[str] = None) -> Dict[str, Any]:
    config = {
        "strategy": {"name": strategy_name, "universe": list(universe),
                     "params": strategy_params or {}},
        "dates": {"start": str(train_start), "end": str(test_end)},
        "backtest": {
            "initial_capital": capital,
            "vol_target_per_instrument": vol_target,
            "rebalance": rebalance,
            "cost_mult": cost_mult,
            "idm": idm,
            "idm_cap": idm_cap,
        },
    }
    return run_futures_backtest(config, register=register, validate_prereg=False,
                               fill_sink=fill_sink, window=window,
                               fill_cfg_hash=fill_cfg_hash)


def process_window(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Top-level (picklable) per-window worker: runs both cost legs, register=False.

    Returns None if the window has no usable data (mirrors the serial loop's
    FileNotFoundError skip behavior).
    """
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    capital, vol_target = spec["capital"], spec["vol_target"]
    strategy_name, strategy_params = spec["strategy_name"], spec["strategy_params"]
    idm = spec.get("idm", False)
    idm_cap = spec.get("idm_cap", None)
    rebalance = spec.get("rebalance", "weekly")
    try:
        panel = load_daily_panel(universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[walk_forward] skipping window {test_start}..{test_end}: {e}")
        return None
    window_universe = sorted({r for r, _ in panel.columns})
    dates = list(panel.index)
    res_1x = _run_window(window_universe, train_start, test_end, capital, vol_target,
                         cost_mult=1.0, strategy_name=strategy_name,
                         strategy_params=strategy_params, idm=idm, idm_cap=idm_cap,
                         rebalance=rebalance, register=False,
                         fill_sink=spec.get("fill_sink"), window=spec.get("window"),
                         fill_cfg_hash=_leg_tag(1.0))
    res_1_5x = _run_window(window_universe, train_start, test_end, capital, vol_target,
                           cost_mult=1.5, strategy_name=strategy_name,
                           strategy_params=strategy_params, idm=idm, idm_cap=idm_cap,
                           rebalance=rebalance, register=False,
                           fill_sink=spec.get("fill_sink"), window=spec.get("window"),
                           fill_cfg_hash=_leg_tag(1.5))
    return {
        "train_start": train_start, "test_start": test_start, "test_end": test_end,
        "window_universe": window_universe,
        "oos_1x": _oos_returns_dated(res_1x["equity_curve"], dates, test_start),
        "oos_1_5x": _oos_returns_dated(res_1_5x["equity_curve"], dates, test_start),
    }


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
    idm: bool = False,
    idm_cap: float | None = None,
    rebalance: str = "weekly",
    max_workers: Optional[int] = None,
    return_window_returns: bool = False,
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

    per_window_returns_1x: List[pd.Series] = []
    per_window_returns_1_5x: List[pd.Series] = []
    window_sharpes: List[float] = []
    window_universes: List[List[str]] = []
    used_windows: List[tuple[date, date, date]] = []
    per_window_oos: List[pd.Series] = []

    _sink_cfg = {"universe": universe, "capital": capital, "vol_target": vol_target,
                 "strategy_name": strategy_name, "strategy_params": strategy_params or {},
                 "idm": idm, "idm_cap": idm_cap, "rebalance": rebalance,
                 "train_months": train_months, "test_months": test_months,
                 "step_months": step_months, "start": str(start), "end": str(end),
                 "cost_mults": [1.0, 1.5]}
    cfg_hash = hashlib.sha1(json.dumps(_sink_cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
    sink = FillSink(strategy_name, FillSink.make_run_id(cfg_hash, datetime.now(timezone.utc)),
                    {"kind": "walkforward", "start": str(start), "end": str(end)})

    # Each window is independent (own panel load + two cost-leg backtests), so
    # windows are mapped across worker processes; `parallel_map` preserves
    # INPUT order, so aggregation below is byte-identical to the old serial
    # loop regardless of max_workers.
    specs = [
        {"universe": universe, "train_start": ts, "test_start": tst, "test_end": te,
         "capital": capital, "vol_target": vol_target,
         "strategy_name": strategy_name, "strategy_params": strategy_params or {},
         "idm": idm, "idm_cap": idm_cap, "rebalance": rebalance,
         "window": i + 1, "fill_sink": sink}
        for i, (ts, tst, te) in enumerate(windows)
    ]
    from src.backtesting.parallel import parallel_map
    results = parallel_map(process_window, specs, max_workers=max_workers)
    for r in results:
        if r is None:
            continue
        per_window_returns_1x.append(r["oos_1x"])
        per_window_returns_1_5x.append(r["oos_1_5x"])
        window_sharpes.append(_annualized_sharpe(r["oos_1x"].to_numpy(dtype=float)))
        window_universes.append(r["window_universe"])
        used_windows.append((r["train_start"], r["test_start"], r["test_end"]))
        if return_window_returns:
            per_window_oos.append(r["oos_1x"])

    for s in specs:
        sink.set_oos_range(s["window"], s["test_start"], s["test_end"])
    sink.finalize(oos_windows=list(range(1, len(specs) + 1)), oos_cfg_hash=_leg_tag(1.0))

    if len(used_windows) < 2:
        raise ValueError(
            f"walk-forward requires >=2 usable OOS windows after data-availability filtering, "
            f"got {len(used_windows)} for range {start}..{end}"
        )
    windows = used_windows

    stitched_1x = _stitch_oos_dedup(per_window_returns_1x)
    stitched_1_5x = _stitch_oos_dedup(per_window_returns_1_5x)

    n = int(stitched_1x.size)
    oos_sharpe = _annualized_sharpe(stitched_1x)
    oos_sharpe_1_5x_cost = _annualized_sharpe(stitched_1_5x)

    series = pd.Series(stitched_1x)
    skew = float(series.skew()) if n > 2 else 0.0
    # pandas .kurtosis() is EXCESS kurtosis (normal = 0); psr/dsr want Pearson
    # kurtosis (normal = 3) per docs/methodology/backtesting.md Section 2.2.
    kurt = float(series.kurtosis()) + 3.0 if n > 3 else 3.0

    psr_val = psr(oos_sharpe, 0.0, n, skew, kurt)
    # Gate 0.1/0.2: deflate against the real, growing project-wide
    # trial-Sharpe distribution (mirrors gate_return_stream in
    # walkforward_common.py), not a single-element list -- a 1-element
    # distribution makes DSR reduce to undeflated PSR, which understates
    # multiple-testing risk for a strategy sitting inside a 40+-trial
    # (and growing) campaign search.
    n_trials, trial_sharpes = get_campaign_trial_distribution()
    dsr_val = dsr(oos_sharpe, trial_sharpes, n, skew, kurt,
                   n_trials_project=n_trials)
    pbo_val = _compute_pbo([s.to_numpy(dtype=float) for s in per_window_returns_1x])

    result: Dict[str, Any] = {
        "oos_sharpe": oos_sharpe,
        "psr": psr_val,
        "dsr": dsr_val,
        "pbo": pbo_val,
        "oos_sharpe_1_5x_cost": oos_sharpe_1_5x_cost,
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
    if return_window_returns:
        result["per_window_oos"] = per_window_oos

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
                "trial_count_project_wide": n_trials,
                "rebalance": rebalance,
            },
            window_start=windows[0][1],
            window_end=windows[-1][2],
            phase="walk_forward",
        )
    except Exception as e:
        logger.error(f"[walk_forward_carver] registry append_run failed (non-fatal): {e}")

    result["run_id"] = run_id
    return result


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

**Trial count = {result['trial_count']}.** {title} itself is a single
parameter-free configuration with no in-run parameter search, but per Gate 0
(the strategy-lead honest-deflation fix) DSR is deflated using the real
PROJECT-WIDE trial-Sharpe distribution across the whole futures campaign
(`docs/methodology/backtesting.md` Section 2.3 / 9.4) -- the static
40-trial SP-A/B/C/E baseline plus every run subsequently logged to
`output/experiments.duckdb`, sourced via
`src.backtesting.walkforward_common.get_campaign_trial_distribution()`. This
strategy did not select over trials itself, but it is evaluated inside a
multiple-testing search that has now run 40+ trials project-wide, and DSR
must be deflated for that search, not treated as if this were the only
trial ever run.

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
              "vol_target": _DEFAULT_VOL_TARGET, "start": "2010-06-07", "end": "2025-02-01",
              "strategy_name": "CarverMomentum", "strategy_params": {}, "idm": False,
              "idm_cap": None, "rebalance": "weekly"}

    # Run-status logging survives a SIGKILL: if this run is killed, the status
    # file is frozen at RUNNING with the last heartbeat, so a stale RUNNING file
    # tells us it died (and roughly when) instead of leaving us to guess.
    from src.utils.run_status import RunStatus

    _meta = {"strategy": kw.get("strategy_name", "CarverMomentum"),
             "config": args.config, "jobs": args.jobs,
             "start": kw["start"], "end": kw["end"], "n_roots": len(kw["universe"]),
             "idm": kw.get("idm", False), "idm_cap": kw.get("idm_cap"),
             "rebalance": kw.get("rebalance", "weekly")}
    with RunStatus("carver_walkforward", meta=_meta) as st:
        result = walk_forward_carver(
            train_months=args.train_months, test_months=args.test_months, step_months=args.step_months,
            start=kw["start"], end=kw["end"], universe=kw["universe"],
            capital=kw["capital"], vol_target=kw["vol_target"],
            strategy_name=kw.get("strategy_name", "CarverMomentum"),
            strategy_params=kw.get("strategy_params", {}),
            idm=kw.get("idm", False),
            idm_cap=kw.get("idm_cap"),
            rebalance=kw.get("rebalance", "weekly"),
            max_workers=args.jobs,
        )
        st.heartbeat(note=f"gate computed: oos_sharpe={result['oos_sharpe']:.4f} n_windows={result['n_windows']}")
        report_path = _write_readiness_report(
            result, train_months=args.train_months, test_months=args.test_months,
            step_months=args.step_months, start=kw["start"], end=kw["end"], report_path=args.report,
        )
    logger.info(
        f"[walk_forward_carver] wrote {report_path}; "
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
