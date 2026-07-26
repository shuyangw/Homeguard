"""FX spread-RV walk-forward + combined statistical gate (Section 2.5), Wave 2
Track B (#35 AudNzdPairs, #37 CointScanner, #30 VolRatioPair).

Mirrors `run_fx_walkforward.py` / `run_fx_carry_seatbelt_walkforward.py`: rolls
36m/12m/12m OOS windows, re-running the spread assembly
(`scripts/backtest_scripts/run_fx_spread_backtest.py`'s strategy/simulator
wiring, inlined here so the expensive spread_book() scan -- which is
cost-independent -- is computed ONCE per window and reused for both cost
legs instead of twice) over [train_start, test_end] (the train segment is
signal lookback warm-up only), keeps the OOS-dated tail of each window's
return series, and stitches across windows. Both cost legs (1.0x, 1.5x) are
computed per window.

PRIMARY gate: methodology Section 2.5 combined statistical gate (PSR>=0.95,
DSR>=0.95 using the honest, growing project-wide trial count from
`get_campaign_trial_distribution()`, PBO<0.25, 1.5x cost-sensitivity
survival) via `walkforward_common._verdict` -- NOT an S&P-relative bar
(inappropriate for market-neutral spreads). S&P correlation / IR / marginal
deflated contribution are book-level context, non-gating, per
`docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md` Section 4
and `docs/superpowers/specs/2026-07-19-fx-spread-engine-design.md` Section 7.

Each strategy is a single, fixed a-priori config (no optimizer round, no
parameter search): exactly one registry append per strategy per call to
`walk_forward_fx_spread` = exactly one new trial on the honest, growing
project-wide DSR trial count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from src.backtesting.benchmark import (
    correlation_over_dates, information_ratio_vs_sp500, load_sp500_daily_returns,
    sp500_aligned_count, sp500_sharpe_over_dates)
from src.backtesting.data.fx_backtest_loader import build_quote_usd_panel, load_fx_daily_panel
from src.backtesting.engine.fill_sink import FillSink
from src.backtesting.engine.fx_backtest import _cost_fn_factory
from src.backtesting.engine.fx_spread_simulator import FxSpreadPortfolioSimulator
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr
from src.backtesting.walkforward_common import (
    _annualized_sharpe, _as_date, _build_windows, _compute_pbo, _verdict,
    get_campaign_trial_distribution)
from scripts.backtest_scripts.run_fx_spread_backtest import (
    _conversion_legs, _make_strategy, _vol_ratio_coupled_sets)
from src.utils import logger
from src.utils.run_status import RunStatus

_REPORT_DIR = Path("docs/reports/fx")
_DEFAULT_CAPITAL = 100_000.0
_COST_LEGS = (1.0, 1.5)


def _leg_tag(mult: float) -> str:
    s = ("%g" % float(mult)).replace(".", "")
    return f"c{s}x"


def _run_window_spread(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Top-level (picklable) per-window worker.

    Builds the spread book ONCE (cost-independent) and simulates BOTH cost
    legs against it, avoiding a redundant second cointegration/vol-ratio
    scan for the 1.5x leg.
    """
    strategy_name = spec["strategy_name"]
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    try:
        load_universe = list(dict.fromkeys(universe + _conversion_legs(universe)))
        panel = load_fx_daily_panel(load_universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[fx_spread_wf] skip {test_start}..{test_end}: {e}")
        return None
    panel_pairs = {c[0] for c in panel.columns}
    present = [p for p in universe if p in panel_pairs]
    if not present:
        logger.warning(f"[fx_spread_wf] skip {test_start}..{test_end}: no traded pairs present")
        return None
    close = panel.xs("close", axis=1, level=1)[present]
    quote_usd = build_quote_usd_panel(panel, present)

    strategy = _make_strategy(strategy_name, present)
    book, sigma = strategy.spread_book(close)

    fill_sink = spec.get("fill_sink")
    window = spec.get("window")

    oos_by_cost: Dict[float, pd.Series] = {}
    for cost_mult in _COST_LEGS:
        sim = FxSpreadPortfolioSimulator(spec["capital"], _cost_fn_factory(),
                                         rebalance=spec["rebalance"], cost_mult=cost_mult,
                                         leverage_cap=spec["leverage_cap"])
        res = sim.run_spreads(close, book, sigma, quote_usd, spec["vol_target"])
        returns = res.equity_curve.pct_change(fill_method=None).dropna()
        oos_by_cost[cost_mult] = returns[returns.index >= test_start]
        if fill_sink is not None and window is not None:
            extras = {"leverage_utilization": res.leverage_utilization.rename(
                "leverage_utilization").reset_index()}
            fill_sink.write_window(res.trades, window, cfg_hash=_leg_tag(cost_mult), extras=extras)

    if oos_by_cost[1.0].empty:
        return None
    return {"train_start": train_start, "test_start": test_start, "test_end": test_end,
            "present_universe": present,
            "oos_1x": oos_by_cost[1.0], "oos_1_5x": oos_by_cost[1.5]}


def walk_forward_fx_spread(
    strategy_name: str,
    universe: Sequence[str],
    start: str,
    end: str,
    vol_target: float = 0.10,
    rebalance: str = "weekly",
    capital: float = _DEFAULT_CAPITAL,
    leverage_cap: float = 4.0,
    train_months: int = 36,
    test_months: int = 12,
    step_months: int = 12,
    max_workers: Optional[int] = None,
    register: bool = True,
) -> Dict[str, Any]:
    universe = list(universe)
    start_d, end_d = _as_date(start), _as_date(end)
    windows = _build_windows(train_months, test_months, step_months, start_d, end_d)
    if len(windows) < 2:
        raise ValueError(
            f"walk-forward requires >=2 OOS windows, got {len(windows)} "
            f"for range {start}..{end} with train={train_months}m test={test_months}m step={step_months}m"
        )

    # Run-scoped fill logging (methodology Section 12 / CLAUDE.md): every
    # simulated run persists its fills. Per-window, per-cost-leg trade files
    # go under output/backtests/<strategy>/runs/<run_id>/, plus a
    # manifest.csv and a trades_oos.csv.gz (1.0x leg sliced to each window's
    # OOS segment) -- mirrors run_fx_walkforward.py's FillSink wiring.
    _sink_cfg = {"strategy_name": strategy_name, "universe": universe, "start": str(start),
                 "end": str(end), "vol_target": vol_target, "rebalance": rebalance,
                 "capital": capital, "leverage_cap": leverage_cap,
                 "train_months": train_months, "test_months": test_months,
                 "step_months": step_months, "cost_legs": list(_COST_LEGS)}
    cfg_hash = hashlib.sha1(json.dumps(_sink_cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
    sink = FillSink(strategy_name, FillSink.make_run_id(cfg_hash, datetime.now(timezone.utc)),
                    {"kind": "walkforward_spread", "start": str(start), "end": str(end)})

    specs = [
        {"strategy_name": strategy_name, "universe": universe, "train_start": ts,
         "test_start": tst, "test_end": te, "vol_target": vol_target,
         "rebalance": rebalance, "capital": capital, "leverage_cap": leverage_cap,
         "window": i + 1, "fill_sink": sink}
        for i, (ts, tst, te) in enumerate(windows)
    ]
    from src.backtesting.parallel import parallel_map
    raw_results = parallel_map(_run_window_spread, specs, max_workers=max_workers)
    # raw_results is in input order (parallel_map contract), so it pairs 1:1
    # with specs; None entries are windows dropped for data-availability.
    oos_windows: List[int] = []
    for s, r in zip(specs, raw_results):
        if r is not None:
            sink.set_oos_range(s["window"], s["test_start"], s["test_end"])
            oos_windows.append(s["window"])
    sink.finalize(oos_windows=oos_windows, oos_cfg_hash=_leg_tag(1.0))

    results = [r for r in raw_results if r is not None]
    if len(results) < 2:
        raise ValueError(
            f"walk-forward requires >=2 usable OOS windows after data-availability filtering, "
            f"got {len(results)} for range {start}..{end}"
        )

    # Adjacent windows share one boundary calendar day (test_end_N ==
    # test_start_{N+1}); dedupe on concat to avoid double-counting that day.
    oos_1x = pd.concat([r["oos_1x"] for r in results]).sort_index(kind="stable")
    oos_1x = oos_1x[~oos_1x.index.duplicated(keep="first")]
    oos_1_5x = pd.concat([r["oos_1_5x"] for r in results]).sort_index(kind="stable")
    oos_1_5x = oos_1_5x[~oos_1_5x.index.duplicated(keep="first")]
    oos_1x.index = pd.DatetimeIndex(oos_1x.index)
    oos_1_5x.index = pd.DatetimeIndex(oos_1_5x.index)

    per_window_1x = [r["oos_1x"].to_numpy(dtype=float) for r in results]

    arr = oos_1x.to_numpy(dtype=float)
    n = int(arr.size)
    sharpe = _annualized_sharpe(arr)
    sharpe_1_5x = _annualized_sharpe(oos_1_5x.to_numpy(dtype=float))
    ser = pd.Series(arr)
    skew = float(ser.skew()) if n > 2 else 0.0
    kurt = float(ser.kurtosis()) + 3.0 if n > 3 else 3.0

    n_trials, trial_sharpes = get_campaign_trial_distribution()
    psr_val = psr(sharpe, 0.0, n, skew, kurt, periods_per_year=252)
    dsr_val = dsr(sharpe, trial_sharpes, n, skew, kurt, n_trials_project=n_trials,
                  periods_per_year=252)
    pbo_val = _compute_pbo(per_window_1x)

    sp = load_sp500_daily_returns()
    sp_sharpe = sp500_sharpe_over_dates(oos_1x.index, sp_returns=sp)
    sp_n = sp500_aligned_count(oos_1x.index, sp_returns=sp)
    corr = correlation_over_dates(oos_1x, sp_returns=sp)
    ir = information_ratio_vs_sp500(oos_1x, sp_returns=sp)
    marginal_contribution = (dsr_val * (1.0 - corr ** 2)) if not np.isnan(corr) else float("nan")

    all_present = sorted({p for r in results for p in r["present_universe"]})
    missing_any_window = sorted(set(universe) - set(all_present))

    result: Dict[str, Any] = {
        "strategy_name": strategy_name,
        "universe": universe,
        "present_universe": all_present,
        "missing_universe": missing_any_window,
        "oos_sharpe": sharpe,
        "oos_sharpe_1_5x_cost": sharpe_1_5x,
        "psr": psr_val,
        "dsr": dsr_val,
        "pbo": pbo_val,
        "n_windows": len(results),
        "n_oos_days": n,
        "skew": skew,
        "kurtosis_pearson": kurt,
        "trial_count": n_trials,
        "window_start": str(results[0]["test_start"]),
        "window_end": str(results[-1]["test_end"]),
        "sp500_sharpe_oos": sp_sharpe,
        "sp500_n_days_oos": sp_n,
        "correlation_sp500": corr,
        "information_ratio_sp500": ir,
        "marginal_deflated_contribution_proxy": marginal_contribution,
        "capital": capital,
        "vol_target": vol_target,
        "rebalance": rebalance,
        "fill_run_id": sink.run_id,
        "fill_run_dir": str(sink.run_dir),
    }
    result["verdict"] = _verdict(result)

    run_id = None
    if register:
        try:
            from src.experiments import append_run

            run_id = append_run(
                strategy_name=strategy_name,
                agent_name="fx-spread-walkforward",
                metrics={k: v for k, v in result.items()
                         if k not in ("universe", "present_universe", "missing_universe")},
                asset_class="fx",
                data_frequency="daily",
                params={
                    "train_months": train_months, "test_months": test_months,
                    "step_months": step_months, "universe": universe,
                    "vol_target": vol_target, "initial_capital": capital,
                    "rebalance": rebalance, "leverage_cap": leverage_cap,
                    "trial_count_project_wide": n_trials,
                },
                window_start=result["window_start"],
                window_end=result["window_end"],
                phase="walk_forward",
            )
        except Exception as e:
            logger.error(f"[fx_spread_wf] registry append_run failed (non-fatal): {e}")
    else:
        logger.info("[fx_spread_wf] register=False: skipping experiment-registry append_run "
                    "(diagnostic re-simulation, not a new trial)")

    result["run_id"] = run_id
    return result


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "n/a" if np.isnan(x) else f"{x:.4f}"
    return str(x)


def write_report(result: Dict[str, Any], strategy_display_name: str, config_path: str, path: str) -> str:
    dropped = ""
    if result["missing_universe"]:
        dropped = (f"\n**Data-coverage note:** declared universe pairs "
                   f"{result['missing_universe']} were absent from the cache in at least "
                   f"one walk-forward window; the verdict below reflects the surviving "
                   f"pairs only ({result['present_universe']}).\n")
    lines = [
        f"# {strategy_display_name} -- FX Wave 2 Track B Combined Gate + Book-Level Context",
        "",
        "Generated by `scripts/backtest_scripts/run_fx_spread_walkforward.py`.",
        f"Config: `{config_path}`. Strategy class: `{result['strategy_name']}`.",
        "",
        "**Primary gate: methodology Section 2.5 combined statistical gate**",
        "(PSR>=0.95, DSR>=0.95 using the honest project-wide growing trial count,",
        "PBO<0.25, 1.5x cost-sensitivity survival) -- NOT an S&P-relative bar",
        "(inappropriate for a market-neutral relative-value spread). S&P",
        "correlation/IR/marginal contribution are book-level context only and do",
        "not gate the verdict.",
        dropped,
        "| Metric | Value |",
        "|---|---|",
        f"| Declared universe | {result['universe']} |",
        f"| OOS Sharpe (1x cost) | {_fmt(result['oos_sharpe'])} |",
        f"| OOS Sharpe (1.5x cost) | {_fmt(result['oos_sharpe_1_5x_cost'])} |",
        f"| PSR (vs SR=0) | {_fmt(result['psr'])} |",
        f"| DSR (honest project-wide N={result['trial_count']}) | {_fmt(result['dsr'])} |",
        f"| PBO | {_fmt(result['pbo'])} |",
        f"| n_windows | {result['n_windows']} |",
        f"| n_oos_days | {result['n_oos_days']} |",
        f"| skew | {_fmt(result['skew'])} |",
        f"| kurtosis (Pearson) | {_fmt(result['kurtosis_pearson'])} |",
        f"| window_start / window_end | {result['window_start']} / {result['window_end']} |",
        f"| capital / vol_target / rebalance | {result['capital']:,.0f} / {result['vol_target']} / {result['rebalance']} |",
        f"| registry run_id (gate) | {result.get('run_id')} |",
        "",
        "### Book-level context (non-gating)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| S&P 500 Sharpe (same OOS dates) | {_fmt(result['sp500_sharpe_oos'])} |",
        f"| S&P aligned day count | {result['sp500_n_days_oos']} |",
        f"| Correlation to S&P | {_fmt(result['correlation_sp500'])} |",
        f"| Information ratio vs S&P | {_fmt(result['information_ratio_sp500'])} |",
        f"| Marginal deflated contribution (proxy, DSR*(1-corr^2)) | "
        f"{_fmt(result['marginal_deflated_contribution_proxy'])} |",
        "",
        f"### Verdict: {result['verdict']}",
        "",
        "## Design notes",
        "",
        "Rolls non-overlapping 36m/12m/12m OOS test windows across the full",
        "declared date range; each window re-runs the spread assembly over",
        "[train_start, test_end] (train segment is signal lookback warm-up),",
        "keeping only the OOS-dated (test_start..test_end) tail. Both cost legs",
        "share ONE spread-book computation per window (book is cost-independent);",
        "only the simulator's cost_mult differs. Trial count is the honest,",
        "growing project-wide count from `get_campaign_trial_distribution()`",
        "(includes Wave 1 + Wave 2 Track A + all prior Track B strategies gated",
        "before this one in the same session).",
        "",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def run_gate(config_path: str, strategy_display_name: str,
             train_months: int = 36, test_months: int = 12, step_months: int = 12,
             max_workers: Optional[int] = None, register: bool = True) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    strat, bt, dts = cfg["strategy"], cfg.get("backtest", {}), cfg["dates"]

    universe = list(strat["universe"])
    if strat["name"] == "VolRatioPair":
        surviving = _vol_ratio_coupled_sets(universe)
        if len(surviving) < 3:
            logger.warning(f"[fx_spread_wf] VolRatioPair coupled sets surviving: {surviving} "
                           f"(declared {len(universe) // 2} sets)")

    result = walk_forward_fx_spread(
        strategy_name=strat["name"], universe=universe,
        start=str(dts["start"]), end=str(dts["end"]),
        vol_target=float(bt.get("vol_target", 0.10)),
        rebalance=bt.get("rebalance", "weekly"),
        capital=float(bt.get("initial_capital", _DEFAULT_CAPITAL)),
        leverage_cap=float(bt.get("leverage_cap", 4.0)),
        train_months=train_months, test_months=test_months, step_months=step_months,
        max_workers=max_workers, register=register,
    )
    return {"strategy_display_name": strategy_display_name, "config_path": config_path, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="FX Wave 2 Track B spread combined gate + book-level context")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True, help="Display name for the report, e.g. '#35 AUD/NZD pairs'")
    parser.add_argument("--report", default=None)
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--no-register", dest="register", action="store_false", default=True,
                        help="Skip the experiment-registry append_run (for re-simulating an "
                             "already-counted, deterministic spec for diagnostics only -- does "
                             "NOT increment the project-wide trial count). Default: register.")
    args = parser.parse_args()

    report_path = args.report or str(_REPORT_DIR / f"{Path(args.config).stem}_wave2_gate.md")

    meta = {"config": args.config, "name": args.name, "register": args.register}
    with RunStatus("fx_spread_walkforward", meta=meta) as st:
        result = run_gate(args.config, args.name, args.train_months, args.test_months,
                          args.step_months, max_workers=args.jobs, register=args.register)
        st.heartbeat(note=f"gate computed: oos_sharpe={result['oos_sharpe']:.4f} verdict={result['verdict'][:40]}")
        out = write_report(result, args.name, args.config, report_path)

    logger.info(
        f"[fx_spread_wf] {args.name}: oos_sharpe={result['oos_sharpe']:.4f} "
        f"dsr={result['dsr']:.4f} psr={result['psr']:.4f} pbo={result['pbo']:.4f} "
        f"corr_sp500={result['correlation_sp500']:.4f} verdict={result['verdict']}"
    )
    logger.info(f"[fx_spread_wf] wrote {out}")


if __name__ == "__main__":
    main()
