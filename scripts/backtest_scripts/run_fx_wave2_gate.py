"""FX Wave-2 Track A gate: combined statistical gate (Section 2.5) is PRIMARY;
S&P 500 correlation / information ratio / marginal deflated contribution are
computed as book-level context (non-gating), per
`docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`.

Reuses `walk_forward_fx` (scripts/backtest_scripts/run_fx_walkforward.py) UNCHANGED
for the gate itself (PSR/DSR/PBO via the honest, growing project-wide trial count
from `get_campaign_trial_distribution()`, and the combined-gate verdict via
`_verdict`) -- this is the SAME mechanism already used and audited for
FxTSMOM/FxXSectMom/FxCarry/FxGoldSilver. `walk_forward_fx` is called EXACTLY
ONCE per strategy so the honest trial count increments by exactly one
registry row per strategy specification (no double-counting).

The S&P book-level context is computed SEPARATELY by re-running the same
walk-forward windows (1.0x cost only) with DATED OOS returns (never
registered again -- informational only) so it can be aligned against the S&P
daily return series.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from src.backtesting.benchmark import (
    correlation_over_dates,
    information_ratio_vs_sp500,
    load_sp500_daily_returns,
    sp500_aligned_count,
    sp500_sharpe_over_dates,
)
from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
from src.backtesting.engine.fx_backtest import run_fx_backtest
from src.backtesting.walkforward_common import _as_date, _build_windows, _oos_returns_dated, _verdict
from scripts.backtest_scripts.run_fx_walkforward import _config_to_kwargs, walk_forward_fx
from src.utils import logger
from src.utils.run_status import RunStatus

_REPORT_DIR = Path("docs/reports/fx")


def _dated_oos_series(universe: Sequence[str], start: str, end: str, capital: float,
                       vol_target: float, strategy_name: str, rebalance: str,
                       idm: bool, idm_cap: Optional[float],
                       train_months: int = 36, test_months: int = 12,
                       step_months: int = 12) -> pd.Series:
    """Book-level-context-only OOS series (1.0x cost). NOT registered -- the
    gate's registry append happens exactly once, inside `walk_forward_fx`."""
    windows = _build_windows(train_months, test_months, step_months, _as_date(start), _as_date(end))
    segments = []
    for train_start, test_start, test_end in windows:
        try:
            panel = load_fx_daily_panel(list(universe), train_start, test_end)
        except FileNotFoundError as e:
            logger.warning(f"[fx_wave2_gate] skip {test_start}..{test_end}: {e}")
            continue
        window_universe = sorted({p for p, _ in panel.columns})
        dates = list(panel.index)
        cfg = {
            "asset_class": "fx",
            "strategy": {"name": strategy_name, "universe": window_universe, "params": {}},
            "dates": {"start": str(train_start), "end": str(test_end)},
            "backtest": {"initial_capital": capital, "vol_target_per_instrument": vol_target,
                         "rebalance": rebalance, "cost_mult": 1.0, "leverage_cap": 4.0,
                         "idm": idm, "idm_cap": idm_cap},
        }
        res = run_fx_backtest(cfg, register=False, log_trades=False)
        segments.append(_oos_returns_dated(res["equity_curve"], dates, test_start))
    if not segments:
        return pd.Series([], dtype=float)
    oos = pd.concat(segments).sort_index(kind="stable")
    oos = oos[~oos.index.duplicated(keep="first")]
    oos.index = pd.DatetimeIndex(oos.index)
    return oos


def run_gate(config_path: str, strategy_display_name: str,
             train_months: int = 36, test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    kw = _config_to_kwargs(cfg)

    # PRIMARY: combined statistical gate (Section 2.5), honest growing N.
    # Exactly one call -> exactly one new registry row -> exactly one new trial.
    gate = walk_forward_fx(
        train_months=train_months, test_months=test_months, step_months=step_months,
        start=kw["start"], end=kw["end"], universe=kw["universe"], capital=kw["capital"],
        vol_target=kw["vol_target"], strategy_name=kw["strategy_name"], tier=kw["tier"],
        idm=kw["idm"], idm_cap=kw["idm_cap"], rebalance=kw.get("rebalance", "weekly"),
    )
    verdict = _verdict(gate)

    # BOOK-LEVEL CONTEXT (non-gating): S&P correlation / IR / marginal
    # deflated contribution. Re-derives the OOS series with dates; never
    # appends to the registry.
    rebalance = cfg.get("backtest", {}).get("rebalance", "weekly")
    oos_dated = _dated_oos_series(
        kw["universe"], kw["start"], kw["end"], kw["capital"], kw["vol_target"],
        kw["strategy_name"], rebalance, kw["idm"], kw["idm_cap"],
        train_months=train_months, test_months=test_months, step_months=step_months,
    )
    sp = load_sp500_daily_returns()
    if len(oos_dated) > 0:
        sp_sharpe = sp500_sharpe_over_dates(oos_dated.index, sp_returns=sp)
        sp_n = sp500_aligned_count(oos_dated.index, sp_returns=sp)
        corr = correlation_over_dates(oos_dated, sp_returns=sp)
        ir = information_ratio_vs_sp500(oos_dated, sp_returns=sp)
    else:
        sp_sharpe = sp_n = corr = ir = float("nan")

    # Marginal deflated cost-net contribution: a simple, documented proxy --
    # the DSR-deflated OOS Sharpe scaled by (1 - corr^2), i.e. the portion of
    # the strategy's deflated edge that is NOT explainable by S&P beta
    # exposure. This is NOT a full mean-variance sleeve-allocation optimizer
    # (out of scope here); it is a directional book-level signal only.
    dsr_val = gate["dsr"]
    marginal_contribution = (dsr_val * (1.0 - corr ** 2)) if not np.isnan(corr) else float("nan")

    return {
        "strategy_display_name": strategy_display_name,
        "strategy_name": kw["strategy_name"],
        "config_path": config_path,
        **gate,
        "verdict": verdict,
        "sp500_sharpe_oos": sp_sharpe,
        "sp500_n_days_oos": sp_n,
        "correlation_sp500": corr,
        "information_ratio_sp500": ir,
        "marginal_deflated_contribution_proxy": marginal_contribution,
        "n_oos_days_context": int(len(oos_dated)),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "n/a" if np.isnan(x) else f"{x:.4f}"
    return str(x)


def write_report(results: list, path: str) -> str:
    lines = [
        "# FX Catalog Wave 2, Track A -- Combined Gate + Book-Level Context",
        "",
        "Generated by `scripts/backtest_scripts/run_fx_wave2_gate.py`.",
        "",
        "**Primary gate: methodology Section 2.5 combined statistical gate**",
        "(PSR>=0.95, DSR>=0.95 using the honest project-wide growing trial count,",
        "PBO<0.25, 1.5x cost-sensitivity survival) -- NOT the S&P-relative bar used",
        "in Wave 1's enhanced forms. S&P correlation/IR/marginal contribution are",
        "book-level context only and do not gate the verdict.",
        "",
        "| # | Strategy | OOS Sharpe (1x) | OOS Sharpe (1.5x) | PSR | DSR | PBO | "
        "Trials (N) | S&P corr | IR vs S&P | Marginal contrib (proxy) | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| | {r['strategy_display_name']} | {_fmt(r['oos_sharpe'])} | "
            f"{_fmt(r['oos_sharpe_1_5x_cost'])} | {_fmt(r['psr'])} | {_fmt(r['dsr'])} | "
            f"{_fmt(r['pbo'])} | {r['trial_count']} | {_fmt(r['correlation_sp500'])} | "
            f"{_fmt(r['information_ratio_sp500'])} | "
            f"{_fmt(r['marginal_deflated_contribution_proxy'])} | {r['verdict'].split(' -- ')[0]} |"
        )
    lines.append("")
    for r in results:
        lines += [
            f"## {r['strategy_display_name']} (`{r['strategy_name']}`)",
            "",
            f"Config: `{r['config_path']}`",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| OOS Sharpe (1x cost) | {_fmt(r['oos_sharpe'])} |",
            f"| OOS Sharpe (1.5x cost) | {_fmt(r['oos_sharpe_1_5x_cost'])} |",
            f"| PSR (vs SR=0) | {_fmt(r['psr'])} |",
            f"| DSR (honest project-wide N={r['trial_count']}) | {_fmt(r['dsr'])} |",
            f"| PBO | {_fmt(r['pbo'])} |",
            f"| n_windows | {r['n_windows']} |",
            f"| n_oos_days | {r['n_oos_days']} |",
            f"| skew | {_fmt(r['skew'])} |",
            f"| kurtosis (Pearson) | {_fmt(r['kurtosis_pearson'])} |",
            f"| window_start / window_end | {r['window_start']} / {r['window_end']} |",
            f"| registry run_id (gate) | {r.get('run_id')} |",
            "",
            "### Book-level context (non-gating)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| S&P 500 Sharpe (same OOS dates) | {_fmt(r['sp500_sharpe_oos'])} |",
            f"| S&P aligned day count | {r['sp500_n_days_oos']} |",
            f"| Correlation to S&P | {_fmt(r['correlation_sp500'])} |",
            f"| Information ratio vs S&P | {_fmt(r['information_ratio_sp500'])} |",
            f"| Marginal deflated contribution (proxy, DSR*(1-corr^2)) | "
            f"{_fmt(r['marginal_deflated_contribution_proxy'])} |",
            "",
            f"### Verdict: {r['verdict']}",
            "",
        ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="FX Wave 2 Track A combined gate + book-level context")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True, help="Display name for the report, e.g. '#33 Turn-of-Month USD'")
    parser.add_argument("--report", default=None)
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    args = parser.parse_args()

    report_path = args.report or str(_REPORT_DIR / f"{Path(args.config).stem}_wave2_gate.md")

    with RunStatus("fx_wave2_gate", meta={"config": args.config, "name": args.name}) as st:
        result = run_gate(args.config, args.name, args.train_months, args.test_months, args.step_months)
        st.heartbeat(note=f"gate computed: oos_sharpe={result['oos_sharpe']:.4f} verdict={result['verdict'][:40]}")
        out = write_report([result], report_path)

    logger.info(
        f"[fx_wave2_gate] {args.name}: oos_sharpe={result['oos_sharpe']:.4f} "
        f"dsr={result['dsr']:.4f} psr={result['psr']:.4f} pbo={result['pbo']:.4f} "
        f"corr_sp500={result['correlation_sp500']:.4f} verdict={result['verdict']}"
    )
    logger.info(f"[fx_wave2_gate] wrote {out}")


if __name__ == "__main__":
    main()
