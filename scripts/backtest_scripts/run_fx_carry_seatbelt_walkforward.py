"""FxCarrySeatbelt walk-forward with the S&P relative gate.

Rolls the standard 36/12/12 OOS windows on a given cadence config, stitches the
DATED OOS return series (both cost legs), and evaluates the primary criterion:
strategy OOS Sharpe > S&P Sharpe over the same dates. PSR/DSR/PBO, correlation,
IR, and per-episode attribution are computed and reported as diagnostics only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.benchmark import (
    load_sp500_daily_returns, sp500_sharpe_over_dates, sp500_aligned_count,
    correlation_over_dates, information_ratio_vs_sp500)
from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
from src.backtesting.engine.fill_sink import FillSink
from src.backtesting.engine.fx_backtest import run_fx_backtest
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr
from src.backtesting.walkforward_common import (
    _as_date, _build_windows, _annualized_sharpe, _compute_pbo, _oos_returns_dated)
from src.utils import logger

_REPORT_PATH = "docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md"
# Named unwind episodes for existence-proof attribution (start, end inclusive).
_EPISODES = {
    "Aug 2024 yen-carry unwind": ("2024-07-15", "2024-08-15"),
    "Mar 2020 COVID unwind": ("2020-02-20", "2020-03-31"),
    "Jan 2019 flash": ("2019-01-01", "2019-01-10"),
}


def _run_window(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    try:
        panel = load_fx_daily_panel(universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[seatbelt_wf] skip {test_start}..{test_end}: {e}")
        return None
    window_universe = sorted({p for p, _ in panel.columns})
    dates = list(panel.index)

    def one(cost_mult: float):
        cfg = {"asset_class": "fx",
               "strategy": {"name": "FxCarrySeatbelt", "universe": window_universe, "params": {}},
               "dates": {"start": str(train_start), "end": str(test_end)},
               "backtest": {"initial_capital": spec["capital"],
                            "vol_target_per_instrument": spec["vol_target"],
                            "rebalance": spec["rebalance"], "cost_mult": cost_mult,
                            "leverage_cap": spec["leverage_cap"], "idm": spec["idm"],
                            "idm_cap": spec["idm_cap"]}}
        leg_sink = spec.get("fill_sink") if cost_mult == 1.0 else None
        res = run_fx_backtest(cfg, register=False, fill_sink=leg_sink, window=spec.get("window"))
        eq = pd.Series(res["equity_curve"], index=pd.Index(dates))
        oos = _oos_returns_dated(res["equity_curve"], dates, test_start)
        is_ret = eq[eq.index < test_start].pct_change().dropna()
        return oos, is_ret

    oos_1x, is_1x = one(1.0)
    oos_1_5x, _ = one(1.5)
    oos_0_5x, _ = one(0.5)
    return {"oos_1x": oos_1x, "oos_1_5x": oos_1_5x, "oos_0_5x": oos_0_5x, "is_1x": is_1x}


def run(config_path: str, cadence_label: str, trial_count: int,
        train_months: int = 36, test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    strat, bt, dts = cfg["strategy"], cfg["backtest"], cfg["dates"]
    universe = list(strat["universe"])
    start_d, end_d = _as_date(dts["start"]), _as_date(dts["end"])
    windows = _build_windows(train_months, test_months, step_months, start_d, end_d)

    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
    sink = FillSink("FxCarrySeatbelt", FillSink.make_run_id(cfg_hash, datetime.now(timezone.utc)),
                    {"kind": "walkforward", "start": str(dts["start"]), "end": str(dts["end"])})

    specs = [{"universe": universe, "train_start": ts, "test_start": tst, "test_end": te,
              "capital": float(bt["initial_capital"]),
              "vol_target": float(bt["vol_target_per_instrument"]),
              "rebalance": bt.get("rebalance", "daily"),
              "leverage_cap": float(bt.get("leverage_cap", 4.0)),
              "idm": bool(bt.get("idm", True)), "idm_cap": bt.get("idm_cap"),
              "window": i + 1, "fill_sink": sink}
             for i, (ts, tst, te) in enumerate(windows)]

    from src.backtesting.parallel import parallel_map
    results = [r for r in parallel_map(_run_window, specs) if r is not None]
    sink.finalize(oos_windows=list(range(1, len(specs) + 1)))
    if len(results) < 2:
        raise ValueError(f"need >=2 usable OOS windows, got {len(results)}")

    # Adjacent windows share one boundary calendar day (test_end_N ==
    # test_start_{N+1}) since _oos_returns_dated returns a closed [start, end]
    # interval; dedupe on concat to avoid double-counting that day.
    oos_1x = pd.concat([r["oos_1x"] for r in results]).sort_index(kind="stable")
    oos_1x = oos_1x[~oos_1x.index.duplicated(keep="first")]
    oos_1_5x = pd.concat([r["oos_1_5x"] for r in results]).sort_index(kind="stable")
    oos_1_5x = oos_1_5x[~oos_1_5x.index.duplicated(keep="first")]
    oos_0_5x = pd.concat([r["oos_0_5x"] for r in results]).sort_index(kind="stable")
    oos_0_5x = oos_0_5x[~oos_0_5x.index.duplicated(keep="first")]
    per_window_1x = [r["oos_1x"].to_numpy(dtype=float) for r in results]

    # _oos_returns_dated indexes with raw datetime.date objects (per its
    # List[date] signature); normalize to a real DatetimeIndex here so
    # downstream alignment against the S&P DatetimeIndex (correlation, IR,
    # episode date slicing) matches on value instead of silently mismatching
    # on dtype (date != Timestamp under Python's comparison rules).
    oos_1x.index = pd.DatetimeIndex(oos_1x.index)
    oos_1_5x.index = pd.DatetimeIndex(oos_1_5x.index)
    oos_0_5x.index = pd.DatetimeIndex(oos_0_5x.index)

    arr = oos_1x.to_numpy(dtype=float)
    n = int(arr.size)
    sharpe = _annualized_sharpe(arr)
    sharpe_1_5x = _annualized_sharpe(oos_1_5x.to_numpy(dtype=float))
    sharpe_0_5x = _annualized_sharpe(oos_0_5x.to_numpy(dtype=float))
    # IS/OOS overfit diagnostic: mean of per-window in-sample Sharpes (avoids
    # double-counting the heavily-overlapping train segments).
    per_window_is = [_annualized_sharpe(r["is_1x"].to_numpy(dtype=float))
                     for r in results if r["is_1x"].size > 1]
    is_sharpe = float(np.nanmean(per_window_is)) if per_window_is else float("nan")
    is_oos_ratio = (is_sharpe / sharpe
                    if sharpe > 0 and not np.isnan(sharpe) else float("nan"))
    ser = pd.Series(arr)
    skew = float(ser.skew()) if n > 2 else 0.0
    kurt = float(ser.kurtosis()) + 3.0 if n > 3 else 3.0
    psr_val = psr(sharpe, 0.0, n, skew, kurt)
    dsr_val = dsr(sharpe, [sharpe], n, skew, kurt, n_trials_project=trial_count)
    pbo_val = _compute_pbo(per_window_1x)

    sp = load_sp500_daily_returns()
    sp_sharpe = sp500_sharpe_over_dates(oos_1x.index, sp_returns=sp)
    sp_n = sp500_aligned_count(oos_1x.index, sp_returns=sp)
    if n > 0 and abs(sp_n - n) / n > 0.05:
        logger.warning(f"[seatbelt_wf] S&P aligned day count {sp_n} differs from strategy OOS day count {n} by >5%; comparison may not be apples-to-apples")
    corr = correlation_over_dates(oos_1x, sp_returns=sp)
    ir = information_ratio_vs_sp500(oos_1x, sp_returns=sp)
    beats = bool(sharpe > sp_sharpe)

    episodes = {}
    for name, (s, e) in _EPISODES.items():
        seg = oos_1x[(oos_1x.index >= pd.Timestamp(s)) & (oos_1x.index <= pd.Timestamp(e))]
        episodes[name] = float((1.0 + seg).prod() - 1.0) if len(seg) else float("nan")

    return {"cadence": cadence_label, "n_oos_days": n, "n_windows": len(results),
            "oos_sharpe": sharpe, "oos_sharpe_1_5x": sharpe_1_5x,
            "oos_sharpe_0_5x": sharpe_0_5x,
            "is_sharpe": is_sharpe, "is_oos_ratio": is_oos_ratio,
            "sp500_sharpe": sp_sharpe, "sp500_n_days": sp_n, "beats_sp500": beats,
            "correlation_sp500": corr, "information_ratio_sp500": ir,
            "psr": psr_val, "dsr": dsr_val, "pbo": pbo_val, "skew": skew,
            "kurtosis_pearson": kurt, "trial_count": trial_count,
            "oos_start": str(oos_1x.index.min().date()),
            "oos_end": str(oos_1x.index.max().date()), "episodes": episodes}


def _write_report(results: List[Dict[str, Any]], path: str = _REPORT_PATH) -> str:
    lines = ["# FxCarrySeatbelt Walk-Forward Readiness Report", "",
             "Generated by `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`.",
             "Primary gate: OOS Sharpe (1x cost) > S&P 500 Sharpe over the same OOS dates.",
             "PSR/DSR/PBO are diagnostics, not gates (see the 2026-07-06 pre-registration).", ""]
    for r in results:
        verdict = "PASS" if r["beats_sp500"] else "FAIL"
        lines += [f"## Cadence: {r['cadence']} -- {verdict}", "",
                  "| Metric | Value |", "|---|---|",
                  f"| OOS Sharpe (1x) | {r['oos_sharpe']:.4f} |",
                  f"| S&P Sharpe (aligned OOS dates) | {r['sp500_sharpe']:.4f} |",
                  f"| S&P observation count | {r['sp500_n_days']} |",
                  f"| Beats S&P | {r['beats_sp500']} |",
                  f"| OOS Sharpe (1.5x cost) | {r['oos_sharpe_1_5x']:.4f} |",
                  f"| IS Sharpe (1x, mean per-window) | {r['is_sharpe']:.4f} |",
                  f"| IS/OOS Sharpe ratio | {r['is_oos_ratio']:.4f} |",
                  f"| Correlation to S&P | {r['correlation_sp500']:.4f} |",
                  f"| Information ratio vs S&P | {r['information_ratio_sp500']:.4f} |",
                  f"| PSR (diag) | {r['psr']:.4f} |",
                  f"| DSR (diag, trials={r['trial_count']}) | {r['dsr']:.4f} |",
                  f"| PBO (diag) | {r['pbo']:.4f} |",
                  f"| n_windows / n_oos_days | {r['n_windows']} / {r['n_oos_days']} |",
                  f"| OOS window | {r['oos_start']} .. {r['oos_end']} |", "",
                  "### Episode attribution (existence proof, not statistics)", "",
                  "| Episode | Strategy OOS return |", "|---|---|"]
        for name, val in r["episodes"].items():
            shown = "n/a (outside OOS)" if np.isnan(val) else f"{val:+.2%}"
            lines.append(f"| {name} | {shown} |")
        lines += ["",
                  "Limitations: carry gate uses the FRED policy-rate differential as a swap",
                  "proxy (optimism bias); the offensive short rests on ~4-6 events.", ""]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def main() -> None:
    import argparse
    from src.utils.run_status import RunStatus
    try:
        from src.experiments import n_trials_project_wide
        base_trials = int(n_trials_project_wide())
    except Exception:
        base_trials = 0

    parser = argparse.ArgumentParser(description="FxCarrySeatbelt walk-forward + S&P gate")
    parser.add_argument("--report", default=_REPORT_PATH)
    args = parser.parse_args()

    configs = [("config/backtesting/fx_carry_seatbelt_daily.yaml", "daily"),
               ("config/backtesting/fx_carry_seatbelt_weekly.yaml", "weekly")]
    # Two new configs increment the project-wide trial count for the DSR diagnostic.
    trial_count = base_trials + len(configs)

    with RunStatus("fx_carry_seatbelt_walkforward", meta={"configs": [c for c, _ in configs]}):
        results = [run(path, label, trial_count) for path, label in configs]
        report_path = _write_report(results, args.report)

    for r in results:
        logger.info(f"[seatbelt_wf] {r['cadence']}: oos_sharpe={r['oos_sharpe']:.4f} "
                    f"sp500={r['sp500_sharpe']:.4f} beats={r['beats_sp500']} "
                    f"dsr={r['dsr']:.4f} pbo={r['pbo']:.4f}")
    logger.info(f"[seatbelt_wf] wrote {report_path}")


if __name__ == "__main__":
    main()
