"""#20 London Open Breakout walk-forward with the S&P relative gate.

Runs the intraday order engine per FX trading day for each pair, collects the
qty/pip-independent R-multiple booked per entry day (`day_r`), converts it to a
per-pair daily return (`risk_frac * day_r`), aggregates the pairs equal-weight
into one combined daily return series, then feeds that series to the standard
FX walk-forward (36m/12m/12m) and evaluates the primary criterion: stitched OOS
Sharpe > S&P 500 Sharpe over the same OOS dates. PSR/DSR/PBO, correlation, IR,
and IS/OOS degradation are computed and reported as diagnostics only.

Cost note: fills are charged the MEASURED hour-of-week spread at the hour each
fill lands on (half the round trip at entry, half at exit), replacing the flat
major-tier constant used through 2026-07-25. The flat model over-charged the
London window it trades by roughly 2-3.7x, so the earlier OOS -1.60 is biased
against the strategy -- the only apparatus defect in this campaign running in
the pessimistic direction.

The 1.5x cost-stress leg the methodology requires is now buildable and was not
before: `cost_mult` scales the charge at the strategy level, which is what the
summed daily R could not be rescaled from. Pass `cost_mult` in the config
params. Any re-gate on this changed cost model is verdict work and belongs to
strategy-lead.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.benchmark import (
    load_sp500_daily_returns, sp500_sharpe_over_dates, sp500_aligned_count,
    correlation_over_dates, information_ratio_vs_sp500)
from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
from src.backtesting.data.fx_intraday_loader import load_fx_1min
from src.backtesting.engine.intraday_order_engine import Bar, OrderEngine
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr
from src.backtesting.utils.indicators import Indicators
from src.backtesting.walkforward_common import _annualized_sharpe, _as_date, _compute_pbo
from src.data.macro_calendar_tier1 import generate_tier1_releases
from src.strategies.advanced.fx_london_breakout import LondonBreakoutStrategy
from src.utils import logger

_REPORT_PATH = "docs/reports/fx/FX_LONDON_BREAKOUT_WALK_FORWARD.md"


def _fx_day_values(utc_index: pd.DatetimeIndex) -> np.ndarray:
    """FX trading day (17:00-ET boundary) for a tz-aware UTC index, DST-safe.

    Mirrors ``fx_clock.fx_trading_day`` (shift NY wall time by +7h so 17:00 ET
    becomes local midnight, then take the calendar date), but operates on
    tz-NAIVE wall times so the +7h shift never re-localizes into a spring-forward
    gap. ``fx_clock``'s ``DateOffset(hours=7)`` on the tz-aware index raises
    NonExistentTimeError on 1m data crossing the DST gap (e.g. 2021-03-14 02:14).
    """
    ny_naive = utc_index.tz_convert("America/New_York").tz_localize(None)
    shifted = ny_naive + pd.Timedelta(hours=7)
    return np.asarray(pd.DatetimeIndex(shifted).date)


def _daily_ohlc_from_panel(pair: str, start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
    try:
        panel = load_fx_daily_panel([pair], start, end)
    except FileNotFoundError:
        return None
    if pair not in {c[0] for c in panel.columns}:
        return None
    sub = panel[pair][["high", "low", "close"]].dropna()
    return sub if not sub.empty else None


def _daily_ohlc_from_1min(bars_1m: pd.DataFrame) -> pd.DataFrame:
    fxday = _fx_day_values(bars_1m.index)
    grp = bars_1m.groupby(fxday)
    ohlc = pd.DataFrame({"high": grp["high"].max(), "low": grp["low"].min(),
                         "close": grp["close"].last()})
    return ohlc.sort_index()


def _daily_atr_prior(pair: str, bars_1m: pd.DataFrame,
                     start: dt.date, end: dt.date) -> Dict[dt.date, float]:
    """ATR(14) on daily FX OHLC, each day mapped to the PRIOR day's ATR (no lookahead).

    Prefers the dedicated fx_daily cache (`load_fx_daily_panel`); falls back to
    daily OHLC aggregated from the pair's own 1m bars grouped by FX trading day
    when the pair is absent from that cache (EURGBP/GBPJPY are not in fx_daily).
    """
    ohlc = _daily_ohlc_from_panel(pair, start, end)
    if ohlc is None:
        ohlc = _daily_ohlc_from_1min(bars_1m)
    atr = Indicators.atr(ohlc["high"], ohlc["low"], ohlc["close"], 14)
    prior = atr.shift(1)
    return {d: float(v) for d, v in prior.items() if pd.notna(v)}


def _bars_for_day(sub: pd.DataFrame) -> List[Bar]:
    ts = sub.index.to_pydatetime()
    o = sub["open"].to_numpy(dtype=float)
    h = sub["high"].to_numpy(dtype=float)
    lo = sub["low"].to_numpy(dtype=float)
    c = sub["close"].to_numpy(dtype=float)
    return [Bar(ts[i], o[i], h[i], lo[i], c[i]) for i in range(len(sub))]


def _pair_daily_returns(pair: str, start: dt.date, end: dt.date, releases: pd.DataFrame,
                        risk_frac: float, tp_fraction: float, offset_pips: float,
                        override_pips: Optional[float] = None,
                        cost_mult: float = 1.0) -> pd.Series:
    bars = load_fx_1min(pair, start, end)
    if bars.empty:
        logger.warning(f"[london_wf] no 1m data for {pair}; contributing empty series")
        return pd.Series(dtype=float)
    atr_prior = _daily_atr_prior(pair, bars, start, end)
    fxday = _fx_day_values(bars.index)
    day_r: Dict[dt.date, float] = {}
    trading_days: List[dt.date] = []
    for day, sub in bars.groupby(fxday):
        trading_days.append(day)
        strat = LondonBreakoutStrategy(pair, atr_d1=atr_prior, risk_frac=risk_frac,
                                       tp_fraction=tp_fraction, offset_pips=offset_pips,
                                       releases=releases, override_pips=override_pips,
                                       cost_mult=cost_mult)
        eng = OrderEngine()
        day_bars = _bars_for_day(sub)
        eng.run(day_bars, strat)
        if eng.position is not None and day_bars:
            last = day_bars[-1]
            eng.flatten(last.close, last.ts, reason="eod_safety")
            strat._book_if_closed(eng)
        for k, v in strat.day_r.items():
            day_r[k] = day_r.get(k, 0.0) + v
    idx = sorted(set(trading_days) | set(day_r.keys()))
    return pd.Series({d: risk_frac * day_r.get(d, 0.0) for d in idx})


_TRADE_LOG_COLS = ["pair", "ts", "side", "price", "qty", "order_id", "day_r"]


def build_trade_log(pair: str, fills, day_r: Optional[float] = None) -> pd.DataFrame:
    """One row per ``OrderEngine`` fill (entries AND exits) for one pair-day.

    Emits every fill in ``fills`` with no entry-only filter, tagged with the
    entry day's qty-independent R-multiple ``day_r``.
    """
    rows = [{"pair": pair, "ts": f.ts, "side": f.side, "price": f.price,
             "qty": f.qty, "order_id": f.order_id, "day_r": day_r} for f in fills]
    return pd.DataFrame(rows, columns=_TRADE_LOG_COLS)


def _pair_trade_log(pair: str, start: dt.date, end: dt.date, releases: pd.DataFrame,
                    risk_frac: float, tp_fraction: float, offset_pips: float,
                    override_pips: Optional[float] = None,
                    cost_mult: float = 1.0) -> pd.DataFrame:
    """Per-pair fills log, mirroring ``_pair_daily_returns``'s day loop.

    Runs a fresh ``OrderEngine`` per FX trading day and passes that day's FULL
    ``eng.fills`` list to ``build_trade_log`` -- every fill the engine recorded,
    entries and any exit fills alike -- tagged with the day's R-multiple
    (``strat.day_r``) before the next day's engine discards them. The FX ``date``
    is attached so the combined log stays sortable.
    """
    cols = ["date"] + _TRADE_LOG_COLS
    bars = load_fx_1min(pair, start, end)
    if bars.empty:
        logger.warning(f"[london_wf] no 1m data for {pair}; contributing empty trade log")
        return pd.DataFrame(columns=cols)
    atr_prior = _daily_atr_prior(pair, bars, start, end)
    fxday = _fx_day_values(bars.index)
    frames: List[pd.DataFrame] = []
    for day, sub in bars.groupby(fxday):
        strat = LondonBreakoutStrategy(pair, atr_d1=atr_prior, risk_frac=risk_frac,
                                       tp_fraction=tp_fraction, offset_pips=offset_pips,
                                       releases=releases, override_pips=override_pips,
                                       cost_mult=cost_mult)
        eng = OrderEngine()
        day_bars = _bars_for_day(sub)
        eng.run(day_bars, strat)
        if eng.position is not None and day_bars:
            last = day_bars[-1]
            eng.flatten(last.close, last.ts, reason="eod_safety")
            strat._book_if_closed(eng)
        frame = build_trade_log(pair, eng.fills, day_r=strat.day_r.get(day))
        frame.insert(0, "date", day)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def build_all_pairs_trade_log(pairs: List[str], start: dt.date, end: dt.date,
                              risk_frac: float = 0.005, tp_fraction: float = 0.5,
                              offset_pips: float = 3.0,
                              override_pips: Optional[float] = None,
                              cost_mult: float = 1.0) -> pd.DataFrame:
    """Fills-level trade log across pairs: one row per ``OrderEngine`` fill
    (entries AND exits), assembled from each pair's per-day ``build_trade_log``
    frames. Artifact backfill only -- not used by the walk-forward gate.
    """
    releases = generate_tier1_releases(start, end)
    cols = ["date"] + _TRADE_LOG_COLS
    frames = [_pair_trade_log(pair, start, end, releases, risk_frac, tp_fraction,
                              offset_pips, override_pips=override_pips, cost_mult=cost_mult)
              for pair in pairs]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def build_daily_returns(pairs: List[str], start: dt.date, end: dt.date,
                        risk_frac: float = 0.005, tp_fraction: float = 0.5,
                        offset_pips: float = 3.0,
                        override_pips: Optional[float] = None,
                        cost_mult: float = 1.0) -> pd.Series:
    releases = generate_tier1_releases(start, end)
    per_pair = []
    for pair in pairs:
        s = _pair_daily_returns(pair, start, end, releases, risk_frac, tp_fraction, offset_pips,
                                override_pips=override_pips, cost_mult=cost_mult)
        if not s.empty:
            per_pair.append(s.rename(pair))
    if not per_pair:
        return pd.Series(dtype=float)
    combined = pd.concat(per_pair, axis=1).fillna(0.0).mean(axis=1)
    combined.index = pd.DatetimeIndex(pd.to_datetime(list(combined.index)))
    return combined.sort_index()


def _wf_segments(returns: pd.Series, train_m: int, test_m: int,
                 step_m: int) -> List[Tuple[pd.Series, pd.Series]]:
    returns = returns.dropna()
    if returns.empty:
        return []
    start, end = returns.index.min(), returns.index.max()
    segs: List[Tuple[pd.Series, pd.Series]] = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_m)
        test_end = train_end + pd.DateOffset(months=test_m)
        is_seg = returns[(returns.index >= cursor) & (returns.index < train_end)]
        oos_seg = returns[(returns.index >= train_end) & (returns.index < test_end)]
        if oos_seg.size >= 10:
            segs.append((is_seg, oos_seg))
        if test_end > end:
            break
        cursor = cursor + pd.DateOffset(months=step_m)
    return segs


def run(config_path: str, trial_count: int,
        train_months: int = 36, test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    strat, dts = cfg["strategy"], cfg["dates"]
    pairs = list(strat["pairs"])
    params = dict(strat.get("params", {}))
    start_d, end_d = _as_date(dts["start"]), _as_date(dts["end"])

    override_pips = params.get("override_pips")
    combined = build_daily_returns(pairs, start_d, end_d,
                                   risk_frac=float(params.get("risk_frac", 0.005)),
                                   tp_fraction=float(params.get("tp_fraction", 0.5)),
                                   offset_pips=float(params.get("offset_pips", 3.0)),
                                   override_pips=(float(override_pips)
                                                  if override_pips is not None else None),
                                   cost_mult=float(params.get("cost_mult", 1.0)))
    segs = _wf_segments(combined, train_months, test_months, step_months)
    if len(segs) < 2:
        raise ValueError(f"need >=2 usable OOS windows, got {len(segs)}")

    per_window_oos = [oos.to_numpy(dtype=float) for _, oos in segs]
    oos_dated = pd.concat([oos for _, oos in segs]).sort_index(kind="stable")
    oos_dated = oos_dated[~oos_dated.index.duplicated(keep="first")]

    # Same-dates gate: restrict the headline strategy series to the dates the S&P
    # also trades, so the strategy OOS Sharpe and the S&P Sharpe are computed over
    # one identical date index ("beats S&P over the same OOS dates" literally).
    sp = load_sp500_daily_returns()
    common = oos_dated.index.intersection(sp.index)
    oos_dated = oos_dated.loc[oos_dated.index.isin(common)]

    arr = oos_dated.to_numpy(dtype=float)
    n = int(arr.size)
    sharpe = _annualized_sharpe(arr)
    per_window_is = [_annualized_sharpe(is_seg.to_numpy(dtype=float))
                     for is_seg, _ in segs if is_seg.size > 1]
    is_sharpe = float(np.nanmean(per_window_is)) if per_window_is else float("nan")
    is_oos_ratio = (is_sharpe / sharpe
                    if sharpe > 0 and not np.isnan(sharpe) else float("nan"))
    ser = pd.Series(arr)
    skew = float(ser.skew()) if n > 2 else 0.0
    kurt = float(ser.kurtosis()) + 3.0 if n > 3 else 3.0
    psr_val = psr(sharpe, 0.0, n, skew, kurt, periods_per_year=252)
    dsr_val = dsr(sharpe, [sharpe], n, skew, kurt, n_trials_project=trial_count,
                  periods_per_year=252)
    pbo_val = _compute_pbo(per_window_oos)

    n_trades = int((oos_dated != 0.0).sum())

    sp_sharpe = sp500_sharpe_over_dates(oos_dated.index, sp_returns=sp)
    sp_n = sp500_aligned_count(oos_dated.index, sp_returns=sp)
    if n > 0 and abs(sp_n - n) / n > 0.05:
        logger.warning(f"[london_wf] S&P aligned day count {sp_n} differs from strategy OOS day count {n} by >5%; comparison may not be apples-to-apples")
    corr = correlation_over_dates(oos_dated, sp_returns=sp)
    ir = information_ratio_vs_sp500(oos_dated, sp_returns=sp)
    beats = bool(sharpe > sp_sharpe)

    return {"n_oos_days": n, "n_windows": len(segs), "n_active_oos_days": n_trades,
            "oos_sharpe": sharpe, "is_sharpe": is_sharpe, "is_oos_ratio": is_oos_ratio,
            "sp500_sharpe": sp_sharpe, "sp500_n_days": sp_n, "beats_sp500": beats,
            "correlation_sp500": corr, "information_ratio_sp500": ir,
            "psr": psr_val, "dsr": dsr_val, "pbo": pbo_val, "skew": skew,
            "kurtosis_pearson": kurt, "trial_count": trial_count,
            "pairs": pairs, "full_days": int(combined.size),
            "oos_start": str(oos_dated.index.min().date()),
            "oos_end": str(oos_dated.index.max().date())}


def _write_report(r: Dict[str, Any], path: str = _REPORT_PATH) -> str:
    verdict = "PASS" if r["beats_sp500"] else "FAIL"
    lines = ["# #20 London Open Breakout Walk-Forward Readiness Report", "",
             "Generated by `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`.",
             "Primary gate: stitched OOS Sharpe > S&P 500 Sharpe over the same OOS dates.",
             "PSR/DSR/PBO are diagnostics, not gates (see the 2026-07-19 pre-registration).",
             "",
             f"## Verdict vs S&P: {verdict}", "",
             f"Pairs: {', '.join(r['pairs'])} (equal-risk, combined daily return).",
             "",
             "Costs: 1m fills are charged the MEASURED hour-of-week spread at the hour",
             "each fill lands on (half the round trip at entry, half at exit), from",
             "config/costs/fx_hour_of_week_spread.csv. Scale the stress leg with the",
             "`cost_mult` param.", "",
             "| Metric | Value |", "|---|---|",
             f"| OOS Sharpe (net) | {r['oos_sharpe']:.4f} |",
             f"| S&P Sharpe (aligned OOS dates) | {r['sp500_sharpe']:.4f} |",
             f"| S&P observation count | {r['sp500_n_days']} |",
             f"| Beats S&P | {r['beats_sp500']} |",
             f"| IS Sharpe (mean per-window) | {r['is_sharpe']:.4f} |",
             f"| IS/OOS Sharpe ratio | {r['is_oos_ratio']:.4f} |",
             f"| Correlation to S&P | {r['correlation_sp500']:.4f} |",
             f"| Information ratio vs S&P | {r['information_ratio_sp500']:.4f} |",
             f"| PSR (diag) | {r['psr']:.4f} |",
             f"| DSR (diag, trials={r['trial_count']}) | {r['dsr']:.4f} |",
             f"| PBO (diag) | {r['pbo']:.4f} |",
             f"| Skew / Kurtosis | {r['skew']:.4f} / {r['kurtosis_pearson']:.4f} |",
             f"| n_windows / n_oos_days | {r['n_windows']} / {r['n_oos_days']} |",
             f"| Active OOS days (>=1 trade) | {r['n_active_oos_days']} |",
             f"| Full daily series length | {r['full_days']} |",
             f"| OOS window | {r['oos_start']} .. {r['oos_end']} |", "",
             "Limitations: conservative 1m fills (worst-of trigger/open, adverse",
             "both-in-one-bar); half-spread slippage is a floor; approximate tier-1",
             "event dates; daily ATR(14) is sourced from the fx_daily cache where",
             "available and otherwise aggregated from the 1m bars per FX trading day",
             "(EURGBP/GBPJPY are absent from fx_daily).", ""]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def _run_trade_log_backfill(config_path: str, out_path: str) -> None:
    """Artifact backfill (NOT a gate re-run): emit the fills-level trade log.

    One primary/representative full-window pass via ``build_all_pairs_trade_log``,
    writing every ``OrderEngine`` fill (tagged with each entry day's R-multiple)
    to ``out_path``. Does not touch the walk-forward gate, PSR/DSR/PBO, or any
    verdict/report file.
    """
    import yaml
    from src.utils.run_status import RunStatus
    cfg = yaml.safe_load(Path(config_path).read_text())
    strat, dts = cfg["strategy"], cfg["dates"]
    pairs = list(strat["pairs"])
    params = dict(strat.get("params", {}))
    start_d, end_d = _as_date(dts["start"]), _as_date(dts["end"])
    override_pips = params.get("override_pips")

    with RunStatus("fx_london_breakout_trade_log", meta={"config": config_path}):
        trades = build_all_pairs_trade_log(pairs, start_d, end_d,
                                           risk_frac=float(params.get("risk_frac", 0.005)),
                                           tp_fraction=float(params.get("tp_fraction", 0.5)),
                                           offset_pips=float(params.get("offset_pips", 3.0)),
                                           override_pips=(float(override_pips)
                                                          if override_pips is not None else None),
                                           cost_mult=float(params.get("cost_mult", 1.0)))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out, index=False)

    logger.info(f"[london_wf] trade log backfill: {len(trades)} fills -> {out_path}")


def main() -> None:
    import argparse
    from src.utils.run_status import RunStatus
    try:
        from src.experiments import n_trials_project_wide
        base_trials = int(n_trials_project_wide())
    except Exception:
        base_trials = 0

    parser = argparse.ArgumentParser(description="#20 London Breakout walk-forward + S&P gate")
    parser.add_argument("--config", default="config/backtesting/fx_london_breakout.yaml")
    parser.add_argument("--report", default=_REPORT_PATH)
    parser.add_argument("--trade-log", default=None,
                        help="Artifact backfill mode: skip the gate, write the fills-level "
                             "entry log to this path instead (e.g. "
                             "output/backtests/fx/LondonBreakout/2011-01-01_to_2026-04-01/"
                             "trades.csv)")
    args = parser.parse_args()

    if args.trade_log is not None:
        _run_trade_log_backfill(args.config, args.trade_log)
        return

    trial_count = base_trials + 1  # this walk-forward is one new project-wide trial

    with RunStatus("fx_london_breakout_walkforward", meta={"config": args.config}):
        result = run(args.config, trial_count)
        report_path = _write_report(result, args.report)

    logger.info(f"[london_wf] oos_sharpe={result['oos_sharpe']:.4f} "
                f"sp500={result['sp500_sharpe']:.4f} beats={result['beats_sp500']} "
                f"dsr={result['dsr']:.4f} pbo={result['pbo']:.4f} "
                f"n_oos={result['n_oos_days']} active={result['n_active_oos_days']}")
    logger.info(f"[london_wf] wrote {report_path}")


if __name__ == "__main__":
    main()
