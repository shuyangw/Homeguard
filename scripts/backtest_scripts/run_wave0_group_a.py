"""Driver for Wave-0 Group A data-property diagnostics (equity-options slate).

Runs D-014/042, D-040a, D-013/030, D-050a and persists every numeric output
under output/wave0/groupA/ so a later agent can re-read without re-running.

MEASUREMENT ONLY. No strategy simulation, no positions, no fills, no equity
curve, no performance metric.

Usage:
    python scripts/backtest_scripts/run_wave0_group_a.py [--stage marks|all]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtesting.diagnostics.session_bars import (
    build_session_marks,
    infer_bar_label_convention,
    load_minute_bars,
    session_close_schedule,
)
from src.backtesting.diagnostics.wave0_group_a import (
    drawdown_shape_census,
    first_hour_trend_events,
    gap_continuation_events,
    non_overlapping_subset,
    regime_downgrade_triggers,
    regime_transitions,
    rv_event_study,
    summarize_signed_returns,
    weekend_variance_stats,
)
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT = Path("output/wave0/groupA")
SYMBOLS = ("SPY", "QQQ")
YEARS = range(2016, 2026)
WINDOW_START = pd.Timestamp("2016-01-01")
WINDOW_END = pd.Timestamp("2025-12-31")
SLICES = {"full_2016plus": WINDOW_START,
          "post_2023": pd.Timestamp("2023-01-01"),
          "post_2024": pd.Timestamp("2024-01-01")}
HAIRCUT_BPS = 2.0
GATE_BPS = 15.0
REGIME_PATH = "H:/Stock_Data/alt_data/regime/regime_state_daily.parquet"


def _slice(df: pd.DataFrame, col: str, start: pd.Timestamp) -> pd.DataFrame:
    d = pd.to_datetime(df[col])
    return df[(d >= start) & (d <= WINDOW_END)]


def build_marks(status) -> dict:
    schedule = session_close_schedule("2015-12-01", "2026-01-31")
    meta = {}
    for sym in SYMBOLS:
        fp = OUT / f"session_marks_{sym}.parquet"
        parts, conv_ev = [], None
        for year in YEARS:
            bars = load_minute_bars(sym, [year])
            if conv_ev is None:
                conv, ev = infer_bar_label_convention(bars["ny"])
                conv_ev = (conv, ev)
            parts.append(build_session_marks(bars, schedule))
            del bars
            status.heartbeat(note=f"{sym} marks {year}")
        marks = pd.concat(parts, ignore_index=True).sort_values("session_date")
        marks = marks.drop_duplicates(subset="session_date").reset_index(drop=True)
        d = pd.to_datetime(marks["session_date"])
        marks = marks[(d >= WINDOW_START) & (d <= WINDOW_END)].reset_index(drop=True)
        marks.to_parquet(fp, index=False)
        meta[sym] = {
            "bar_label_convention": conv_ev[0],
            "bar_label_evidence": conv_ev[1],
            "n_sessions": int(len(marks)),
            "n_quarantined": int(marks["quarantined"].sum()),
            "n_early_close": int(marks["is_early_close"].sum()),
            "first_session": str(marks["session_date"].min()),
            "last_session": str(marks["session_date"].max()),
            "path": str(fp),
        }
        logger.info(f"[+] {sym} marks: {meta[sym]}")
    (OUT / "marks_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_marks(sym: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"session_marks_{sym}.parquet")


def run_d014_042() -> None:
    all_events, rows = [], []
    for sym in SYMBOLS:
        marks = load_marks(sym)
        arms = {
            "gap_continuation": gap_continuation_events(marks, gap_threshold=0.005),
            "first_hour_trend": first_hour_trend_events(marks, threshold=0.0035),
        }
        for arm, ev in arms.items():
            ev = ev.copy()
            ev["symbol"] = sym
            all_events.append(ev)
            for sname, start in SLICES.items():
                sub = _slice(ev, "session_date", start)
                s = summarize_signed_returns(sub["signed_bps"].to_numpy(),
                                             haircut_bps=HAIRCUT_BPS,
                                             n_boot=10000, seed=42)
                s_rc = summarize_signed_returns(
                    sub["signed_bps_realclose"].to_numpy(),
                    haircut_bps=HAIRCUT_BPS, n_boot=10000, seed=42)
                years = pd.to_datetime(sub["session_date"]).dt.year
                n_years = years.nunique() or 1
                rows.append({
                    "arm": arm, "symbol": sym, "slice": sname,
                    "window_start": str(start.date()),
                    "window_end": str(WINDOW_END.date()),
                    **s,
                    "events_per_year": len(sub) / n_years,
                    "mean_bps_net_realclose": s_rc["mean_bps_net"],
                    "n_realclose": s_rc["n"],
                    "gate_threshold_bps": GATE_BPS,
                    "gate_pass": (bool(s["mean_bps_net"] >= GATE_BPS)
                                  if np.isfinite(s["mean_bps_net"]) else False),
                })
    events = pd.concat(all_events, ignore_index=True)
    events.to_parquet(OUT / "d014_042_events.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "d014_042_summary.csv", index=False)

    per_year = (events.assign(year=pd.to_datetime(events["session_date"]).dt.year)
                .groupby(["symbol", "arm", "year"])
                .agg(n=("signed_bps", "size"),
                     mean_bps_gross=("signed_bps", "mean"))
                .reset_index())
    per_year["mean_bps_net"] = per_year["mean_bps_gross"] - HAIRCUT_BPS
    per_year.to_csv(OUT / "d014_042_per_year.csv", index=False)
    logger.info(f"[+] D-014/042 written: {len(events)} events, {len(summary)} rows")


def run_d014_042_prior_close_sensitivity() -> None:
    """Sensitivity: the whole gap arm re-anchored on the REAL session close
    (15:59) instead of the registered 15:45 snapshot -- both the prior-close
    gap reference AND the exit. The GATE is applied to the registered 15:45
    version in run_d014_042(); this is reported alongside it, not instead."""
    rows = []
    for sym in SYMBOLS:
        m = load_marks(sym).copy()
        m["c_snap"] = m["c_realclose"]
        ev = gap_continuation_events(m, gap_threshold=0.005)
        for sname, start in SLICES.items():
            sub = _slice(ev, "session_date", start)
            s = summarize_signed_returns(sub["signed_bps"].to_numpy(),
                                         haircut_bps=HAIRCUT_BPS,
                                         n_boot=10000, seed=42)
            rows.append({"arm": "gap_continuation_realclose_anchored",
                         "symbol": sym, "slice": sname, **s})
    pd.DataFrame(rows).to_csv(OUT / "d014_sensitivity_realclose_anchor.csv",
                              index=False)
    logger.info("[+] D-014 prior-close sensitivity written")


def run_d040a() -> None:
    rows = []
    for sym in SYMBOLS:
        marks = load_marks(sym)
        for sname, start in SLICES.items():
            sub = _slice(marks, "session_date", start)
            out = weekend_variance_stats(sub)
            gaps = out.pop("gaps")
            if sname == "full_2016plus":
                gaps.to_parquet(OUT / f"d040a_gaps_{sym}.parquet", index=False)
            rows.append({"symbol": sym, "slice": sname,
                         "window_start": str(start.date()),
                         "window_end": str(WINDOW_END.date()),
                         "n_quarantined_sessions": int(sub["quarantined"].sum()),
                         **out})
    pd.DataFrame(rows).to_csv(OUT / "d040a_summary.csv", index=False)
    logger.info("[+] D-040a written")


def _dd_census_summary(ep: pd.DataFrame, label: str) -> dict:
    if ep.empty:
        return {"subset": label, "n": 0}
    shp = ep["shape"].value_counts().to_dict()
    return {
        "subset": label,
        "n": int(len(ep)),
        "n_gap": int(shp.get("GAP", 0)),
        "n_grind": int(shp.get("GRIND", 0)),
        "n_none": int(shp.get("NONE", 0)),
        "n_meaningful_dd_gt_3pct": int(ep["meaningful"].sum()),
        "n_not_meaningful": int((~ep["meaningful"]).sum()),
        "mean_gap_share": float(ep["gap_share"].mean()),
        "median_gap_share": float(ep["gap_share"].median()),
        "mean_max_dd": float(ep["max_dd"].mean()),
        "median_max_dd": float(ep["max_dd"].median()),
        "p90_max_dd": float(ep["max_dd"].quantile(0.90)),
        "max_max_dd": float(ep["max_dd"].max()),
        "mean_gap_share_meaningful": float(ep.loc[ep["meaningful"], "gap_share"].mean()),
        "n_gap_meaningful": int(((ep["shape"] == "GAP") & ep["meaningful"]).sum()),
        "n_grind_meaningful": int(((ep["shape"] == "GRIND") & ep["meaningful"]).sum()),
    }


def run_d013_030() -> None:
    regime = pd.read_parquet(REGIME_PATH)
    regime = regime[(regime["date"] >= WINDOW_START) & (regime["date"] <= WINDOW_END)]
    trig = regime_downgrade_triggers(regime)
    trig.to_csv(OUT / "d013_030_triggers.csv", index=False)

    marks = load_marks("SPY")
    ep = drawdown_shape_census(marks, trig, window_sessions=60, meaningful_dd=0.03)
    ep.to_csv(OUT / "d013_030_episodes_all.csv", index=False)
    ep_no = non_overlapping_subset(ep, marks, window_sessions=60)
    ep_no.to_csv(OUT / "d013_030_episodes_nonoverlap.csv", index=False)

    pd.DataFrame([_dd_census_summary(ep, "all_triggers"),
                  _dd_census_summary(ep_no, "non_overlapping")]).to_csv(
        OUT / "d013_030_summary.csv", index=False)

    dec = pd.DataFrame({
        "decile": [f"p{int(q * 100)}" for q in np.arange(0, 1.01, 0.1)],
        "gap_share_all": [ep["gap_share"].quantile(q) for q in np.arange(0, 1.01, 0.1)],
        "gap_share_nonoverlap": [ep_no["gap_share"].quantile(q)
                                 for q in np.arange(0, 1.01, 0.1)],
        "max_dd_all": [ep["max_dd"].quantile(q) for q in np.arange(0, 1.01, 0.1)],
    })
    dec.to_csv(OUT / "d013_030_gap_share_deciles.csv", index=False)

    by_tr = (ep.groupby("transition")
             .agg(n=("gap_share", "size"),
                  n_gap=("shape", lambda s: int((s == "GAP").sum())),
                  n_grind=("shape", lambda s: int((s == "GRIND").sum())),
                  mean_gap_share=("gap_share", "mean"),
                  mean_max_dd=("max_dd", "mean"),
                  n_meaningful=("meaningful", "sum"))
             .reset_index().sort_values("n", ascending=False))
    by_tr.to_csv(OUT / "d013_030_by_transition.csv", index=False)
    logger.info(f"[+] D-013/030 written: {len(ep)} episodes, "
                f"{len(ep_no)} non-overlapping")


def _session_rv(sym: str) -> pd.Series:
    marks = load_marks(sym)
    marks = marks[~marks["quarantined"].astype(bool)]
    s = pd.Series(marks["intraday_rv"].to_numpy(),
                  index=pd.to_datetime(marks["session_date"]))
    return s.dropna()


def run_d050a() -> None:
    regime = pd.read_parquet(REGIME_PATH)
    regime = regime[(regime["date"] >= WINDOW_START) & (regime["date"] <= WINDOW_END)]
    tr = regime_transitions(regime)
    tr.to_csv(OUT / "d050a_transitions.csv", index=False)

    into_unpred = tr[tr["regime"] == "UNPREDICTABLE"]
    out_of_sb = tr[tr["prev"] == "STRONG_BULL"]
    opt050 = pd.concat([into_unpred, out_of_sb]).drop_duplicates(
        subset="date").sort_values("date")

    rv_spy = _session_rv("SPY")
    detector_rv = pd.Series(regime["realized_vol"].to_numpy(),
                            index=pd.to_datetime(regime["date"])).dropna()

    sets = {
        "all_transitions": tr["date"],
        "into_UNPREDICTABLE": into_unpred["date"],
        "out_of_STRONG_BULL": out_of_sb["date"],
        "opt050_pooled": opt050["date"],
    }
    rows = []
    for src_name, series in (("spy_intraday_rv", rv_spy),
                             ("detector_realized_vol", detector_rv)):
        for sname, dates in sets.items():
            res = rv_event_study(series, pd.DatetimeIndex(dates),
                                 offsets=range(-10, 11), trailing_window=60)
            prof = res.pop("profile")
            pairs = res.pop("pairs")
            if len(prof):
                prof.insert(0, "trigger_set", sname)
                prof.insert(0, "rv_source", src_name)
                prof.to_csv(OUT / f"d050a_profile_{src_name}_{sname}.csv", index=False)
            if len(pairs):
                pairs.to_parquet(
                    OUT / f"d050a_pairs_{src_name}_{sname}.parquet", index=False)
            rows.append({"rv_source": src_name, "trigger_set": sname, **res})
    pd.DataFrame(rows).to_csv(OUT / "d050a_summary.csv", index=False)
    logger.info("[+] D-050a written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["marks", "measure", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with RunStatus("wave0_groupA", meta={"stage": args.stage,
                                         "symbols": list(SYMBOLS),
                                         "window": "2016-01-01..2025-12-31"}) as status:
        if args.stage in ("marks", "all"):
            build_marks(status)
            status.heartbeat(note="marks built")
        if args.stage in ("measure", "all"):
            run_d014_042()
            run_d014_042_prior_close_sensitivity()
            status.heartbeat(note="D-014/042 done")
            run_d040a()
            status.heartbeat(note="D-040a done")
            run_d013_030()
            status.heartbeat(note="D-013/030 done")
            run_d050a()
            status.heartbeat(note="D-050a done")
    logger.info("[+] Wave-0 Group A complete")


if __name__ == "__main__":
    main()
