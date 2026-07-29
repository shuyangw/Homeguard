"""Wave-0 Group A data-property measurements for the equity-options slate.

Four diagnostics:
  D-014/042  gap-continuation and first-hour-trend effect size (underlying)
  D-040a     weekend realized-variance share
  D-013/030  drawdown-shape census (gap vs grind) after regime downgrades
  D-050a     realized-vol timing around regime-classifier transitions

These measure conditional means / variances / episode counts of the UNDERLYING
price series and the regime state. There is no position, no fill, no equity
curve and no performance metric anywhere in this module.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

REGIME_RANK = {
    "STRONG_BULL": 4,
    "WEAK_BULL": 3,
    "SIDEWAYS": 2,
    "UNPREDICTABLE": 1,
    "BEAR": 0,
}

BPS = 1e4


# --------------------------------------------------------------------------
# D-014 / D-042 : event extraction
# --------------------------------------------------------------------------

def _clean(marks: pd.DataFrame) -> pd.DataFrame:
    return marks[~marks["quarantined"].astype(bool)].reset_index(drop=True)


def gap_continuation_events(marks: pd.DataFrame,
                            gap_threshold: float = 0.005) -> pd.DataFrame:
    """OPT-014 arm: open gap >= +/-0.5% with a 30-minute confirmation.

    gap        = log(open_0930_t / c_snap_{t-1})
    qualify    = |gap| >= gap_threshold
    confirm    = gap up: close of the 10:00 bar > opening-range high
                 gap dn: close of the 10:00 bar < opening-range low
    entry      = close of the 10:01 bar   (decide bar t, enter bar t+1)
    exit       = close of the registered snapshot bar (15:45)
    signed_bps = sign(gap) * log(exit/entry) * 1e4
    """
    m = _clean(marks)
    prev_snap = m["c_snap"].shift(1)
    prev_date = m["session_date"].shift(1)
    gap = np.log(m["open_0930"] / prev_snap)

    up = gap >= gap_threshold
    dn = gap <= -gap_threshold
    qualify = up | dn
    confirm = np.where(up, m["c_1000"] > m["or_high"],
                       np.where(dn, m["c_1000"] < m["or_low"], False))

    sel = qualify & pd.Series(confirm, index=m.index) & prev_snap.notna()
    ev = m.loc[sel, ["session_date", "c_1001", "c_snap", "c_realclose"]].copy()
    ev["prev_session_date"] = prev_date[sel].values
    ev["gap"] = gap[sel].values
    ev["direction"] = np.sign(gap[sel].values).astype(int)
    ev = ev.rename(columns={"c_1001": "entry", "c_snap": "exit"})
    ev["signed_bps"] = ev["direction"] * np.log(ev["exit"] / ev["entry"]) * BPS
    ev["signed_bps_realclose"] = (ev["direction"]
                                  * np.log(ev["c_realclose"] / ev["entry"]) * BPS)
    ev["arm"] = "gap_continuation"
    return ev.reset_index(drop=True)


def first_hour_trend_events(marks: pd.DataFrame,
                            threshold: float = 0.0035) -> pd.DataFrame:
    """OPT-042 arm: first-hour move >= +/-0.35% -> rest of day.

    first_hour = log(c_1030 / open_0930); entry = close of the 10:31 bar;
    exit = close of the registered snapshot bar (15:45).
    """
    m = _clean(marks)
    fh = np.log(m["c_1030"] / m["open_0930"])
    sel = fh.abs() >= threshold
    ev = m.loc[sel, ["session_date", "c_1031", "c_snap", "c_realclose"]].copy()
    ev["first_hour"] = fh[sel].values
    ev["direction"] = np.sign(fh[sel].values).astype(int)
    ev = ev.rename(columns={"c_1031": "entry", "c_snap": "exit"})
    ev["signed_bps"] = ev["direction"] * np.log(ev["exit"] / ev["entry"]) * BPS
    ev["signed_bps_realclose"] = (ev["direction"]
                                  * np.log(ev["c_realclose"] / ev["entry"]) * BPS)
    ev["arm"] = "first_hour_trend"
    return ev.reset_index(drop=True)


def summarize_signed_returns(x, haircut_bps: float = 2.0,
                             n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Descriptive stats + bootstrap CI on the mean of a signed-bps sample."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n == 0:
        keys = ["mean_bps_gross", "mean_bps_net", "median_bps", "std_bps",
                "t_stat", "hit_rate", "ci_low_gross", "ci_high_gross",
                "ci_low_net", "ci_high_net"]
        out = {k: float("nan") for k in keys}
        out["n"] = 0
        return out
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": n,
        "mean_bps_gross": mean,
        "mean_bps_net": mean - haircut_bps,
        "median_bps": float(np.median(x)),
        "std_bps": std,
        "t_stat": float(mean / (std / np.sqrt(n))) if n > 1 and std > 0 else float("nan"),
        "hit_rate": float((x > 0).mean()),
        "ci_low_gross": float(lo),
        "ci_high_gross": float(hi),
        "ci_low_net": float(lo - haircut_bps),
        "ci_high_net": float(hi - haircut_bps),
    }


# --------------------------------------------------------------------------
# D-040a : weekend realized-variance share
# --------------------------------------------------------------------------

def weekend_variance_stats(marks: pd.DataFrame) -> Dict[str, object]:
    """Variance of session-to-session gap returns, split weekend vs weekday.

    gap return = log(open_0930 of session t / c_snap of session t-1).
    A pair is a WEEKEND pair when the two sessions fall in different ISO weeks;
    the CONTROL set is same-week pairs exactly 1 calendar day apart.
    """
    m = _clean(marks).copy()
    d = pd.to_datetime(m["session_date"])
    m["ts"] = d
    m["gap_ret"] = np.log(m["open_0930"] / m["c_snap"].shift(1))
    m["from_date"] = m["session_date"].shift(1)
    m["calendar_days"] = (d - d.shift(1)).dt.days
    iso = d.dt.isocalendar()
    same_week = (iso["week"].values == iso["week"].shift(1).values) & \
                (iso["year"].values == iso["year"].shift(1).values)
    m["is_weekend"] = ~pd.Series(same_week, index=m.index)

    gaps = m.iloc[1:][["from_date", "session_date", "calendar_days", "gap_ret",
                       "is_weekend"]].copy()
    gaps = gaps.rename(columns={"session_date": "to_date"})
    gaps = gaps[np.isfinite(gaps["gap_ret"])]
    gaps["is_weekend"] = gaps["is_weekend"].astype(bool)

    wk = gaps[gaps["is_weekend"]]["gap_ret"].to_numpy()
    ctl = gaps[(~gaps["is_weekend"]) & (gaps["calendar_days"] == 1)]["gap_ret"].to_numpy()

    cc = np.log(m["c_snap"] / m["c_snap"].shift(1)).to_numpy()
    cc = cc[np.isfinite(cc)]
    intraday_rv = m["intraday_rv"].to_numpy()
    intraday_rv = intraday_rv[np.isfinite(intraday_rv)]

    var_wk = float(np.var(wk, ddof=1)) if len(wk) > 1 else float("nan")
    var_ctl = float(np.var(ctl, ddof=1)) if len(ctl) > 1 else float("nan")
    var_cc = float(np.var(cc, ddof=1)) if len(cc) > 1 else float("nan")
    var_id = float(intraday_rv.mean()) if len(intraday_rv) else float("nan")
    mean_wk_cal_days = float(np.mean(
        gaps[gaps["is_weekend"]]["calendar_days"])) if len(wk) else float("nan")

    return {
        "gaps": gaps.reset_index(drop=True),
        "n_weekend": int(len(wk)),
        "n_control_overnight": int(len(ctl)),
        "n_sessions": int(len(m)),
        "weekend_var": var_wk,
        "control_overnight_var": var_ctl,
        "session_var_close_to_close": var_cc,
        "session_var_intraday_rv": var_id,
        "mean_weekend_calendar_days": mean_wk_cal_days,
        "ratio_weekend_over_3x_cc": var_wk / (3.0 * var_cc) if var_cc else float("nan"),
        "ratio_weekend_over_3x_intraday": (var_wk / (3.0 * var_id)
                                           if var_id else float("nan")),
        "ratio_weekend_over_3x_control_overnight": (
            var_wk / (3.0 * var_ctl) if var_ctl else float("nan")),
        "weekend_var_per_calendar_day": (var_wk / mean_wk_cal_days
                                         if mean_wk_cal_days else float("nan")),
        "weekday_var_per_calendar_day": var_ctl,
        "weekend_vs_weekday_per_calendar_day": (
            (var_wk / mean_wk_cal_days) / var_ctl if var_ctl else float("nan")),
        "weekend_share_of_total_gap_var": float(
            np.sum(wk ** 2) / (np.sum(wk ** 2) + np.sum(ctl ** 2))
        ) if len(wk) and len(ctl) else float("nan"),
    }


# --------------------------------------------------------------------------
# D-013 / D-030 : drawdown-shape census
# --------------------------------------------------------------------------

def _transition_frame(regime: pd.DataFrame) -> pd.DataFrame:
    r = regime[["date", "regime"]].dropna().sort_values("date").reset_index(drop=True)
    r["prev"] = r["regime"].shift(1)
    r["rank"] = r["regime"].map(REGIME_RANK)
    r["prev_rank"] = r["rank"].shift(1)
    r["transition"] = r["prev"] + "->" + r["regime"]
    return r


def regime_transitions(regime: pd.DataFrame) -> pd.DataFrame:
    """Any state change (session t state != session t-1 state)."""
    r = _transition_frame(regime)
    out = r[r["prev"].notna() & (r["regime"] != r["prev"])]
    return out[["date", "prev", "regime", "transition"]].reset_index(drop=True)


def regime_downgrade_triggers(regime: pd.DataFrame) -> pd.DataFrame:
    """Sessions where the regime rank STRICTLY decreases vs the prior session."""
    r = _transition_frame(regime)
    out = r[r["prev_rank"].notna() & (r["rank"] < r["prev_rank"])]
    return out[["date", "prev", "regime", "transition"]].reset_index(drop=True)


def drawdown_shape_census(marks: pd.DataFrame, triggers: pd.DataFrame,
                          window_sessions: int = 60,
                          meaningful_dd: float = 0.03) -> pd.DataFrame:
    """For each trigger, the max peak-to-trough decline in the next N sessions,
    decomposed into overnight-gap vs intraday components.

    Marks used: c_snap (the registered 15:45 mark) and open_0930.
    overnight_s = log(open_0930_s / c_snap_{s-1});  intraday_s = log(c_snap_s / open_0930_s)
    gap_share = |sum of NEGATIVE overnight| / (|sum neg overnight| + |sum neg intraday|)
    """
    m = _clean(marks).reset_index(drop=True)
    dates = pd.to_datetime(m["session_date"])
    snap = m["c_snap"].to_numpy()
    op = m["open_0930"].to_numpy()

    overnight = np.full(len(m), np.nan)
    overnight[1:] = np.log(op[1:] / snap[:-1])
    intraday = np.log(snap / op)

    rows = []
    for _, t in triggers.iterrows():
        pos = int(dates.searchsorted(pd.Timestamp(t["date"]), side="left"))
        if pos >= len(m):
            continue
        end = min(pos + window_sessions, len(m))
        w_snap = snap[pos:end]
        if len(w_snap) < 2:
            continue
        run_max = np.maximum.accumulate(w_snap)
        dd = w_snap / run_max - 1.0
        trough = int(np.argmin(dd))
        peak = int(np.argmax(w_snap[:trough + 1]))
        max_dd = float(-dd[trough])

        lo, hi = pos + peak + 1, pos + trough + 1
        on = overnight[lo:hi]
        idr = intraday[lo:hi]
        on = on[np.isfinite(on)]
        idr = idr[np.isfinite(idr)]
        neg_on = float(-on[on < 0].sum())
        neg_id = float(-idr[idr < 0].sum())
        denom = neg_on + neg_id
        gap_share = float(neg_on / denom) if denom > 0 else float("nan")
        rows.append({
            "trigger_date": pd.Timestamp(t["date"]).date(),
            "transition": t["transition"],
            "window_start": m["session_date"].iloc[pos],
            "window_end": m["session_date"].iloc[end - 1],
            "n_sessions_in_window": int(end - pos),
            "peak_date": m["session_date"].iloc[pos + peak],
            "trough_date": m["session_date"].iloc[pos + trough],
            "max_dd": max_dd,
            "meaningful": bool(max_dd > meaningful_dd),
            "sum_overnight_signed": float(on.sum()),
            "sum_intraday_signed": float(idr.sum()),
            "neg_overnight": neg_on,
            "neg_intraday": neg_id,
            "gap_share": gap_share,
            "shape": ("GAP" if (denom > 0 and gap_share > 0.5)
                      else ("GRIND" if denom > 0 else "NONE")),
        })
    return pd.DataFrame(rows)


def non_overlapping_subset(episodes: pd.DataFrame, marks: pd.DataFrame,
                           window_sessions: int = 60) -> pd.DataFrame:
    """Greedy: take triggers in date order, skip any within `window_sessions`
    sessions of the previously accepted trigger."""
    m = _clean(marks).reset_index(drop=True)
    dates = pd.to_datetime(m["session_date"])
    ep = episodes.sort_values("trigger_date").reset_index(drop=True)
    keep, last_pos = [], -10 ** 9
    for i, row in ep.iterrows():
        pos = int(dates.searchsorted(pd.Timestamp(row["trigger_date"]), side="left"))
        if pos - last_pos >= window_sessions:
            keep.append(i)
            last_pos = pos
    return ep.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# D-050a : realized-vol timing around classifier transitions
# --------------------------------------------------------------------------

def rv_event_study(rv: pd.Series, transition_dates: pd.DatetimeIndex,
                   offsets: Iterable[int] = range(-10, 11),
                   trailing_window: int = 60,
                   pre_post: int = 5) -> Dict[str, object]:
    """Event study of session realized variance around classifier transitions.

    RV at each offset is normalised by the event's TRAILING mean RV over
    sessions [t-trailing_window, t-1], so the SHAPE (where RV peaks) is visible.
    Also returns the paired pre/post comparison over [t-5,t-1] vs [t+1,t+5].
    """
    rv = rv.dropna().sort_index()
    idx = rv.index
    vals = rv.to_numpy()
    offsets = list(offsets)

    prof_rows = {o: [] for o in offsets}
    pre_list, post_list = [], []
    used_dates, skipped = [], 0

    for d in pd.DatetimeIndex(transition_dates):
        pos = int(idx.searchsorted(d, side="left"))
        if pos >= len(idx):
            skipped += 1
            continue
        if pos - trailing_window < 0 or pos + max(offsets) >= len(idx) \
                or pos + min(offsets) < 0:
            skipped += 1
            continue
        base = float(vals[pos - trailing_window:pos].mean())
        if not np.isfinite(base) or base <= 0:
            skipped += 1
            continue
        for o in offsets:
            prof_rows[o].append(vals[pos + o] / base)
        pre_list.append(float(vals[pos - pre_post:pos].mean()))
        post_list.append(float(vals[pos + 1:pos + 1 + pre_post].mean()))
        used_dates.append(idx[pos])

    if not used_dates:
        return {"n_events": 0, "n_skipped": skipped, "profile": pd.DataFrame(),
                "peak_offset_mean": None, "peak_offset_median": None,
                "mean_ratio_post_pre": float("nan"),
                "median_ratio_post_pre": float("nan"),
                "wilcoxon_p": float("nan"), "ttest_log_p": float("nan"),
                "pairs": pd.DataFrame()}

    profile = pd.DataFrame({
        "offset": offsets,
        "mean_norm": [float(np.mean(prof_rows[o])) for o in offsets],
        "median_norm": [float(np.median(prof_rows[o])) for o in offsets],
        "n": [len(prof_rows[o]) for o in offsets],
    })

    pre = np.array(pre_list)
    post = np.array(post_list)
    ok = np.isfinite(pre) & np.isfinite(post) & (pre > 0) & (post > 0)
    ratio = post[ok] / pre[ok]
    try:
        w_p = float(stats.wilcoxon(post[ok], pre[ok]).pvalue)
    except ValueError:
        w_p = float("nan")
    t_p = float(stats.ttest_rel(np.log(post[ok]), np.log(pre[ok])).pvalue) \
        if ok.sum() > 1 else float("nan")

    return {
        "n_events": int(len(used_dates)),
        "n_skipped": int(skipped),
        "n_pairs": int(ok.sum()),
        "profile": profile,
        "peak_offset_mean": int(profile.loc[profile["mean_norm"].idxmax(), "offset"]),
        "peak_offset_median": int(profile.loc[profile["median_norm"].idxmax(), "offset"]),
        "mean_ratio_post_pre": float(ratio.mean()) if len(ratio) else float("nan"),
        "median_ratio_post_pre": float(np.median(ratio)) if len(ratio) else float("nan"),
        "mean_pre": float(pre[ok].mean()) if ok.sum() else float("nan"),
        "mean_post": float(post[ok].mean()) if ok.sum() else float("nan"),
        "wilcoxon_p": w_p,
        "ttest_log_p": t_p,
        "pairs": pd.DataFrame({"date": np.array(used_dates)[ok],
                               "rv_pre": pre[ok], "rv_post": post[ok],
                               "ratio": ratio}),
    }
