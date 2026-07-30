"""Wave 0 Group B1 diagnostics -- D-018, D-027, D-033, D-040b, D-048, D-049,
D-050b, D-011.

DATA-PROPERTY MEASUREMENTS ONLY. No strategy backtest, no P&L, no positions,
no fills, no equity curve, no Sharpe. Every number here is a property of the
option-chain / price / regime data.

All gates are PRE-REGISTERED and are applied exactly as written. Every
operationalization is registered in the code and echoed into the JSON output
BEFORE the measurement is read.

Outputs -> output/wave0/groupB1/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtesting.diagnostics import options_iv_state as ivs
from src.backtesting.diagnostics.session_bars import session_close_schedule
from src.data.options.derived_store import load_derived_table
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT = Path("output/wave0/groupB1")


def _load():
    d = {}
    for name in ("atm_iv_daily", "atm_per_expiry", "skew_daily",
                 "term_slope_daily", "iv_rank_daily", "rv_daily",
                 "put_iv_daily"):
        t = load_derived_table(name)
        if "session_date" in t.columns:
            t["session_date"] = pd.to_datetime(t["session_date"]).dt.date
        d[name] = t
    return d


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _window(dates) -> str:
    d = sorted(dates)
    return f"{d[0]} .. {d[-1]} (n={len(d)})" if len(d) else "EMPTY"


# ===========================================================================
# D-018 -- IV-rank state-transition matrix
# ===========================================================================

def d018(tabs) -> dict:
    res = {
        "registered_measurement": ("IV-rank state-transition matrix (does "
                                   "IVR>50 persist / lead rising vol?)"),
        "registered_gate": "Descriptive; frames the conditioning trap",
        "gates_candidate": "OPT-018 (short 0.16-delta strangle only when 1y IV rank > 50)",
        "verdict": "DESCRIPTIVE",
        "registered_operationalizations": [
            "IVR = iv_rank_1y on the 30-DTE constant-maturity ATM IV bucket, "
            "(iv - min)/(max - min) over the 252 sessions STRICTLY BEFORE t.",
            "'IVR>50' means iv_rank_1y > 0.50.",
            "Deciles = floor(10 * iv_rank_1y), capped at 9.",
            "Run length uses find_episodes(merge_gap_sessions=0) -- raw "
            "uninterrupted runs of the IVR>50 state.",
            "Conditioning-trap test: DELTA_RV = yz_20d(t+21) - yz_20d(t) "
            "(annualized Yang-Zhang vol on regular-session daily OHLC). "
            "t+21 is 21 SESSIONS ahead on the joined axis.",
        ],
        "roots": {},
    }
    rank = tabs["iv_rank_daily"]
    rv = tabs["rv_daily"]
    for root in ("SPY", "QQQ"):
        g = rank[(rank["root"] == root) & (rank["dte_bucket"] == 30)].copy()
        g = g.sort_values("session_date").reset_index(drop=True)
        r = rv[rv["root"] == root][["session_date", "yz_20d", "close"]]
        g = g.merge(r, on="session_date", how="left")
        v = g["iv_rank_1y"].to_numpy(dtype=float)
        ok = np.isfinite(v)
        n_missing = int((~ok).sum())

        dec = np.floor(np.where(ok, v, np.nan) * 10.0)
        dec = np.where(np.isfinite(dec), np.minimum(dec, 9), np.nan)

        def tmat(h):
            a, b = dec[:-h], dec[h:]
            m = np.isfinite(a) & np.isfinite(b)
            M = np.zeros((10, 10))
            for i, j in zip(a[m].astype(int), b[m].astype(int)):
                M[i, j] += 1
            rs = M.sum(axis=1, keepdims=True)
            return (M / np.where(rs == 0, np.nan, rs)), int(m.sum())

        m1, n1 = tmat(1)
        m21, n21 = tmat(21)

        hi = v > 0.50
        eps = ivs.find_episodes(g["session_date"].tolist(), hi, 0, valid=ok)
        durs = [e["duration"] for e in eps]
        uncond = float(hi[ok].mean())
        a, b = hi[:-21], hi[21:]
        mm = ok[:-21] & ok[21:]
        p_cond = float(b[mm & a].mean()) if (mm & a).sum() else float("nan")

        yz = g["yz_20d"].to_numpy(dtype=float)
        d_rv = np.full(len(g), np.nan)
        d_rv[:-21] = yz[21:] - yz[:-21]
        rel = np.full(len(g), np.nan)
        rel[:-21] = np.where(yz[:-21] > 0, yz[21:] / yz[:-21], np.nan)
        cond_m = ok & hi & np.isfinite(d_rv)
        all_m = ok & np.isfinite(d_rv)

        res["roots"][root] = {
            "window_used": _window(g.loc[ok, "session_date"]),
            "n_sessions_total": int(len(g)),
            "n_sessions_ivr_measurable": int(ok.sum()),
            "n_quarantined_ivr_not_measurable": n_missing,
            "transition_matrix_1session": np.round(m1, 4).tolist(),
            "transition_matrix_1session_n": n1,
            "transition_matrix_21session": np.round(m21, 4).tolist(),
            "transition_matrix_21session_n": n21,
            "diag_persistence_1session": float(np.nanmean(np.diag(m1))),
            "diag_persistence_21session": float(np.nanmean(np.diag(m21))),
            "unconditional_P_IVR_gt50": uncond,
            "P_IVR_gt50_at_t21_given_IVR_gt50_at_t": p_cond,
            "lift": (p_cond / uncond) if uncond > 0 else float("nan"),
            "n_episodes_IVR_gt50": len(eps),
            "mean_run_duration_sessions": float(np.mean(durs)) if durs else float("nan"),
            "median_run_duration_sessions": float(np.median(durs)) if durs else float("nan"),
            "max_run_duration_sessions": int(max(durs)) if durs else 0,
            "delta_yz20_fwd21_conditional_on_IVR_gt50": ivs.describe(pd.Series(d_rv[cond_m])),
            "delta_yz20_fwd21_unconditional": ivs.describe(pd.Series(d_rv[all_m])),
            "ratio_yz20_t21_over_t_conditional": ivs.describe(pd.Series(rel[ok & hi & np.isfinite(rel)])),
            "ratio_yz20_t21_over_t_unconditional": ivs.describe(pd.Series(rel[ok & np.isfinite(rel)])),
            "frac_conditional_rv_RISING": float((d_rv[cond_m] > 0).mean()) if cond_m.sum() else float("nan"),
            "frac_unconditional_rv_RISING": float((d_rv[all_m] > 0).mean()) if all_m.sum() else float("nan"),
        }
    return res


# ===========================================================================
# D-027 -- skew-percentile persistence + conditional crash frequency
# ===========================================================================

def d027(tabs) -> dict:
    res = {
        "registered_measurement": ("Skew-percentile state persistence + realized "
                                   "crash frequency conditional on steepness"),
        "registered_gate": "Descriptive",
        "gates_candidate": ("OPT-027 (sell 0.25/0.10-delta put spread when "
                            "25-delta skew > trailing-2y 70th percentile)"),
        "verdict": "DESCRIPTIVE",
        "registered_operationalizations": [
            "skew_25d = iv_25d_put - iv_25d_call at the 45-DTE bucket, both legs "
            "taken from the SAME representative listed expiry (nearest 45 DTE, "
            "tolerance max(7, 0.25*45)=11.25 days).",
            "'steep' = trailing-2y (504 sessions, STRICTLY BEFORE t) percentile "
            "of skew_25d > 0.70.",
            "Crash horizon = 45 SESSIONS forward; the underlying series is the "
            "15:45 snapshot underlying_px carried on the chain rows (NOT the "
            "official close) so the QQQ pre-2016 window is covered.",
            "Forward drawdown = min over the forward window of "
            "(price / running-max-of-window - 1). This is a price-series "
            "measurement, not a P&L.",
            "Run length uses find_episodes(merge_gap_sessions=0).",
        ],
        "roots": {},
    }
    skew = tabs["skew_daily"]
    pe = tabs["atm_per_expiry"]
    ul = (pe.groupby(["root", "session_date"], as_index=False)["underlying_px"]
            .first())
    for root in ("SPY", "QQQ"):
        g = skew[(skew["root"] == root) & (skew["dte_bucket"] == 45)].copy()
        g = g.sort_values("session_date").reset_index(drop=True)
        g = g.merge(ul[ul["root"] == root][["session_date", "underlying_px"]],
                    on="session_date", how="left")
        p = ivs.trailing_percentile(g["skew_25d"], 504).to_numpy()
        ok = np.isfinite(p)
        steep = p > 0.70
        eps = ivs.find_episodes(g["session_date"].tolist(), steep, 0, valid=ok)
        durs = [e["duration"] for e in eps]
        uncond = float(steep[ok].mean())
        a, b = steep[:-21], steep[21:]
        mm = ok[:-21] & ok[21:]
        p_cond = float(b[mm & a].mean()) if (mm & a).sum() else float("nan")

        mdd = ivs.forward_max_drawdown(g["underlying_px"], 45).to_numpy()
        cond = ok & steep & np.isfinite(mdd)
        base = ok & np.isfinite(mdd)
        thr = {}
        for t in (-0.05, -0.10, -0.20):
            thr[f"{int(t*100)}pct"] = {
                "freq_conditional_on_steep": float((mdd[cond] <= t).mean()) if cond.sum() else float("nan"),
                "freq_unconditional": float((mdd[base] <= t).mean()) if base.sum() else float("nan"),
                "n_conditional": int(cond.sum()), "n_unconditional": int(base.sum()),
            }
        res["roots"][root] = {
            "window_used": _window(g.loc[ok, "session_date"]),
            "n_sessions_with_skew": int(len(g)),
            "n_sessions_pctile_measurable": int(ok.sum()),
            "n_quarantined_pctile_warmup_or_missing": int((~ok).sum()),
            "unconditional_P_steep": uncond,
            "P_steep_at_t21_given_steep_at_t": p_cond,
            "lift": (p_cond / uncond) if uncond > 0 else float("nan"),
            "n_episodes_steep": len(eps),
            "mean_run_duration_sessions": float(np.mean(durs)) if durs else float("nan"),
            "median_run_duration_sessions": float(np.median(durs)) if durs else float("nan"),
            "max_run_duration_sessions": int(max(durs)) if durs else 0,
            "fwd45_drawdown_conditional_on_steep": ivs.describe(pd.Series(mdd[cond])),
            "fwd45_drawdown_unconditional": ivs.describe(pd.Series(mdd[base])),
            "crash_frequency": thr,
            "skew_25d_level": ivs.describe(g["skew_25d"]),
        }
    return res


# ===========================================================================
# D-033 -- backwardation episode count
# ===========================================================================

def d033(tabs) -> dict:
    res = {
        "registered_measurement": ("Non-overlapping M2/M1 < 0.97 episode count, "
                                   "2012-2026 (2007-2026 if the ORATS extension "
                                   "is purchased)"),
        "registered_gate": (">= 12 episodes -> Wave 3 backtest; < 12 -> route to "
                            "forward paper validation, do not spend the "
                            "historical sample"),
        "gates_candidate": "OPT-033",
        "registered_operationalizations": [
            "PRIMARY non-overlapping rule (registered BEFORE counting): an "
            "episode STARTS on the first session with slope < 0.97 and ENDS on "
            "the last session before slope >= 0.97; consecutive episodes "
            "separated by FEWER THAN 30 sessions are MERGED (the candidate "
            "holds a 60-DTE structure).",
            "Sensitivities also reported: raw (no merge) and 60-session merge.",
            "slope = m2_atm_iv / m1_atm_iv on STANDARD MONTHLY expiries only.",
            "ORATS was DEFERRED (2026-07-27) so the window is the OWNED window; "
            "the SPY greeks boundary binds SPY to 2017+. QQQ is the only "
            "full-history root (2012-06+).",
        ],
        "roots": {},
    }
    t = tabs["term_slope_daily"]
    for root in ("SPY", "QQQ"):
        g = t[t["root"] == root].sort_values("session_date").reset_index(drop=True)
        s = g["slope"].to_numpy(dtype=float)
        ok = np.isfinite(s)
        flag = s < 0.97
        counts = {}
        for label, gap in (("raw_no_merge", 0), ("merge_30", 30), ("merge_60", 60)):
            eps = ivs.find_episodes(g["session_date"].tolist(), flag, gap, valid=ok)
            counts[label] = {
                "n_episodes": len(eps),
                "episodes": [{"start": str(e["start"]), "end": str(e["end"]),
                              "duration_sessions": e["duration"],
                              "min_slope": float(np.nanmin(s[e["start_idx"]:e["end_idx"] + 1]))}
                             for e in eps],
            }
        n = counts["merge_30"]["n_episodes"]
        res["roots"][root] = {
            "window_used": _window(g.loc[ok, "session_date"]),
            "n_sessions": int(ok.sum()),
            "n_sessions_backwardated": int((flag & ok).sum()),
            "frac_sessions_backwardated": float((flag & ok).mean()),
            "slope_distribution": ivs.describe(g["slope"]),
            "episode_counts": counts,
            "PRIMARY_n_episodes_merge30": n,
            "gate_result": "PASS" if n >= 12 else "FAIL",
            "gate_routing": ("Wave 3 backtest eligible" if n >= 12
                             else "forward paper validation; do not spend the "
                                  "historical sample"),
        }
    res["gate_read_on"] = (
        "QQQ is the primary read: it is the ONLY root with full-history usable "
        "greeks (2012-06+). SPY is reported but is bounded to 2017+ by the "
        "ETF/index greeks boundary and additionally loses most of 2017 to a "
        "source-store coverage gap, so its episode count is a strict "
        "UNDERCOUNT of the 2012-2026 window the gate text names.")
    return res


# ===========================================================================
# D-040b -- Friday vs Thursday term-adjusted ATM IV discount
# ===========================================================================

# D-040a companion measurement (SPY 1m, 2016-01-04..2025-12-31), used as the
# SECONDARY empirical denominator. Units: variance of log returns.
D040A = {
    "weekend_var": 8.31e-5,
    "control_overnight_var": 4.67e-5,
    "mean_intraday_session_rv": 7.22e-5,
    "close_to_close_session_var": 1.203e-4,
    "weekend_var_per_cal_day": 2.65e-5,
    "weekday_var_per_cal_day": 4.67e-5,
    "weekend_to_weekday_per_cal_day": 0.562,
}


def _sessions_remaining(sess_dates, expiry, cal_sorted):
    """Trading sessions strictly after `sess` up to and including `expiry`."""
    import bisect
    out = []
    for s in sess_dates:
        i = bisect.bisect_right(cal_sorted, s)
        j = bisect.bisect_right(cal_sorted, expiry)
        out.append(j - i)
    return out


def d040b(tabs) -> dict:
    res = {
        "registered_measurement": "Friday vs Thursday term-adjusted ATM IV discount",
        "registered_gate": ("Proceed only if measured Friday discount < 50% of "
                            "calendar-day theta differential; else DROP 040 "
                            "(expected)"),
        "gates_candidate": ("OPT-040 (sell 7-10 DTE 0.16-delta strangle Friday "
                            "15:45, close Monday 09:45)"),
        "registered_operationalizations": [
            "PAIR: consecutive trading sessions (Thursday t, Friday t+1) with no "
            "session between them.",
            "TERM ADJUSTMENT (registered): the pair is compared on the SAME "
            "LISTED EXPIRY, so the only term change is the 1 calendar day / 1 "
            "session that elapses. Under the calendar-uniform null (variance "
            "accrues uniformly per CALENDAR day) the quoted annualized IV is "
            "INVARIANT to that elapse, so the raw same-expiry Friday-minus-"
            "Thursday IV difference IS the term-adjusted difference.",
            "Expiry selected per pair: the listed expiry whose FRIDAY dte is in "
            "[7, 10] and closest to 8 (the OPT-040 range). Robustness read: "
            "Friday dte in [12, 16].",
            "ATM IV = IV at the strike nearest underlying_px, call and put "
            "averaged (build_atm_iv_daily's per-expiry surface).",
            "DENOMINATOR (a) PRIMARY -- theoretical pure-calendar-day theta "
            "differential: iv is quoted per CALENDAR year (T = D/365) while "
            "variance accrues only in TRADING time (N sessions). The FULL "
            "trading-time discount for the pair is "
            "iv_Thu * (sqrt((N_Fri/D_Fri)/(N_Thu/D_Thu)) - 1), a NEGATIVE "
            "number. D = calendar days to expiry, N = trading sessions strictly "
            "after the session through expiry.",
            "DENOMINATOR (b) SECONDARY -- empirical: identical formula with the "
            "session count N replaced by the D-040a variance-weighted session "
            "count V, where a weekend-spanning close-to-close leg carries "
            "(weekend_var + mean_intraday_session_rv) / "
            "close_to_close_session_var = 1.291 session-units and every other "
            "leg carries 1.000.",
            "RATIO = measured (iv_Fri - iv_Thu) / denominator. The gate is read "
            "on (a). Both are computed and BOTH are reported; neither was "
            "chosen after seeing which passes.",
            "EXCLUSIONS: pairs where Thursday or Friday is an early close; "
            "pairs where the Friday is not the last session of its calendar "
            "week; pairs where the following Monday is an exchange holiday "
            "(long weekend) -- these are EXCLUDED from the headline and "
            "reported as a separate subset.",
        ],
        "roots": {},
    }
    pe = tabs["atm_per_expiry"]
    sched = session_close_schedule("2010-01-01", "2026-07-01")
    cal = sorted(sched.keys())
    early = {d for d, c in sched.items() if c.strftime("%H:%M") < "16:00"}
    cal_set = set(cal)

    wk_unit = ((D040A["weekend_var"] + D040A["mean_intraday_session_rv"])
               / D040A["close_to_close_session_var"])
    res["empirical_weekend_leg_session_units"] = float(wk_unit)
    res["d040a_inputs"] = D040A

    for root in ("SPY", "QQQ"):
        g = pe[pe["root"] == root].copy()
        g["session_date"] = pd.to_datetime(g["session_date"]).dt.date
        g["expiry"] = pd.to_datetime(g["expiry"]).dt.date
        sess = sorted(g["session_date"].unique())
        idx = {s: i for i, s in enumerate(sess)}
        by_sess = {s: sub for s, sub in g.groupby("session_date")}

        out = {}
        for label, (lo_dte, hi_dte, tgt) in {
                "primary_fri_dte_7_10": (7, 10, 8),
                "robustness_fri_dte_12_16": (12, 16, 14)}.items():
            rows = []
            exc = {"early_close": 0, "no_thursday_pair": 0, "no_expiry_in_range": 0,
                   "long_weekend_monday_holiday": 0, "not_weekend_pair": 0,
                   "non_finite_iv": 0}
            for i, fri in enumerate(sess):
                if i == 0:
                    continue
                thu = sess[i - 1]
                if pd.Timestamp(fri).weekday() != 4 or pd.Timestamp(thu).weekday() != 3:
                    exc["no_thursday_pair"] += 1
                    continue
                if (fri - thu).days != 1:
                    exc["no_thursday_pair"] += 1
                    continue
                if fri in early or thu in early:
                    exc["early_close"] += 1
                    continue
                nxt = sess[i + 1] if i + 1 < len(sess) else None
                mon = fri + pd.Timedelta(days=3).to_pytimedelta()
                long_wk = mon not in cal_set
                gf, gt = by_sess[fri], by_sess[thu]
                cand = gf[(gf["dte"] >= lo_dte) & (gf["dte"] <= hi_dte)]
                if not len(cand):
                    exc["no_expiry_in_range"] += 1
                    continue
                k = int((cand["dte"] - tgt).abs().values.argmin())
                exp = cand["expiry"].iloc[k]
                mt = gt[gt["expiry"] == exp]
                if not len(mt):
                    exc["no_expiry_in_range"] += 1
                    continue
                iv_f = float(cand["atm_iv"].iloc[k])
                iv_t = float(mt["atm_iv"].iloc[0])
                if not (np.isfinite(iv_f) and np.isfinite(iv_t) and iv_t > 0):
                    exc["non_finite_iv"] += 1
                    continue
                D_f = (exp - fri).days
                D_t = (exp - thu).days
                import bisect
                je = bisect.bisect_right(cal, exp)
                N_f = je - bisect.bisect_right(cal, fri)
                N_t = je - bisect.bisect_right(cal, thu)
                if D_f <= 0 or D_t <= 0 or N_f <= 0 or N_t <= 0:
                    exc["no_expiry_in_range"] += 1
                    continue
                # variance-weighted session counts (b): count weekend-spanning legs
                legs_f = [(cal[a], cal[a + 1]) for a in
                          range(bisect.bisect_right(cal, fri) - 1,
                                bisect.bisect_right(cal, exp) - 1)]
                legs_t = [(cal[a], cal[a + 1]) for a in
                          range(bisect.bisect_right(cal, thu) - 1,
                                bisect.bisect_right(cal, exp) - 1)]
                V_f = sum(wk_unit if (y - x).days >= 3 else 1.0 for x, y in legs_f)
                V_t = sum(wk_unit if (y - x).days >= 3 else 1.0 for x, y in legs_t)
                den_a = iv_t * (np.sqrt((N_f / D_f) / (N_t / D_t)) - 1.0)
                den_b = (iv_t * (np.sqrt((V_f / D_f) / (V_t / D_t)) - 1.0)
                         if V_f > 0 and V_t > 0 else np.nan)
                rows.append({
                    "thu": thu, "fri": fri, "expiry": exp,
                    "dte_fri": int(cand["dte"].iloc[k]), "iv_thu": iv_t,
                    "iv_fri": iv_f, "diff": iv_f - iv_t,
                    "rel_diff": (iv_f - iv_t) / iv_t,
                    "den_theoretical_a": den_a, "den_empirical_b": den_b,
                    "N_thu": N_t, "N_fri": N_f, "D_thu": D_t, "D_fri": D_f,
                    "long_weekend": long_wk,
                })
            df = pd.DataFrame(rows)
            if not len(df):
                out[label] = {"n_pairs": 0, "exclusions": exc}
                continue
            core = df[~df["long_weekend"]]
            exc["long_weekend_monday_holiday"] = int(df["long_weekend"].sum())

            def block(d):
                if not len(d):
                    return {"n": 0}
                t_st = sps.ttest_1samp(d["diff"], 0.0)
                num = float(d["diff"].mean())
                da = float(d["den_theoretical_a"].mean())
                db = float(d["den_empirical_b"].mean())
                return {
                    "n_pairs": int(len(d)),
                    "window": _window(list(d["fri"])),
                    "mean_iv_thu": float(d["iv_thu"].mean()),
                    "mean_iv_fri": float(d["iv_fri"].mean()),
                    "mean_diff_fri_minus_thu": num,
                    "median_diff_fri_minus_thu": float(d["diff"].median()),
                    "mean_rel_diff": float(d["rel_diff"].mean()),
                    "t_stat": float(t_st.statistic), "p_value": float(t_st.pvalue),
                    "frac_pairs_fri_below_thu": float((d["diff"] < 0).mean()),
                    "denominator_a_theoretical_mean": da,
                    "denominator_b_empirical_mean": db,
                    "ratio_a_measured_over_theoretical": (num / da) if da != 0 else float("nan"),
                    "ratio_b_measured_over_empirical": (db and num / db) or float("nan"),
                }
            out[label] = {
                "core_excl_long_weekends": block(core),
                "long_weekend_subset": block(df[df["long_weekend"]]),
                "all_pairs": block(df),
                "exclusions": exc,
            }
            df.to_csv(OUT / f"d040b_pairs_{root}_{label}.csv", index=False)
        res["roots"][root] = out

    # gate read on SPY primary, core subset, denominator (a)
    sp = res["roots"]["SPY"]["primary_fri_dte_7_10"]["core_excl_long_weekends"]
    ratio = sp.get("ratio_a_measured_over_theoretical", float("nan"))
    res["gate_read"] = {
        "root": "SPY", "subset": "primary_fri_dte_7_10 / core_excl_long_weekends",
        "measured_mean_diff": sp.get("mean_diff_fri_minus_thu"),
        "theoretical_full_discount": sp.get("denominator_a_theoretical_mean"),
        "ratio": ratio,
        "gate": "measured Friday discount < 50% of calendar-day theta differential",
        "result": ("PASS -> proceed" if np.isfinite(ratio) and ratio < 0.50
                   else "FAIL -> DROP OPT-040"),
    }
    return res


# ===========================================================================
# D-048 -- IV percentile in compressed-RV states
# ===========================================================================

def d048(tabs) -> dict:
    res = {
        "registered_measurement": ("IV percentile in compressed-RV states (is IV "
                                   "already at its own floor?)"),
        "registered_gate": ("If IV sits at its floor, sellers are not "
                            "extrapolating -> DROP 048 pre-trial"),
        "gates_candidate": ("OPT-048 (long ATM straddle when 10d YZ RV < 20th "
                            "trailing-2y percentile AND 20-day price range < "
                            "15th percentile)"),
        "registered_operationalizations": [
            "COMPRESSED = trailing-2y (504 sessions, STRICTLY BEFORE t) "
            "percentile of yz_10d < 0.20 AND trailing-2y percentile of the "
            "20-session normalized price range ((max high - min low)/close) "
            "< 0.15. Both thresholds are OPT-048's own fixed parameters.",
            "REGISTERED OPERATIONALIZATION OF THE QUALITATIVE GATE, fixed "
            "BEFORE measuring: IV IS 'AT ITS FLOOR' if the MEAN iv_pctile_2y "
            "of the 30-DTE ATM IV across compressed sessions is < 0.20 "
            "(i.e. the same 20th-percentile bar OPT-048 applies to RV). "
            "'DROP 048 pre-trial' follows if and only if that holds.",
            "SURVIVAL CHECK: the thesis survives if the mean IV percentile in "
            "compressed states is MATERIALLY HIGHER than the mean RV percentile "
            "in the same states -- i.e. sellers are NOT fully extrapolating.",
            "Episodes use find_episodes(merge_gap_sessions=0) (raw runs).",
        ],
        "roots": {},
    }
    rv = tabs["rv_daily"]
    rank = tabs["iv_rank_daily"]
    for root in ("SPY", "QQQ"):
        r = rv[rv["root"] == root].sort_values("session_date").reset_index(drop=True)
        p_yz = ivs.trailing_percentile(r["yz_10d"], 504).to_numpy()
        p_rg = ivs.trailing_percentile(r["range_20d_norm"], 504).to_numpy()
        ok = np.isfinite(p_yz) & np.isfinite(p_rg)
        comp = ok & (p_yz < 0.20) & (p_rg < 0.15)
        r = r.assign(pctile_yz10=p_yz, pctile_range20=p_rg, compressed=comp)

        buckets = {}
        for b in (30, 45):
            k = rank[(rank["root"] == root) & (rank["dte_bucket"] == b)]
            m = r.merge(k[["session_date", "iv_pctile_2y", "iv_pctile_1y", "atm_iv"]],
                        on="session_date", how="left")
            cm = m["compressed"] & np.isfinite(m["iv_pctile_2y"])
            v = m.loc[cm, "iv_pctile_2y"]
            allv = m.loc[np.isfinite(m["iv_pctile_2y"]), "iv_pctile_2y"]
            buckets[f"dte_{b}"] = {
                "n_compressed_sessions_with_iv_pctile": int(cm.sum()),
                "n_compressed_sessions_missing_iv_pctile": int((m["compressed"] & ~np.isfinite(m["iv_pctile_2y"])).sum()),
                "iv_pctile_2y_in_compressed": ivs.describe(v),
                "iv_pctile_2y_deciles": [float(v.quantile(q / 10.0)) for q in range(1, 10)] if len(v) else [],
                "frac_iv_pctile_below_0.10": float((v < 0.10).mean()) if len(v) else float("nan"),
                "frac_iv_pctile_below_0.20": float((v < 0.20).mean()) if len(v) else float("nan"),
                "iv_pctile_2y_unconditional": ivs.describe(allv),
                "mean_rv_pctile_yz10_in_compressed": float(m.loc[cm, "pctile_yz10"].mean()) if cm.sum() else float("nan"),
                "mean_range_pctile_in_compressed": float(m.loc[cm, "pctile_range20"].mean()) if cm.sum() else float("nan"),
            }
            if b == 30 and cm.sum():
                mean_iv_p = float(v.mean())
                mean_rv_p = float(m.loc[cm, "pctile_yz10"].mean())
                buckets["GATE_READ_dte30"] = {
                    "mean_iv_pctile_2y": mean_iv_p,
                    "registered_threshold": 0.20,
                    "iv_at_its_floor": bool(mean_iv_p < 0.20),
                    "result": ("DROP 048 pre-trial" if mean_iv_p < 0.20
                               else "KEEP -- IV is NOT at its floor"),
                    "mean_rv_pctile_yz10": mean_rv_p,
                    "iv_minus_rv_pctile": mean_iv_p - mean_rv_p,
                    "survival_check": ("thesis SURVIVES: IV pctile materially "
                                       "above RV pctile" if mean_iv_p - mean_rv_p > 0.10
                                       else "sellers ARE extrapolating: IV pctile "
                                            "not materially above RV pctile"),
                }
        eps = ivs.find_episodes(r["session_date"].tolist(),
                               r["compressed"].to_numpy(), 0, valid=ok)
        res["roots"][root] = {
            "window_used": _window(r.loc[ok, "session_date"]),
            "n_sessions_measurable": int(ok.sum()),
            "n_quarantined_warmup": int((~ok).sum()),
            "n_compressed_sessions": int(comp.sum()),
            "frac_sessions_compressed": float(comp[ok].mean()) if ok.sum() else float("nan"),
            "n_compressed_episodes": len(eps),
            "mean_compressed_episode_duration": float(np.mean([e["duration"] for e in eps])) if eps else float("nan"),
            "buckets": buckets,
        }
    return res


# ===========================================================================
# D-049 -- HAR-RV forecast > IV_30 episode count
# ===========================================================================

def d049(tabs) -> dict:
    res = {
        "registered_measurement": ("Non-overlapping episodes where HAR-RV "
                                   "forecast > IV_30 (owned IV, V5-gated), "
                                   "aligned window"),
        "registered_gate": ">= 15 episodes -> Wave 2; fewer -> forward paper route",
        "gates_candidate": "OPT-049",
        "registered_operationalizations": [
            "HAR is the FROZEN (1,5,22) spec from src/backtesting/vol/har_rv.py, "
            "expanding causal OLS, 252-session warmup, reused unmodified. No "
            "order selection, no refit-schedule tuning.",
            "Daily RV fed to HAR = rv_1m_daily_var: sum of squared 1-minute log "
            "CLOSE-to-CLOSE returns INSIDE the regular session (structurally "
            "excludes the overnight gap).",
            "CONVERSION: annualized forecast vol = sqrt(har_forecast_var * 252).",
            "HORIZON MISMATCH (registered caveat): HAR forecasts NEXT-DAY RV "
            "while IV_30 prices 30 calendar days. Handled by reporting BOTH "
            "(i) the raw next-day HAR vs IV_30 and (ii) a horizon-matched proxy "
            "= the trailing 22-session mean of the daily HAR forecast (strictly "
            "backward looking, includes t). The proxy does NOT iterate the HAR "
            "forward -- that would change the frozen spec. BOTH counts are "
            "reported; neither was selected after seeing which is nicer.",
            "EPISODE definition, registered before counting (same shape as "
            "D-033): starts on the first crossing, ends on the last session "
            "before the first un-crossing, consecutive episodes separated by "
            "FEWER THAN 30 sessions are MERGED (the candidate holds 30 DTE). "
            "Sensitivities: raw no-merge and 60-session merge.",
            "ALIGNED WINDOW binding constraints: 1m bars start 2016-01-04; the "
            "SPY ETF greeks boundary starts 2017-01; the HAR 252-session warmup "
            "consumes the first 252 RV sessions; the SPY chain store is "
            "materially incomplete through 2017-10.",
        ],
    }
    rv = tabs["rv_daily"]
    atm = tabs["atm_iv_daily"]
    from src.data.derivations.regime_state import load_regime_state_daily
    reg = load_regime_state_daily()
    vix = pd.DataFrame({"session_date": [d.date() for d in reg.index],
                        "vix": reg["vix"].values})

    r = rv[rv["root"] == "SPY"].sort_values("session_date").reset_index(drop=True)
    r["har_vol_ann"] = np.sqrt(r["har_forecast_var"] * 252.0)
    r["har_vol_ann_22"] = r["har_vol_ann"].rolling(22, min_periods=22).mean()
    iv30 = atm[(atm["root"] == "SPY") & (atm["dte_bucket"] == 30)][["session_date", "atm_iv"]]
    m = r.merge(iv30, on="session_date", how="inner").merge(vix, on="session_date", how="left")
    m = m.sort_values("session_date").reset_index(drop=True)

    out = {"n_joined_sessions": int(len(m))}
    for tag, col in (("next_day_har", "har_vol_ann"),
                     ("horizon_matched_trailing22_har", "har_vol_ann_22")):
        gap = m[col] - m["atm_iv"]
        ok = np.isfinite(gap).to_numpy()
        flag = (gap > 0).to_numpy()
        counts = {}
        for label, g in (("raw_no_merge", 0), ("merge_30", 30), ("merge_60", 60)):
            eps = ivs.find_episodes(m["session_date"].tolist(), flag, g, valid=ok)
            counts[label] = {
                "n_episodes": len(eps),
                "episodes": [{"start": str(e["start"]), "end": str(e["end"]),
                              "duration_sessions": e["duration"],
                              "max_gap_vol_pts": float(np.nanmax(gap.values[e["start_idx"]:e["end_idx"] + 1]))}
                             for e in eps],
            }
        n = counts["merge_30"]["n_episodes"]
        vv = m["vix"].to_numpy(dtype=float)
        cm = ok & np.isfinite(vv)
        corr = float(np.corrcoef(gap.to_numpy()[cm], vv[cm])[0, 1]) if cm.sum() > 2 else float("nan")
        out[tag] = {
            "window_used": _window(m.loc[ok, "session_date"]),
            "n_measurable_sessions": int(ok.sum()),
            "n_quarantined": int((~ok).sum()),
            "frac_sessions_forecast_above_iv30": float(flag[ok].mean()),
            "gap_distribution_vol_points": ivs.describe(gap[ok]),
            "episode_counts": counts,
            "PRIMARY_n_episodes_merge30": n,
            "gate_result": "PASS" if n >= 15 else "FAIL",
            "gate_routing": "Wave 2 eligible" if n >= 15 else "forward paper route",
            "corr_gap_vs_VIX_level": corr,
        }
    return {**res, **out}


# ===========================================================================
# D-050b -- entry-day IV percentile at classifier transitions
# ===========================================================================

def _transitions(reg: pd.DataFrame):
    r = reg.copy()
    r["prev"] = r["regime"].shift(1)
    r = r.dropna(subset=["prev"])
    ch = r[r["regime"] != r["prev"]]
    return ch


def d050b(tabs) -> dict:
    res = {
        "registered_measurement": "Entry-day IV percentile at classifier transitions",
        "registered_gate": "If mean pctile > 80, classifier lags the vol -> DROP 050",
        "gates_candidate": ("OPT-050 (long 21-30 DTE ATM straddle for 5 sessions "
                            "after a transition INTO UNPREDICTABLE or OUT OF "
                            "STRONG_BULL)"),
        "registered_operationalizations": [
            "TRIGGER SET = transitions INTO UNPREDICTABLE plus transitions OUT "
            "OF STRONG_BULL, on the causal-replay regime_state_daily classifier.",
            "ENTRY SESSIONS: two readings are reported and both were registered "
            "before measuring -- (i) the transition session t0 ONLY, and (ii) "
            "the 5-session window t0..t0+5 (6 sessions).",
            "IV percentile = iv_pctile_2y and iv_pctile_1y of the 30-DTE "
            "constant-maturity ATM IV, both strictly backward looking.",
            "GATE is applied to the MEAN iv_pctile_2y expressed on the same 0-100 "
            "scale as the registered threshold of 80 (i.e. mean fraction > 0.80).",
            "REGIME CAVEAT (registered upstream): regime_state_daily is a CAUSAL "
            "REPLAY computed on TODAY'S SPY/VIX vintage; it is not a "
            "point-in-time classifier log.",
        ],
    }
    from src.data.derivations.regime_state import load_regime_state_daily
    reg = load_regime_state_daily()
    reg = reg.reset_index().rename(columns={"date": "dt"})
    reg["session_date"] = pd.to_datetime(reg["dt"]).dt.date
    ch = _transitions(reg)
    trig = ch[(ch["regime"] == "UNPREDICTABLE") | (ch["prev"] == "STRONG_BULL")]

    rank = tabs["iv_rank_daily"]
    out = {}
    for root in ("SPY", "QQQ"):
        k = rank[(rank["root"] == root) & (rank["dte_bucket"] == 30)]
        k = k.sort_values("session_date").reset_index(drop=True)
        sess = k["session_date"].tolist()
        pos = {s: i for i, s in enumerate(sess)}

        def gather(dates, span):
            idxs = set()
            lost = 0
            for d in dates:
                i = pos.get(d)
                if i is None:
                    # snap forward to the next available chain session
                    later = [j for j, s in enumerate(sess) if s >= d]
                    if not later:
                        lost += 1
                        continue
                    i = later[0]
                    if (sess[i] - d).days > 5:
                        lost += 1
                        continue
                for j in range(i, min(i + span, len(sess))):
                    idxs.add(j)
            return sorted(idxs), lost

        blocks = {}
        for tag, dates in (("OPT050_trigger_set", trig["session_date"].tolist()),
                           ("ALL_transitions", ch["session_date"].tolist())):
            for span, sl in ((1, "transition_session_only"), (6, "window_t0_to_t5")):
                ii, lost = gather(dates, span)
                v2 = k.loc[ii, "iv_pctile_2y"].dropna()
                v1 = k.loc[ii, "iv_pctile_1y"].dropna()
                blocks[f"{tag}.{sl}"] = {
                    "n_trigger_events_total": len(dates),
                    "n_events_lost_to_chain_coverage": lost,
                    "n_entry_sessions": len(ii),
                    "n_with_iv_pctile_2y": int(len(v2)),
                    "mean_iv_pctile_2y": float(v2.mean()) if len(v2) else float("nan"),
                    "median_iv_pctile_2y": float(v2.median()) if len(v2) else float("nan"),
                    "mean_iv_pctile_1y": float(v1.mean()) if len(v1) else float("nan"),
                    "median_iv_pctile_1y": float(v1.median()) if len(v1) else float("nan"),
                }
        allv = k["iv_pctile_2y"].dropna()
        blocks["UNCONDITIONAL_baseline"] = {
            "n": int(len(allv)), "mean_iv_pctile_2y": float(allv.mean()),
            "median_iv_pctile_2y": float(allv.median()),
        }
        gk = blocks["OPT050_trigger_set.window_t0_to_t5"]["mean_iv_pctile_2y"]
        blocks["GATE_READ"] = {
            "basis": "OPT050_trigger_set / window_t0_to_t5 / mean_iv_pctile_2y",
            "measured_mean_pctile_0_100": gk * 100.0 if np.isfinite(gk) else float("nan"),
            "registered_threshold": 80.0,
            "result": ("DROP 050 -- classifier lags the vol"
                       if np.isfinite(gk) and gk * 100.0 > 80.0
                       else "KEEP -- classifier does NOT land at an IV extreme"),
        }
        blocks["window_used"] = _window(k["session_date"].dropna().tolist())
        out[root] = blocks

    res["roots"] = out
    res["boundary_note"] = (
        "SPY chain greeks start 2017-01 while the classifier log starts "
        "2012-06, so SPY loses every trigger event before the chain window. "
        "QQQ carries full 2012+ greeks and is reported for the pre-2017 view -- "
        "it is a DIFFERENT UNDERLYING and is NOT a substitute for the SPY read.")
    res["n_trigger_events_by_year"] = (
        pd.to_datetime(trig["session_date"]).dt.year.value_counts().sort_index().to_dict())
    return res


# ===========================================================================
# D-011 -- BLOCKED, plus a scope-reduced supplementary measurement
# ===========================================================================

def d011(tabs) -> dict:
    res = {
        "registered_measurement": ("Entry-day put-IV percentile census at "
                                   "breakdown triggers"),
        "registered_gate": ("Descriptive; if entries systematically land at IV "
                            "pctile > 80, expect the drift edge to be consumed"),
        "gates_candidate": "OPT-011",
        "verdict": "BLOCKED",
        "blocked_reason": (
            "OPT-011's registered universe is 'bottom-decile momentum S&P 500 "
            "names, max 10'. Those single-name option chains are NOT on disk: "
            "the store holds 31 roots, the universe top-up route is dead "
            "(ThetaData subscription cancelled, logged 2026-07-27) and the ORATS "
            "extension is deferred. The registered D-011 census therefore CANNOT "
            "be run as specified. No proxy is substituted for the gate."),
    }
    from src.data.derivations.regime_state import load_regime_state_daily
    reg = load_regime_state_daily().reset_index()
    reg["session_date"] = pd.to_datetime(reg["date"]).dt.date
    reg = reg[["session_date", "regime"]]

    pe = tabs["atm_per_expiry"]
    put = tabs["put_iv_daily"]
    supp = {
        "LABEL": ("SCOPE-REDUCED SUPPLEMENTARY MEASUREMENT -- index-level "
                  "analogue on SPY/QQQ. This is NOT the registered D-011 census "
                  "and it is NOT the gate."),
        "registered_operationalizations": [
            "TRIGGER = regime == BEAR AND the 15:45 snapshot underlying_px is a "
            "20-session LOW (<= the min of the trailing 20 sessions INCLUDING t).",
            "Put IV = the put whose |delta| is nearest 0.50 and nearest 0.30 "
            "(tolerance 0.07) on the representative listed expiry for the 45 and "
            "60 DTE buckets.",
            "Percentile = trailing-2y (504 sessions, STRICTLY BEFORE t) "
            "percentile of that put-IV series.",
        ],
        "roots": {},
    }
    for root in ("SPY", "QQQ"):
        ul = (pe[pe["root"] == root]
              .groupby("session_date", as_index=False)["underlying_px"].first()
              .sort_values("session_date").reset_index(drop=True))
        ul["low20"] = ul["underlying_px"].rolling(20, min_periods=20).min()
        ul["is_20d_low"] = ul["underlying_px"] <= ul["low20"]
        ul = ul.merge(reg, on="session_date", how="left")
        trig = ul["is_20d_low"] & (ul["regime"] == "BEAR")
        b = {}
        for bucket in (45, 60):
            p = put[(put["root"] == root) & (put["dte_bucket"] == bucket)]
            p = p.sort_values("session_date").reset_index(drop=True)
            for col, name in (("iv_50d", "put_50delta"), ("iv_30d", "put_30delta")):
                if col not in p.columns:
                    continue
                pc = ivs.trailing_percentile(p[col], 504)
                pp = pd.DataFrame({"session_date": p["session_date"],
                                   "pctile": pc.values})
                j = ul.merge(pp, on="session_date", how="left")
                tv = j.loc[trig.values & np.isfinite(j["pctile"]), "pctile"]
                av = j.loc[np.isfinite(j["pctile"]), "pctile"]
                b[f"dte{bucket}_{name}"] = {
                    "n_trigger_sessions_with_pctile": int(len(tv)),
                    "mean_iv_pctile_2y_on_triggers": float(tv.mean()) if len(tv) else float("nan"),
                    "median_iv_pctile_2y_on_triggers": float(tv.median()) if len(tv) else float("nan"),
                    "frac_triggers_above_0.80": float((tv > 0.80).mean()) if len(tv) else float("nan"),
                    "unconditional_mean_iv_pctile_2y": float(av.mean()) if len(av) else float("nan"),
                    "n_unconditional": int(len(av)),
                }
        supp["roots"][root] = {
            "window_used": _window(ul["session_date"].tolist()),
            "n_trigger_sessions": int(trig.sum()),
            "buckets": b,
        }
    res["supplementary_index_level_analogue"] = supp
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with RunStatus("wave0_b1_diagnostics", meta={"scope": "groupB1"}) as status:
        tabs = _load()
        cov = {r: {"n_partitions": len(ivs.available_partitions(r))}
               for r in ("SPY", "QQQ")}
        results = {"coverage": cov,
                   "scope": ("DATA-PROPERTY MEASUREMENTS ONLY -- no backtest, no "
                             "P&L, no positions, no fills, no equity curve.")}
        for name, fn in (("D-018", d018), ("D-027", d027), ("D-033", d033),
                         ("D-040b", d040b), ("D-048", d048), ("D-049", d049),
                         ("D-050b", d050b), ("D-011", d011)):
            status.heartbeat(note=name)
            logger.info(f"[*] {name}")
            results[name] = fn(tabs)
            (OUT / f"{name}.json").write_text(
                json.dumps(_json_safe(results[name]), indent=2), encoding="utf-8")
        (OUT / "wave0_groupB1_all.json").write_text(
            json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    logger.info("[+] group B1 diagnostics written to output/wave0/groupB1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
