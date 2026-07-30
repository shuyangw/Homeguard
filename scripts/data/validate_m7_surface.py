"""M7 validation battery -- the four registered checks plus the D-047/030 gate.

Registered in `docs/strategies/research/options-slate/20260730_m7_prereg.md`
BEFORE any of these numbers were computed:

  1. fit residuals by moneyness bucket (distributions, not just means)
  2. static no-arbitrage: butterfly and calendar violation rates by bucket
  3. day-over-day stability of the surface
  4. D-047/030 integrity gate at 0.05 delta, with the threshold fixed in advance

Nothing here repairs anything. Violations are measured and reported.

Writes CSVs to `output/m7_validation/` and prints a summary.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import polars as pl

from src.data.options.canonical import EOD_STORE_NAME
from src.data.options.iv_surface import (
    REASON_OK,
    SVIParams,
    black76_iv_vec,
    gatheral_g_vec,
    smooth_iv,
    svi_total_variance,
)
from src.data.options.iv_surface_build import (
    SMOOTH_STORE_NAME,
    SURFACE_STORE_NAME,
    _available_months,
    _month_path,
)
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 20260730
OUT_DIR = Path("output") / "m7_validation"

#: Matches the V-battery / spread-census |delta| bucket edges.
DELTA_EDGES = [0.0, 0.05, 0.15, 0.35, 0.65, 1.01]
DELTA_LABELS = ["0.00-0.05", "0.05-0.15", "0.15-0.35", "0.35-0.65", "0.65-1.00"]

#: Registered D-047/030 gate (pre-reg Section 6).
GATE_DTE = (21, 45)
GATE_TARGET_DELTA = 0.05
GATE_SESSIONS_PER_ROOT = 200
GATE_MIN_INSIDE_FRAC = 0.90
GATE_MAX_MEDIAN_ABS_DIFF = 0.015


def _read_store(root: str, store: str, months, sessions=None,
                columns=None) -> pl.DataFrame:
    """Read a store, filtering per-partition so memory stays bounded.

    The EOD chain is ~424 MB compressed across both roots; reading it whole and
    filtering afterwards is what makes this battery fall over.
    """
    keep = set(sessions) if sessions is not None else None
    frames = []
    for y, m in months:
        p = _month_path(root, y, m, store)
        if not p.exists():
            continue
        df = pl.read_parquet(p, columns=columns)
        if keep is not None:
            df = df.filter(pl.col("session_date").is_in(list(keep)))
            if df.height == 0:
                continue
        for c, d in df.schema.items():
            if d == pl.Float32:
                df = df.with_columns(pl.col(c).cast(pl.Float64))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def _bucket(abs_delta: np.ndarray) -> np.ndarray:
    idx = np.digitize(abs_delta, DELTA_EDGES[1:-1], right=False)
    return np.array(DELTA_LABELS, dtype=object)[np.clip(idx, 0, len(DELTA_LABELS) - 1)]


def _describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size), "mean": float(np.mean(x)), "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)), "p50": float(np.median(x)),
        "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)), "max": float(np.max(x)),
    }


# ---------------------------------------------------------------------------
# Refusal census
# ---------------------------------------------------------------------------


def refusal_census(params: pl.DataFrame) -> pl.DataFrame:
    return (
        params.with_columns(pl.col("session_date").dt.year().alias("year"))
        .group_by(["root", "year", "reason"])
        .len()
        .sort(["root", "year", "reason"])
    )


def refusal_by_dte(params: pl.DataFrame) -> pl.DataFrame:
    return (
        params.with_columns(
            pl.when(pl.col("dte") == 0).then(pl.lit("0"))
            .when(pl.col("dte") <= 7).then(pl.lit("1-7"))
            .when(pl.col("dte") <= 30).then(pl.lit("8-30"))
            .when(pl.col("dte") <= 60).then(pl.lit("31-60"))
            .when(pl.col("dte") <= 90).then(pl.lit("61-90"))
            .when(pl.col("dte") <= 180).then(pl.lit("91-180"))
            .otherwise(pl.lit("181+")).alias("dte_bucket")
        )
        .group_by(["root", "dte_bucket", "reason"]).len()
        .sort(["root", "dte_bucket", "reason"])
    )


# ---------------------------------------------------------------------------
# 1. Fit residuals by moneyness bucket
# ---------------------------------------------------------------------------


def residual_census(root: str, months, n_sessions: int) -> pl.DataFrame:
    """Per-point residual (smoothed IV minus quote-implied IV), by |delta| bucket.

    Also records whether the surface lands INSIDE the point's own bid-ask IV
    band -- the economically meaningful question.
    """
    params = _read_store(root, SURFACE_STORE_NAME, months).filter(
        pl.col("reason") == REASON_OK
    )
    if params.height == 0:
        return pl.DataFrame()
    rng = np.random.default_rng(SEED)
    sessions = params["session_date"].unique().sort().to_list()
    pick = set(rng.choice(len(sessions), min(n_sessions, len(sessions)),
                          replace=False).tolist())
    keep = {sessions[i] for i in pick}
    params = params.filter(pl.col("session_date").is_in(list(keep)))

    chain = _read_store(root, EOD_STORE_NAME, months, sessions=keep)
    smooth = _read_store(root, SMOOTH_STORE_NAME, months, sessions=keep)
    joined = chain.join(
        smooth.select("session_date", "expiry", "strike", "right",
                      "iv_smooth", "delta_smooth", "extrapolated"),
        on=["session_date", "expiry", "strike", "right"], how="inner",
    ).join(
        params.select("session_date", "expiry", "forward", "discount", "T"),
        on=["session_date", "expiry"], how="inner",
    ).filter(pl.col("iv_smooth").is_not_null())

    rows = []
    for (sd, expiry), g in joined.partition_by(
        ["session_date", "expiry"], as_dict=True
    ).items():
        F = float(g["forward"][0]); D = float(g["discount"][0]); T = float(g["T"][0])
        K = g["strike"].to_numpy()
        is_call = (g["right"].to_numpy() == "C")
        # OTM side only -- the fit's own domain
        otm = np.where(K >= F, is_call, ~is_call)
        if not otm.any():
            continue
        K = K[otm]; is_call = is_call[otm]
        mid = g["mid"].to_numpy()[otm]
        bid = g["bid"].to_numpy()[otm]
        ask = g["ask"].to_numpy()[otm]
        ivs = g["iv_smooth"].to_numpy()[otm]
        dl = np.abs(g["delta_smooth"].to_numpy()[otm])
        ex = g["extrapolated"].to_numpy()[otm]
        iv_mid = black76_iv_vec(mid, F, K, T, D, is_call)
        iv_bid = black76_iv_vec(bid, F, K, T, D, is_call)
        iv_ask = black76_iv_vec(ask, F, K, T, D, is_call)
        ok = np.isfinite(iv_mid) & np.isfinite(ivs)
        if not ok.any():
            continue
        inside = (ivs >= np.minimum(iv_bid, iv_ask)) & (ivs <= np.maximum(iv_bid, iv_ask))
        for i in np.flatnonzero(ok):
            rows.append({
                "root": root, "session_date": sd, "dte": int(g["dte"][0]),
                "abs_delta": float(dl[i]), "resid": float(ivs[i] - iv_mid[i]),
                "iv_band": float(iv_ask[i] - iv_bid[i])
                if np.isfinite(iv_ask[i]) and np.isfinite(iv_bid[i]) else np.nan,
                "inside_band": bool(inside[i]) if np.isfinite(inside[i]) else None,
                "extrapolated": bool(ex[i]) if ex[i] is not None else None,
            })
    return pl.DataFrame(rows)


def summarize_residuals(res: pl.DataFrame) -> pl.DataFrame:
    if res.height == 0:
        return pl.DataFrame()
    res = res.with_columns(
        pl.Series("bucket", _bucket(res["abs_delta"].to_numpy()), dtype=pl.String)
    )
    out = []
    for (root, bucket), g in res.partition_by(["root", "bucket"], as_dict=True).items():
        d = _describe(np.abs(g["resid"].to_numpy()))
        d.update(root=root, bucket=bucket,
                 signed_median=float(np.median(g["resid"].to_numpy())),
                 inside_band_frac=float(np.nanmean(
                     g["inside_band"].cast(pl.Float64).to_numpy())),
                 median_iv_band=float(np.nanmedian(g["iv_band"].to_numpy())),
                 extrap_frac=float(np.nanmean(
                     g["extrapolated"].cast(pl.Float64).to_numpy())))
        out.append(d)
    return pl.DataFrame(out).sort(["root", "bucket"])


# ---------------------------------------------------------------------------
# 2. Static no-arbitrage
# ---------------------------------------------------------------------------


def butterfly_census(root: str, months) -> pl.DataFrame:
    """Gatheral g(k) evaluated at the ACTUAL contract strikes, bucketed by delta."""
    params = _read_store(root, SURFACE_STORE_NAME, months).filter(
        pl.col("reason") == REASON_OK
    )
    smooth = _read_store(root, SMOOTH_STORE_NAME, months).filter(
        pl.col("iv_smooth").is_not_null()
    )
    if params.height == 0 or smooth.height == 0:
        return pl.DataFrame()
    j = smooth.join(
        params.select("session_date", "expiry", "T", "svi_a", "svi_b",
                      "svi_rho", "svi_m", "svi_sigma"),
        on=["session_date", "expiry"], how="inner",
    ).filter(pl.col("k").is_not_null() & pl.col("delta_smooth").is_not_null())
    if j.height == 0:
        return pl.DataFrame()
    # One vectorized pass over every contract -- no per-slice grouping.
    g = gatheral_g_vec(
        j["k"].to_numpy(), j["svi_a"].to_numpy(), j["svi_b"].to_numpy(),
        j["svi_rho"].to_numpy(), j["svi_m"].to_numpy(), j["svi_sigma"].to_numpy(),
    )
    dl = np.abs(j["delta_smooth"].to_numpy())
    df = pl.DataFrame({"g": g, "bucket": _bucket(dl)})
    return (
        df.group_by("bucket")
        .agg(pl.len().alias("n"),
             (pl.col("g") < 0).sum().alias("n_violations"),
             pl.col("g").min().alias("min_g"))
        .with_columns((pl.col("n_violations") / pl.col("n")).alias("viol_rate"),
                      pl.lit(root).alias("root"))
        .sort("bucket")
    )


def calendar_census(root: str, months) -> pl.DataFrame:
    """Total variance must not fall with maturity at fixed k."""
    params = _read_store(root, SURFACE_STORE_NAME, months).filter(
        pl.col("reason") == REASON_OK
    )
    if params.height == 0:
        return pl.DataFrame()
    grid = np.linspace(-0.3, 0.2, 51)
    recs = []
    for (sd,), g in params.partition_by(["session_date"], as_dict=True).items():
        g = g.sort("T")
        if g.height < 2:
            continue
        rows = list(g.iter_rows(named=True))
        ws = [
            svi_total_variance(grid, SVIParams(r["svi_a"], r["svi_b"], r["svi_rho"],
                                               r["svi_m"], r["svi_sigma"]))
            for r in rows
        ]
        for i in range(len(ws) - 1):
            bad = ws[i + 1] < ws[i] - 1e-12
            # Worst violation expressed in vol points at the far expiry, so a
            # numerically-trivial crossing is distinguishable from a real one.
            gap = np.maximum(ws[i] - ws[i + 1], 0.0)
            T_far = rows[i + 1]["T"]
            vol_pts = float(np.max(np.sqrt(np.maximum(ws[i], 1e-12) / T_far)
                                   - np.sqrt(np.maximum(ws[i + 1], 1e-12) / T_far))
                            ) if bad.any() else 0.0
            recs.append({
                "near_dte": rows[i]["dte"], "far_dte": rows[i + 1]["dte"],
                "dte_gap": rows[i + 1]["dte"] - rows[i]["dte"],
                "any_violation": bool(bad.any()),
                "points_violating": int(bad.sum()), "points": int(bad.size),
                "max_gap_w": float(gap.max()), "max_vol_points": vol_pts,
            })
    if not recs:
        return pl.DataFrame()
    df = pl.DataFrame(recs).with_columns(
        pl.when(pl.col("near_dte") <= 7).then(pl.lit("near<=7"))
        .when(pl.col("near_dte") <= 30).then(pl.lit("near 8-30"))
        .when(pl.col("near_dte") <= 90).then(pl.lit("near 31-90"))
        .otherwise(pl.lit("near 91+")).alias("bucket")
    )
    out = (
        df.group_by("bucket")
        .agg(
            pl.len().alias("adjacent_pairs"),
            pl.col("any_violation").sum().alias("pairs_with_violation"),
            pl.col("points_violating").sum().alias("points_violating"),
            pl.col("points").sum().alias("points"),
            pl.col("max_vol_points").median().alias("median_max_vol_points"),
            pl.col("max_vol_points").max().alias("worst_vol_points"),
        )
        .with_columns(
            (pl.col("pairs_with_violation") / pl.col("adjacent_pairs")).alias("pair_viol_rate"),
            (pl.col("points_violating") / pl.col("points")).alias("point_viol_rate"),
            pl.lit(root).alias("root"),
        )
        .sort("bucket")
    )
    total = pl.DataFrame([{
        "bucket": "ALL", "adjacent_pairs": df.height,
        "pairs_with_violation": int(df["any_violation"].sum()),
        "points_violating": int(df["points_violating"].sum()),
        "points": int(df["points"].sum()),
        "median_max_vol_points": float(df["max_vol_points"].median()),
        "worst_vol_points": float(df["max_vol_points"].max()),
        "pair_viol_rate": float(df["any_violation"].mean()),
        "point_viol_rate": float(df["points_violating"].sum() / df["points"].sum()),
        "root": root,
    }])
    return pl.concat([out, total], how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# 3. Day-over-day stability
# ---------------------------------------------------------------------------


def stability_census(root: str, months) -> pl.DataFrame:
    """Session-over-session change in the surface at fixed log-moneyness."""
    params = _read_store(root, SURFACE_STORE_NAME, months).filter(
        pl.col("reason") == REASON_OK
    )
    if params.height == 0:
        return pl.DataFrame()
    rows = []
    for r in params.iter_rows(named=True):
        p = SVIParams(r["svi_a"], r["svi_b"], r["svi_rho"], r["svi_m"], r["svi_sigma"])
        rows.append({
            "root": root, "session_date": r["session_date"], "expiry": r["expiry"],
            "dte": r["dte"], "spot": r["spot"],
            "iv_atm": float(smooth_iv(p, 0.0, r["T"])),
            "iv_k20": float(smooth_iv(p, -0.20, r["T"])),
        })
    df = pl.DataFrame(rows).sort(["expiry", "session_date"])
    df = df.with_columns(
        (pl.col("iv_atm") - pl.col("iv_atm").shift(1).over("expiry")).alias("d_atm"),
        (pl.col("iv_k20") - pl.col("iv_k20").shift(1).over("expiry")).alias("d_k20"),
        (pl.col("spot") / pl.col("spot").shift(1).over("expiry") - 1).alias("spot_ret"),
        (pl.col("session_date") - pl.col("session_date").shift(1).over("expiry"))
        .dt.total_days().alias("gap_days"),
    ).filter(pl.col("gap_days") <= 4)  # consecutive sessions only
    return df


def summarize_stability(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame()
    out = []
    for col in ("d_atm", "d_k20"):
        x = np.abs(df[col].to_numpy())
        d = _describe(x)
        big = x > 0.05
        d.update(metric=col, frac_jump_gt_5volpts=float(np.nanmean(big)))
        sr = np.abs(df["spot_ret"].to_numpy())
        with np.errstate(invalid="ignore"):
            d["median_abs_spot_ret_on_jumps"] = (
                float(np.nanmedian(sr[big])) if big.any() else float("nan")
            )
            d["median_abs_spot_ret_overall"] = float(np.nanmedian(sr))
        out.append(d)
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------
# 4. D-047/030 integrity gate
# ---------------------------------------------------------------------------


def d047_gate(root: str, months) -> tuple:
    params = _read_store(root, SURFACE_STORE_NAME, months).filter(
        (pl.col("reason") == REASON_OK)
        & pl.col("dte").is_between(GATE_DTE[0], GATE_DTE[1])
    )
    if params.height == 0:
        return pl.DataFrame(), {}
    rng = np.random.default_rng(SEED)
    sessions = params["session_date"].unique().sort().to_list()
    idx = rng.choice(len(sessions), min(GATE_SESSIONS_PER_ROOT, len(sessions)),
                     replace=False)
    keep = [sessions[i] for i in sorted(idx.tolist())]
    params = params.filter(pl.col("session_date").is_in(keep))

    chain = _read_store(root, EOD_STORE_NAME, months, sessions=keep).filter(
        pl.col("dte").is_between(GATE_DTE[0], GATE_DTE[1])
    )
    smooth = _read_store(root, SMOOTH_STORE_NAME, months, sessions=keep).filter(
        pl.col("iv_smooth").is_not_null()
    )
    j = chain.join(
        smooth.select("session_date", "expiry", "strike", "right", "iv_smooth",
                      "delta_smooth", "extrapolated"),
        on=["session_date", "expiry", "strike", "right"], how="inner",
    ).join(
        params.select("session_date", "expiry", "forward", "discount", "T"),
        on=["session_date", "expiry"], how="inner",
    ).with_columns(
        (pl.col("delta_smooth").abs() - GATE_TARGET_DELTA).abs().alias("dist")
    )
    # per (session, expiry, right): the contract nearest 0.05 smoothed delta
    picked = j.sort("dist").group_by(
        ["session_date", "expiry", "right"], maintain_order=False
    ).first()

    F = picked["forward"].to_numpy(); D = picked["discount"].to_numpy()
    T = picked["T"].to_numpy(); K = picked["strike"].to_numpy()
    is_call = picked["right"].to_numpy() == "C"
    ivs = picked["iv_smooth"].to_numpy()
    iv_mid = np.empty(len(K)); iv_bid = np.empty(len(K)); iv_ask = np.empty(len(K))
    for i in range(len(K)):
        one = np.array([K[i]]); c = np.array([is_call[i]])
        iv_mid[i] = black76_iv_vec(np.array([picked["mid"][i]]), F[i], one, T[i], D[i], c)[0]
        iv_bid[i] = black76_iv_vec(np.array([picked["bid"][i]]), F[i], one, T[i], D[i], c)[0]
        iv_ask[i] = black76_iv_vec(np.array([picked["ask"][i]]), F[i], one, T[i], D[i], c)[0]

    det = picked.select(
        "root", "session_date", "expiry", "strike", "right", "dte",
        "delta_smooth", "iv_smooth", "extrapolated"
    ).with_columns(
        pl.Series("iv_mid_raw", iv_mid), pl.Series("iv_bid_raw", iv_bid),
        pl.Series("iv_ask_raw", iv_ask),
        pl.Series("abs_diff", np.abs(ivs - iv_mid)),
        pl.Series("inside_band", (ivs >= np.minimum(iv_bid, iv_ask))
                  & (ivs <= np.maximum(iv_bid, iv_ask))),
    ).filter(pl.col("iv_mid_raw").is_finite())

    inside = float(det["inside_band"].mean()) if det.height else float("nan")
    med = float(det["abs_diff"].median()) if det.height else float("nan")
    verdict = {
        "root": root, "n_sampled": det.height,
        "inside_band_frac": inside, "median_abs_diff": med,
        "p90_abs_diff": float(np.nanpercentile(det["abs_diff"].to_numpy(), 90))
        if det.height else float("nan"),
        "p99_abs_diff": float(np.nanpercentile(det["abs_diff"].to_numpy(), 99))
        if det.height else float("nan"),
        "signed_median": float(np.nanmedian(
            (det["iv_smooth"] - det["iv_mid_raw"]).to_numpy())) if det.height else float("nan"),
        "extrap_frac": float(det["extrapolated"].mean()) if det.height else float("nan"),
        "gate_a_inside_ge_90pct": bool(inside >= GATE_MIN_INSIDE_FRAC),
        "gate_b_median_le_1p5volpts": bool(med <= GATE_MAX_MEDIAN_ABS_DIFF),
    }
    verdict["PASS"] = bool(
        verdict["gate_a_inside_ge_90pct"] and verdict["gate_b_median_le_1p5volpts"]
    )
    return det, verdict


# ---------------------------------------------------------------------------
# Supplementary (NOT one of the four registered checks): strike-selection
# agreement. P1 consumes the surface to SELECT a strike, not to price one, so
# the decision-relevant question is whether smoothed and shipped delta pick the
# same contract.
# ---------------------------------------------------------------------------


def selection_agreement(root: str, months) -> pl.DataFrame:
    params = _read_store(root, SURFACE_STORE_NAME, months).filter(
        (pl.col("reason") == REASON_OK)
        & pl.col("dte").is_between(GATE_DTE[0], GATE_DTE[1])
    )
    if params.height == 0:
        return pl.DataFrame()
    rng = np.random.default_rng(SEED)
    sessions = params["session_date"].unique().sort().to_list()
    idx = rng.choice(len(sessions), min(GATE_SESSIONS_PER_ROOT, len(sessions)),
                     replace=False)
    keep = [sessions[i] for i in sorted(idx.tolist())]

    chain = _read_store(root, EOD_STORE_NAME, months, sessions=keep).filter(
        pl.col("dte").is_between(GATE_DTE[0], GATE_DTE[1])
    )
    smooth = _read_store(root, SMOOTH_STORE_NAME, months, sessions=keep).filter(
        pl.col("iv_smooth").is_not_null()
    )
    j = chain.join(
        smooth.select("session_date", "expiry", "strike", "right", "delta_smooth"),
        on=["session_date", "expiry", "strike", "right"], how="inner",
    ).filter(pl.col("delta").is_not_nan() & pl.col("delta").is_not_null())
    if j.height == 0:
        return pl.DataFrame()

    rows = []
    for target in (0.05, 0.10, 0.25):
        a = j.with_columns(
            (pl.col("delta_smooth").abs() - target).abs().alias("ds"),
            (pl.col("delta").abs() - target).abs().alias("dh"),
        )
        pick_s = a.sort("ds").group_by(
            ["session_date", "expiry", "right"], maintain_order=False
        ).first().select("session_date", "expiry", "right",
                         pl.col("strike").alias("k_smooth"))
        pick_h = a.sort("dh").group_by(
            ["session_date", "expiry", "right"], maintain_order=False
        ).first().select("session_date", "expiry", "right",
                         pl.col("strike").alias("k_shipped"))
        m = pick_s.join(pick_h, on=["session_date", "expiry", "right"], how="inner")
        same = (m["k_smooth"] == m["k_shipped"]).mean()
        diff = (m["k_smooth"] - m["k_shipped"]).abs()
        rows.append({
            "root": root, "target_delta": target, "n": m.height,
            "same_strike_frac": float(same),
            "median_abs_strike_diff": float(diff.median()),
            "p95_abs_strike_diff": float(np.percentile(diff.to_numpy(), 95)),
        })
    return pl.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M7 validation battery.")
    ap.add_argument("--roots", default="SPY,QQQ")
    ap.add_argument("--resid-sessions", type=int, default=150)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]

    all_params, resid, bfly, cal, stab, gates, gate_det, sel = (
        [], [], [], [], [], [], [], []
    )
    for root in roots:
        months = _available_months(root)
        logger.info(f"[+] {root}: {len(months)} months")
        all_params.append(_read_store(root, SURFACE_STORE_NAME, months))
        resid.append(residual_census(root, months, args.resid_sessions))
        bfly.append(butterfly_census(root, months))
        cal.append(calendar_census(root, months))
        stab.append(stability_census(root, months))
        sel.append(selection_agreement(root, months))
        d, v = d047_gate(root, months)
        if v:
            gates.append(v)
            gate_det.append(d)

    params = pl.concat([p for p in all_params if p.height], how="vertical_relaxed")
    res = pl.concat([r for r in resid if r.height], how="vertical_relaxed")
    stb = pl.concat([s for s in stab if s.height], how="vertical_relaxed")

    outputs = {
        "refusal_by_root_year.csv": refusal_census(params),
        "refusal_by_dte.csv": refusal_by_dte(params),
        "residuals_by_bucket.csv": summarize_residuals(res),
        "butterfly_by_bucket.csv": pl.concat([b for b in bfly if b.height],
                                             how="vertical_relaxed"),
        "calendar.csv": pl.concat([c for c in cal if c.height],
                                  how="vertical_relaxed"),
        "stability.csv": summarize_stability(stb),
        "selection_agreement.csv": pl.concat([s for s in sel if s.height],
                                             how="vertical_relaxed")
        if any(s.height for s in sel) else pl.DataFrame(),
        "d047_gate.csv": pl.DataFrame(gates),
        "d047_gate_detail.csv": pl.concat([d for d in gate_det if d.height],
                                          how="vertical_relaxed"),
    }
    for name, df in outputs.items():
        if df is not None and df.height:
            df.write_csv(OUT_DIR / name)
            logger.info(f"[+] wrote {OUT_DIR / name} ({df.height} rows)")

    logger.info("[+] D-047/030 gate:")
    for v in gates:
        logger.info(
            f"    {v['root']}: n={v['n_sampled']} inside={v['inside_band_frac']:.4f} "
            f"median_abs_diff={v['median_abs_diff']:.5f} -> "
            f"{'PASS' if v['PASS'] else 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
