"""Materialize the M7 smoothed IV surface from `options_chain_eod`.

Writes TWO hive-partitioned tables alongside the source store, matching the
`root=/year=/month=/data.parquet` convention `canonical.py` already uses:

* `options_iv_surface` -- one row per (root, session_date, expiry): the fitted
  SVI parameters, the parity-implied forward, fit diagnostics, and the REASON
  CODE. Refused slices are present here with `reason != OK` and null params, so
  a refusal is a positive record rather than a missing row.

* `options_iv_smooth` -- one row per contract-session: `iv_smooth`,
  `delta_smooth`, and provenance (`iv_source`, `surface_reason`,
  `extrapolated`). This is what P1 joins to `options_chain_eod`.

The source table is NOT mutated. A join table keeps M7 re-runnable and keeps
the canonical chain immutable.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from src.data.options.canonical import EOD_STORE_NAME
from src.data.options.iv_surface import (
    REASON_OK,
    ExpiryFit,
    butterfly_violation_rate,
    fit_expiry,
    implied_forward,
    smooth_delta,
    smooth_iv,
    zero_rate,
)
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

SURFACE_STORE_NAME = "options_iv_surface"
SMOOTH_STORE_NAME = "options_iv_smooth"

#: Bumped whenever the fit changes meaningfully, so stale rows are identifiable.
FIT_VERSION = "m7_svi_v1"

#: Registered: calendar-day year fraction.
DAYS_PER_YEAR = 365.0

_PARAM_SCHEMA = {
    "root": pl.String, "session_date": pl.Date, "expiry": pl.Date,
    "dte": pl.Int32, "T": pl.Float64, "forward": pl.Float64,
    "discount": pl.Float64, "spot": pl.Float64, "implied_div_yield": pl.Float64,
    "n_pairs": pl.Int32, "svi_a": pl.Float64, "svi_b": pl.Float64,
    "svi_rho": pl.Float64, "svi_m": pl.Float64, "svi_sigma": pl.Float64,
    "n_points": pl.Int32, "rmse_vol_points": pl.Float64,
    "max_abs_resid_vol_points": pl.Float64, "k_min": pl.Float64,
    "k_max": pl.Float64, "butterfly_viol_rate": pl.Float64,
    "reason": pl.String, "fit_version": pl.String,
}

_SMOOTH_SCHEMA = {
    "root": pl.String, "session_date": pl.Date, "expiry": pl.Date,
    "strike": pl.Float64, "right": pl.String, "dte": pl.Int32,
    "k": pl.Float64, "iv_smooth": pl.Float64, "delta_smooth": pl.Float64,
    "extrapolated": pl.Boolean, "iv_source": pl.String,
    "surface_reason": pl.String, "fit_version": pl.String,
}


def _month_path(root: str, year: int, month: int, store: str) -> Path:
    return (
        get_local_storage_dir() / "options" / store / f"root={root}"
        / f"year={year:04d}" / f"month={month:02d}" / "data.parquet"
    )


def _session_spot(sess: pl.DataFrame) -> float:
    """One spot per session. Prefer rows taken at the registered snapshot minute."""
    clean = sess.filter(~pl.col("snapshot_fallback"))
    src = clean if clean.height else sess
    return float(src["underlying_px"].median())


def _param_row(root: str, sd: date, expiry: date, dte: int, spot: float,
               ff, fit: ExpiryFit) -> dict:
    p = fit.params
    bfly = float("nan")
    if p is not None and np.isfinite(fit.k_min):
        grid = np.linspace(fit.k_min, fit.k_max, 201)
        bfly = butterfly_violation_rate(grid, p, fit.T)
    return {
        "root": root, "session_date": sd, "expiry": expiry, "dte": dte,
        "T": fit.T, "forward": fit.forward, "discount": fit.discount,
        "spot": spot, "implied_div_yield": ff.implied_div_yield,
        "n_pairs": ff.n_pairs,
        "svi_a": p.a if p else None, "svi_b": p.b if p else None,
        "svi_rho": p.rho if p else None, "svi_m": p.m if p else None,
        "svi_sigma": p.sigma if p else None,
        "n_points": fit.n_points, "rmse_vol_points": fit.rmse_vol_points,
        "max_abs_resid_vol_points": fit.max_abs_resid_vol_points,
        "k_min": fit.k_min, "k_max": fit.k_max, "butterfly_viol_rate": bfly,
        "reason": fit.reason, "fit_version": FIT_VERSION,
    }


def _smooth_rows(root: str, sd: date, expiry: date, dte: int, spot: float,
                 strikes: np.ndarray, rights: np.ndarray,
                 fit: ExpiryFit) -> List[dict]:
    """Per-contract smoothed marks. Refused slices emit NULL + the reason."""
    p = fit.params
    out = []
    for K, right in zip(strikes, rights):
        row = {
            "root": root, "session_date": sd, "expiry": expiry,
            "strike": float(K), "right": str(right), "dte": dte,
            "k": None, "iv_smooth": None, "delta_smooth": None,
            "extrapolated": None, "iv_source": None,
            "surface_reason": fit.reason, "fit_version": FIT_VERSION,
        }
        if p is not None and np.isfinite(fit.forward) and K > 0:
            k = float(np.log(K / fit.forward))
            iv = float(smooth_iv(p, k, fit.T))
            row.update(
                k=k,
                iv_smooth=iv,
                delta_smooth=smooth_delta(
                    p, k, fit.T, fit.forward, fit.discount, spot, str(right)
                ),
                extrapolated=bool(k < fit.k_min or k > fit.k_max),
                iv_source=FIT_VERSION,
            )
        out.append(row)
    return out


def build_month_frames(root: str, year: int, month: int
                       ) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Fit every (session, expiry) in one root-month."""
    src = _month_path(root, year, month, EOD_STORE_NAME)
    if not src.exists():
        raise FileNotFoundError(f"[-] no options_chain_eod partition at {src}")

    # Divergence: float dtypes are mixed across EOD partitions (Float32 in some,
    # Float64 in others). Widen on read -- no value is altered.
    df = pl.read_parquet(src)
    for col, dtype in df.schema.items():
        if dtype == pl.Float32:
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    params, smooth = [], []
    for (sd,), sess in sorted(df.partition_by("session_date", as_dict=True).items()):
        spot = _session_spot(sess)
        if not np.isfinite(spot) or spot <= 0:
            logger.warning(f"[!] {root} {sd}: no usable spot, session skipped")
            continue
        for (expiry,), e in sorted(sess.partition_by("expiry", as_dict=True).items()):
            dte = int(e["dte"][0])
            T = max(dte, 0) / DAYS_PER_YEAR
            try:
                D = float(np.exp(-zero_rate(sd, max(T, 1e-6)) * T))
            except ValueError as exc:
                logger.error(f"[-] {root} {sd}: no discount curve: {exc}")
                continue
            c = e.filter(pl.col("right") == "C")
            p_ = e.filter(pl.col("right") == "P")
            ff = implied_forward(
                c["strike"].to_numpy(), c["mid"].to_numpy(),
                p_["strike"].to_numpy(), p_["mid"].to_numpy(),
                D=D, spot=spot, T=max(T, 1e-6),
            )
            strikes = e["strike"].to_numpy()
            rights = e["right"].to_numpy()
            if ff.reason != REASON_OK:
                # Forward refusals propagate; the slice is never fitted on a
                # forward we do not believe.
                fit = ExpiryFit(None, ff.reason, ff.forward, D, T, 0,
                                float("nan"), float("nan"),
                                float("nan"), float("nan"))
            else:
                fit = fit_expiry(strikes, rights, e["bid"].to_numpy(),
                                 e["ask"].to_numpy(), forward=ff.forward,
                                 discount=D, T=T, dte=dte, spot=spot)
            params.append(_param_row(root, sd, expiry, dte, spot, ff, fit))
            smooth.extend(
                _smooth_rows(root, sd, expiry, dte, spot, strikes, rights, fit)
            )

    return (
        pl.DataFrame(params, schema=_PARAM_SCHEMA),
        pl.DataFrame(smooth, schema=_SMOOTH_SCHEMA),
    )


def build_month(root: str, year: int, month: int,
                overwrite: bool = False) -> Optional[Path]:
    """Materialize both tables for one root-month."""
    pp = _month_path(root, year, month, SURFACE_STORE_NAME)
    sp = _month_path(root, year, month, SMOOTH_STORE_NAME)
    if pp.exists() and sp.exists() and not overwrite:
        logger.info(f"[+] exists, skipping -> {pp}")
        return pp

    params, smooth = build_month_frames(root, year, month)
    if params.height == 0:
        logger.warning(f"[!] {root} {year}-{month:02d}: nothing to write")
        return None
    for path, frame in ((pp, params), (sp, smooth)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression="zstd")

    ok = int((params["reason"] == REASON_OK).sum())
    logger.info(
        f"[+] {root} {year}-{month:02d}: {params.height:,} slices "
        f"({ok:,} OK, {params.height - ok:,} refused), "
        f"{smooth.height:,} contract rows -> {pp.parent}"
    )
    return pp


def _available_months(root: str) -> List[tuple]:
    base = get_local_storage_dir() / "options" / EOD_STORE_NAME / f"root={root}"
    found = []
    for ydir in sorted(base.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            if (mdir / "data.parquet").exists():
                found.append(
                    (int(ydir.name.split("=")[1]), int(mdir.name.split("=")[1]))
                )
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from src.utils.run_status import RunStatus

    ap = argparse.ArgumentParser(description="Build the M7 smoothed IV surface.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    months = _available_months(args.root)
    if args.year is not None:
        months = [m for m in months if m[0] == args.year]
    if args.month is not None:
        months = [m for m in months if m[1] == args.month]
    if not months:
        logger.error(f"[-] no options_chain_eod months matched root={args.root}")
        return 1

    with RunStatus(
        "m7_iv_surface", meta={"root": args.root, "months": len(months)}
    ) as status:
        for i, (y, m) in enumerate(months, start=1):
            try:
                build_month(args.root, y, m, overwrite=args.overwrite)
            except Exception as exc:
                logger.error(f"[-] {args.root} {y}-{m:02d} failed: {exc}")
                raise
            status.heartbeat(note=f"{args.root} {y}-{m:02d} ({i}/{len(months)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
