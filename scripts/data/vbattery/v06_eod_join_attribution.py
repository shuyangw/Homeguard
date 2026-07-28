"""V6 -- EOD-join date attribution for `open_interest_eod` / `gamma_eod`.

Registered V6 gate (work order Section 3, verbatim):
    "HARD GATE. (a) read the join in `combine_options_data.py`; (b) verify
    `open_interest_eod` is constant across all minutes of a session (if it varies
    intraday the join is broken); (c) establish whether the value on session t is
    OI as-of t's close or t-1's close. Same-day join => all OI/gamma features must
    lag >= 1 session, enforced in the primitive layer, not per-strategy. If the
    answer cannot be established from code + data => all OI/gamma work BLOCKED."

This script answers sub-item (c), which the prior Phase-0 pass did not settle.

Discriminator for OI
--------------------
Open interest changes because of the trading that happened during a session. So:
  - if the value stamped on date t is t's CLOSING OI, then OI[t] - OI[t-1] is
    driven by volume[t];
  - if it is the START-OF-DAY OI (i.e. t-1's close, which is what exchanges
    actually publish on the morning of t), then OI[t+1] - OI[t] is driven by
    volume[t].
Comparing the two correlations, and the ratio mean(|dOI|)/volume (which should sit
near 1.0 for the correct alignment), identifies which. A secondary check: on a
contract's first traded session, start-of-day OI must be 0.

This is a DATA-PROPERTY measurement. No strategy P&L is computed anywhere.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = os.path.join("output", "vbattery", "v06")

_NEW_LISTING_BUFFER = 15


def _month_files(root: str, year: int, months: list[int]) -> list[str]:
    base = os.path.join(get_local_storage_dir(), "options", "options_combined")
    paths = [
        os.path.join(base, f"root={root}", f"year={year}", f"month={m:02d}", "data.parquet")
        for m in months
    ]
    return [p for p in paths if os.path.exists(p)]


def contract_sessions(files: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(files)
    frame = (
        lf.select(
            ["timestamp", "expiration", "strike", "right", "volume", "open_interest_eod"]
        )
        .with_columns(pl.col("timestamp").str.slice(0, 10).alias("session"))
        .group_by(["session", "expiration", "strike", "right"])
        .agg(
            pl.col("volume").sum().alias("vol"),
            pl.col("open_interest_eod").first().alias("oi"),
            pl.col("open_interest_eod").n_unique().alias("oi_nunique"),
        )
        .collect()
    )
    return frame.sort(["expiration", "strike", "right", "session"])


def measure(frame: pl.DataFrame) -> dict:
    key = ["expiration", "strike", "right"]
    frame = frame.with_columns(
        [
            pl.col("oi").diff().over(key).alias("doi_same"),
            pl.col("oi").diff().shift(-1).over(key).alias("doi_next"),
        ]
    )

    intraday_varying = int((frame["oi_nunique"] > 1).sum())

    traded = frame.drop_nulls(["doi_same", "doi_next"]).filter(pl.col("vol") > 0)
    vol = traded["vol"].to_numpy().astype(float)
    same = traded["doi_same"].to_numpy().astype(float)
    nxt = traded["doi_next"].to_numpy().astype(float)

    # A contract whose first OBSERVED session is the window start is almost always
    # window-truncated, not newly listed -- including it destroys the test. Require
    # the first observed session to sit at least `_NEW_LISTING_BUFFER` sessions
    # inside the window so the contract is genuinely newly listed.
    sessions = sorted(frame["session"].unique().to_list())
    if len(sessions) <= _NEW_LISTING_BUFFER:
        cutoff = sessions[-1] if sessions else ""
    else:
        cutoff = sessions[_NEW_LISTING_BUFFER]

    first_session = (
        frame.group_by(key)
        .agg(pl.col("session").min().alias("first_session"))
        .join(frame, on=key)
        .filter(pl.col("session") == pl.col("first_session"))
        .filter(pl.col("first_session") > cutoff)
        .filter(pl.col("vol") > 0)
    )

    return {
        "contract_sessions": int(len(frame)),
        "contract_sessions_with_intraday_varying_oi": intraday_varying,
        "traded_contract_sessions": int(len(traded)),
        "corr_vol_t_vs_oi_t_minus_oi_tm1": float(np.corrcoef(vol, same)[0, 1]),
        "corr_vol_t_vs_oi_tp1_minus_oi_t": float(np.corrcoef(vol, nxt)[0, 1]),
        "mean_abs_doi_over_vol_same_day": float(np.mean(np.abs(same) / vol)),
        "mean_abs_doi_over_vol_next_day": float(np.mean(np.abs(nxt) / vol)),
        "first_traded_sessions": int(len(first_session)),
        "frac_first_traded_session_oi_zero": float((first_session["oi"] == 0).mean()),
    }


def verdict(stats: dict) -> str:
    same = stats["corr_vol_t_vs_oi_t_minus_oi_tm1"]
    nxt = stats["corr_vol_t_vs_oi_tp1_minus_oi_t"]
    if nxt > same:
        return "START_OF_DAY (date-t value == t-1 close OI; same-day join is NOT a leak)"
    if same > nxt:
        return "SAME_DAY_CLOSE (date-t value == t close OI; same-day join IS a leak)"
    return "INDETERMINATE"


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 EOD-join date attribution")
    parser.add_argument("--root", default="SPY")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1,2")
    args = parser.parse_args()

    months = [int(m) for m in args.months.split(",")]
    files = _month_files(args.root, args.year, months)
    if not files:
        logger.error(f"[-] no partitions found for root={args.root} year={args.year}")
        return

    logger.info(f"[*] V6 attribution: root={args.root} year={args.year} months={months}")
    frame = contract_sessions(files)
    stats = measure(frame)
    stats["root"] = args.root
    stats["year"] = args.year
    stats["months"] = args.months
    stats["verdict"] = verdict(stats)

    for name, value in stats.items():
        logger.info(f"    {name}: {value}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"v06_attribution_{args.root}_{args.year}.parquet")
    pl.DataFrame([stats]).write_parquet(out)
    logger.info(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
