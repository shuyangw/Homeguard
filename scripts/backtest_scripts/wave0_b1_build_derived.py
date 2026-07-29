"""Wave 0 Group B1 -- build the derived daily state tables.

DATA-PROPERTY MEASUREMENT ONLY. No positions, no P&L.

Outputs (parquet) under output/wave0/derived/:
    atm_iv_daily.parquet      root, session_date, dte_bucket, atm_iv, source_dte, n_contracts
    atm_per_expiry.parquet    the full per-listed-expiry ATM surface (support table)
    skew_daily.parquet        root, session_date, dte_bucket, iv_25d_put/call, skew_25d, curvature
    term_slope_daily.parquet  root, session_date, m1_atm_iv, m2_atm_iv, slope
    iv_rank_daily.parquet     root, session_date, dte_bucket, iv_rank_1y, iv_pctile_1y/2y
    rv_daily.parquet          root, session_date, yz_10d, yz_20d, rv_1m_daily_var, har_forecast_var
    build_census.json         every exclusion, counted
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtesting.diagnostics import options_iv_state as ivs
from src.backtesting.diagnostics.session_bars import session_close_schedule
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT = Path("output/wave0/derived")
ROOT_WINDOWS = {"SPY": (2017, 2025), "QQQ": (2012, 2025)}
RV_SYMBOLS = {"SPY": range(2016, 2027), "QQQ": range(2016, 2027)}


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return str(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    census = {}
    with RunStatus("wave0_b1_derived", meta={"scope": "groupB1_derived_tables"}) as status:
        td = set(session_close_schedule("2010-01-01", "2026-07-01").keys())
        census["coverage"] = {
            r: {"n_partitions_on_disk": len(ivs.available_partitions(r)),
                "first": f"{ivs.available_partitions(r)[0][0]}-{ivs.available_partitions(r)[0][1]:02d}",
                "last": f"{ivs.available_partitions(r)[-1][0]}-{ivs.available_partitions(r)[-1][1]:02d}"}
            for r in ROOT_WINDOWS}

        atm_all, pe_all, skew_all, term_all, put_all = [], [], [], [], []
        for root, (lo, hi) in ROOT_WINDOWS.items():
            status.heartbeat(note=f"atm {root}")
            a, pe, st = ivs.build_atm_iv_daily(root, lo, hi)
            census[f"atm_iv_daily.{root}"] = st
            atm_all.append(a)
            pe_all.append(pe)
            logger.info(f"[+] atm_iv_daily {root}: {len(a)} rows, {st['n_sessions']} sessions")

            status.heartbeat(note=f"skew {root}")
            s, st2 = ivs.build_skew_daily(root, lo, hi)
            census[f"skew_daily.{root}"] = st2
            skew_all.append(s)
            logger.info(f"[+] skew_daily {root}: {len(s)} rows")

            status.heartbeat(note=f"term {root}")
            t, st3 = ivs.build_term_slope_daily(root, lo, hi, trading_days=td)
            census[f"term_slope_daily.{root}"] = st3
            term_all.append(t)
            logger.info(f"[+] term_slope_daily {root}: {len(t)} rows")

            status.heartbeat(note=f"putiv {root}")
            p, st4 = ivs.build_delta_iv_daily(root, lo, hi, right="P",
                                              targets=(0.50, 0.30),
                                              buckets=(45, 60))
            census[f"put_iv_daily.{root}"] = st4
            put_all.append(p)
            logger.info(f"[+] put_iv_daily {root}: {len(p)} rows")

        atm = pd.concat(atm_all, ignore_index=True)
        atm.to_parquet(OUT / "atm_iv_daily.parquet", index=False)
        pd.concat(pe_all, ignore_index=True).to_parquet(
            OUT / "atm_per_expiry.parquet", index=False)
        pd.concat(skew_all, ignore_index=True).to_parquet(
            OUT / "skew_daily.parquet", index=False)
        pd.concat(term_all, ignore_index=True).to_parquet(
            OUT / "term_slope_daily.parquet", index=False)
        pd.concat(put_all, ignore_index=True).to_parquet(
            OUT / "put_iv_daily.parquet", index=False)

        status.heartbeat(note="iv_rank")
        rank = ivs.build_iv_rank_daily(atm)
        rank.to_parquet(OUT / "iv_rank_daily.parquet", index=False)
        first_valid = {}
        for (r, b), g in rank.groupby(["root", "dte_bucket"]):
            v1 = g.dropna(subset=["iv_pctile_1y"])
            v2 = g.dropna(subset=["iv_pctile_2y"])
            first_valid[f"{r}_{b}"] = {
                "first_1y": str(v1["session_date"].min()) if len(v1) else None,
                "first_2y": str(v2["session_date"].min()) if len(v2) else None,
                "n_rows": int(len(g)),
                "n_valid_1y": int(len(v1)), "n_valid_2y": int(len(v2))}
        census["iv_rank_daily.first_valid"] = first_valid
        logger.info(f"[+] iv_rank_daily: {len(rank)} rows")

        rv_all = []
        for sym, yrs in RV_SYMBOLS.items():
            status.heartbeat(note=f"rv {sym}")
            d, st = ivs.build_rv_daily(sym, list(yrs))
            census[f"rv_daily.{sym}"] = st
            rv_all.append(d)
            logger.info(f"[+] rv_daily {sym}: {len(d)} sessions")
        pd.concat(rv_all, ignore_index=True).to_parquet(
            OUT / "rv_daily.parquet", index=False)

        (OUT / "build_census.json").write_text(
            json.dumps(_json_safe(census), indent=2), encoding="utf-8")
    logger.info("[+] derived tables written to output/wave0/derived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
