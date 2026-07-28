"""Main verification sweep: V1, V2, V3, V5, V9, V11 in a SINGLE pass per root.

MEASUREMENT ONLY -- no backtest, no P&L, no imputation, no greek recomputation,
no threshold adjustment after measurement. Every excluded row is counted.

Usage (repo root, PYTHONPATH=.):
    python scripts/data/vbattery/sweep_v1_v2_v3_v5_v9_v11.py --root SPY
    python scripts/data/vbattery/sweep_v1_v2_v3_v5_v9_v11.py --root SPY --year 2024 --month 1
    python scripts/data/vbattery/sweep_v1_v2_v3_v5_v9_v11.py --aggregate

Resumable: a root whose shards already exist is skipped unless --force.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.vbattery._sweep_lib import (
    ABS_DELTA_LABELS, BASE, DTE_LABELS, MONEYNESS_LABELS, PCTS, ROOTS,
    Reservoir, VOL_STATE_LABELS, READ_COLS, abs_delta_code, dte_code,
    iter_partition, moneyness_code, observed_dtypes, quote_fields,
    root_partitions,
)
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)

OUT_DIR = Path("output/vbattery/sweep")
SHARD_DIR = OUT_DIR / "shards"
CENSUS_STORE = Path(get_local_storage_dir()) / "options" / "derived" / "spread_census"

N_MNY, N_DTE, N_DEL, N_VOL = 8, 8, 6, 4
N_CELLS = N_MNY * N_DTE * N_DEL * N_VOL
RES_CAP = 1500
RV_LOOKBACK = 20
RV_MIN_HISTORY = 60


# ---------------------------------------------------------------- vol proxy

class VolProxy:
    """Causal, strictly-backward-looking realized-vol state proxy.

    NOT the repo's regime_state_daily. Trailing-20-session realized vol of the
    per-session last `underlying_px`, percentile-ranked against the expanding
    history of PRIOR rv observations for the same root. Labelled as a proxy.
    """

    def __init__(self):
        self.sessions: list[str] = []
        self.closes: list[float] = []
        self.rv_hist: list[float] = []
        self.state: dict[str, int] = {}

    def ingest_month(self, sess_close: dict[str, float]) -> None:
        for s in sorted(sess_close):
            if self.sessions and s <= self.sessions[-1]:
                continue
            # state for session s uses ONLY sessions strictly before s
            code = 3
            if len(self.closes) > RV_LOOKBACK and len(self.rv_hist) >= RV_MIN_HISTORY:
                c = np.array(self.closes[-(RV_LOOKBACK + 1):], dtype=float)
                r = np.diff(np.log(c))
                rv = float(np.std(r, ddof=1)) if np.all(np.isfinite(r)) else np.nan
                if np.isfinite(rv):
                    hist = np.array(self.rv_hist, dtype=float)
                    pct = float((hist < rv).mean())
                    code = 0 if pct < 0.33 else (1 if pct < 0.67 else 2)
            self.state[s] = code
            px = sess_close[s]
            if np.isfinite(px) and px > 0:
                self.sessions.append(s)
                self.closes.append(px)
                if len(self.closes) > RV_LOOKBACK:
                    c = np.array(self.closes[-(RV_LOOKBACK + 1):], dtype=float)
                    r = np.diff(np.log(c))
                    if np.all(np.isfinite(r)):
                        self.rv_hist.append(float(np.std(r, ddof=1)))


def _session_closes(path: Path) -> dict[str, float]:
    out: dict[str, tuple[str, float]] = {}
    for df in iter_partition(path, ["timestamp", "underlying_px"], batch_size=1_000_000):
        ts = df["timestamp"].astype(str)
        sess = ts.str.slice(0, 10)
        hh = ts.str.slice(11, 19)
        tmp = pd.DataFrame({"s": sess, "h": hh, "u": df["underlying_px"]})
        tmp = tmp.dropna(subset=["u"])
        if tmp.empty:
            continue
        g = tmp.sort_values("h").groupby("s").tail(1)
        for s, h, u in zip(g["s"], g["h"], g["u"]):
            cur = out.get(s)
            if cur is None or h >= cur[0]:
                out[s] = (h, float(u))
    return {k: v[1] for k, v in out.items()}


# ------------------------------------------------------------- accumulators

class RootAcc:
    def __init__(self, root: str):
        self.root = root
        self.v1 = defaultdict(lambda: np.zeros((len(ABS_DELTA_LABELS), 2), dtype=np.int64))
        self.v3 = defaultdict(lambda: np.zeros(6, dtype=np.int64))
        self.v3_zb = defaultdict(lambda: np.zeros((N_MNY, 2), dtype=np.int64))
        self.v5 = defaultdict(lambda: np.zeros(12, dtype=np.int64))
        self.sess = {}
        self.sess_minutes = defaultdict(set)
        self.sess_contracts = defaultdict(set)
        self.schema = []
        self.census_year = None
        self.census_n = None
        self.census_excl = None
        self.census_res = {}
        self.census_rows = []
        self.bad_variants = []

    def _sess(self, s: str):
        r = self.sess.get(s)
        if r is None:
            r = {"n_rows": 0, "n_rth": 0, "n_nonrth": 0, "n_vol0": 0, "n_qv": 0,
                 "n_vol0_qv": 0, "n_volpos": 0, "first_bar": "99:99", "last_bar": "00:00"}
            self.sess[s] = r
        return r

    def start_year(self, year: int):
        if self.census_year is not None:
            self.flush_year()
        self.census_year = year
        self.census_n = np.zeros(N_CELLS, dtype=np.int64)
        self.census_excl = np.zeros(N_CELLS, dtype=np.int64)
        self.census_res = {}

    def flush_year(self):
        if self.census_year is None:
            return
        y = self.census_year
        nz = np.nonzero(self.census_n + self.census_excl)[0]
        for c in nz:
            vol = c % N_VOL
            d = (c // N_VOL) % N_DEL
            dte = (c // (N_VOL * N_DEL)) % N_DTE
            mny = c // (N_VOL * N_DEL * N_DTE)
            row = {"root": self.root, "year": y,
                   "moneyness_bucket": MONEYNESS_LABELS[mny],
                   "dte_bucket": DTE_LABELS[dte],
                   "abs_delta_bucket": ABS_DELTA_LABELS[d],
                   "vol_state_proxy": VOL_STATE_LABELS[vol],
                   "n_valid_quotes": int(self.census_n[c]),
                   "n_excluded_invalid_quote": int(self.census_excl[c])}
            res = self.census_res.get(int(c))
            samp = res.sample() if res is not None else np.empty((0, 3))
            row["n_sampled"] = int(samp.shape[0])
            for j, nm in enumerate(["spread_abs", "spread_rel", "mid"]):
                if samp.shape[0]:
                    row[f"{nm}_mean"] = float(np.mean(samp[:, j]))
                    qs = np.percentile(samp[:, j], PCTS)
                    for p, q in zip(PCTS, qs):
                        row[f"{nm}_p{p}"] = float(q)
                else:
                    row[f"{nm}_mean"] = np.nan
                    for p in PCTS:
                        row[f"{nm}_p{p}"] = np.nan
            self.census_rows.append(row)
        self.census_year = None
        self.census_res = {}


def process_partition(acc: RootAcc, year: int, month: int, path: Path,
                      vol: VolProxy, batch_size: int) -> int:
    dt = observed_dtypes(path)
    acc.schema.append({"root": acc.root, "year": year, "month": month,
                       "n_cols": len(dt), **{f"dtype_{k}": v for k, v in dt.items()}})
    known = {"expiration": {"large_string", "string", "date32[day]"},
             "volume": {"int32", "int64"}}
    for c, ok in known.items():
        if c in dt and dt[c] not in ok:
            acc.bad_variants.append({"root": acc.root, "year": year, "month": month,
                                     "column": c, "dtype": dt[c]})

    vol.ingest_month(_session_closes(path))

    n_rows = 0
    for df in iter_partition(path, READ_COLS, batch_size=batch_size):
        n = len(df)
        if n == 0:
            continue
        n_rows += n
        ts = df["timestamp"].astype(str)
        sess = ts.str.slice(0, 10).to_numpy()
        hhmm = ts.str.slice(11, 16).to_numpy()

        bid = df["bid_close"].to_numpy(dtype=np.float64)
        ask = df["ask_close"].to_numpy(dtype=np.float64)
        q = quote_fields(bid, ask)
        qv = q["quote_valid"]
        volume = df["volume"].to_numpy(dtype=np.int64)
        vol0 = volume == 0

        in_rth = (hhmm >= "09:30") & (hhmm <= "16:00")

        # ---- session-level (V2 / V9) ----
        scodes, suniq = pd.factorize(sess)
        ns = len(suniq)
        cnt = np.bincount(scodes, minlength=ns)
        c_rth = np.bincount(scodes[in_rth], minlength=ns)
        c_vol0 = np.bincount(scodes[vol0], minlength=ns)
        c_qv = np.bincount(scodes[qv], minlength=ns)
        c_v0qv = np.bincount(scodes[vol0 & qv], minlength=ns)
        mins = (np.char.replace(hhmm.astype("U5"), ":", "").astype(np.int64))
        mins = (mins // 100) * 60 + (mins % 100)
        for i, s in enumerate(suniq):
            r = acc._sess(str(s))
            r["n_rows"] += int(cnt[i])
            r["n_rth"] += int(c_rth[i])
            r["n_nonrth"] += int(cnt[i] - c_rth[i])
            r["n_vol0"] += int(c_vol0[i])
            r["n_qv"] += int(c_qv[i])
            r["n_vol0_qv"] += int(c_v0qv[i])
            r["n_volpos"] += int(cnt[i] - c_vol0[i])
        hmin = pd.Series(hhmm).groupby(scodes).min()
        hmax = pd.Series(hhmm).groupby(scodes).max()
        for i, s in enumerate(suniq):
            r = acc._sess(str(s))
            r["first_bar"] = min(r["first_bar"], str(hmin.iloc[i]))
            r["last_bar"] = max(r["last_bar"], str(hmax.iloc[i]))

        # ---- contract identity, DTE ----
        exp_s = df["expiration"].astype(str).str.slice(0, 10)
        ec, eu = pd.factorize(exp_s)
        eu_ord = pd.to_datetime(pd.Series(eu), errors="coerce").values.astype(
            "datetime64[D]").astype(np.int64)
        exp_ord = eu_ord[ec]
        su_ord = pd.to_datetime(pd.Series(suniq), errors="coerce").values.astype(
            "datetime64[D]").astype(np.int64)
        sess_ord = su_ord[scodes]
        dte = (exp_ord - sess_ord).astype(np.float64)

        strike = df["strike"].to_numpy(dtype=np.float64)
        rgt = df["right"].astype(str).to_numpy()
        is_call = (rgt == "CALL")
        skm = np.where(np.isfinite(strike), np.round(strike * 1000.0), -1).astype(np.int64)
        ckey = (exp_ord * 2 + is_call.astype(np.int64)) * 20_000_000 + skm
        pair = scodes.astype(np.int64) * 1_000_000_000_000 + ckey
        up = np.unique(pair)
        for pv in up:
            si = int(pv // 1_000_000_000_000)
            acc.sess_contracts[str(suniq[si])].add(int(pv % 1_000_000_000_000))
        mpair = np.unique(scodes.astype(np.int64) * 10_000 + mins)
        for pv in mpair:
            acc.sess_minutes[str(suniq[int(pv // 10_000)])].add(int(pv % 10_000))

        # ---- V1 (RTH only) ----
        dcode = abs_delta_code(df["delta"].to_numpy(dtype=np.float64))
        a1 = acc.v1[year]
        a1[:, 0] += np.bincount(dcode[in_rth], minlength=len(ABS_DELTA_LABELS))
        a1[:, 1] += np.bincount(dcode[in_rth & qv], minlength=len(ABS_DELTA_LABELS))

        # ---- V3 ----
        und = df["underlying_px"].to_numpy(dtype=np.float64)
        mcode, _ = moneyness_code(strike, und)
        a3 = acc.v3[year]
        a3[0] += int(in_rth.sum())
        a3[1] += int((q["crossed"] & in_rth).sum())
        a3[2] += int((q["locked"] & in_rth).sum())
        a3[3] += int((q["zero_bid"] & in_rth).sum())
        a3[4] += int((qv & in_rth).sum())
        a3[5] += int(((~q["finite"]) & in_rth).sum())
        z3 = acc.v3_zb[year]
        z3[:, 0] += np.bincount(mcode[in_rth], minlength=N_MNY)
        z3[:, 1] += np.bincount(mcode[in_rth & q["zero_bid"]], minlength=N_MNY)

        # ---- V5 ----
        iv = df["implied_vol"].to_numpy(dtype=np.float64)
        dl = df["delta"].to_numpy(dtype=np.float64)
        th = df["theta"].to_numpy(dtype=np.float64)
        vg = df["vega"].to_numpy(dtype=np.float64)
        a5 = acc.v5[year]
        a5[0] += n
        a5[1] += int(np.isfinite(iv).sum())
        a5[2] += int(np.isfinite(dl).sum())
        a5[3] += int(np.isfinite(th).sum())
        a5[4] += int(np.isfinite(vg).sum())
        a5[5] += int((np.isfinite(iv) & (iv > 0.01) & (iv < 5.0)).sum())
        a5[6] += int((np.isfinite(dl) & (np.abs(dl) <= 1.0)).sum())
        sign_ok = (np.isfinite(dl) & ((is_call & (dl >= 0)) | ((~is_call) & (dl <= 0))))
        a5[7] += int(sign_ok.sum())
        iv_half = np.isfinite(iv) & (iv == 0.5)
        a5[8] += int(iv_half.sum())
        a5[9] += int((iv_half & (~qv)).sum())
        a5[10] += int((~qv).sum())
        a5[11] += int((np.isfinite(dl) & is_call & (dl < 0)).sum() +
                      (np.isfinite(dl) & (~is_call) & (dl > 0)).sum())

        # ---- V11 census (valid quotes only) ----
        vcode = np.array([vol.state.get(str(s), 3) for s in suniq], dtype=np.int8)[scodes]
        cell = (((mcode.astype(np.int64) * N_DTE + dte_code(dte)) * N_DEL
                 + dcode) * N_VOL + vcode)
        cell_rth = cell[in_rth]
        sel = qv[in_rth]
        acc.census_n += np.bincount(cell_rth[sel], minlength=N_CELLS)
        acc.census_excl += np.bincount(cell_rth[~sel], minlength=N_CELLS)
        if sel.any():
            b = bid[in_rth][sel]
            a = ask[in_rth][sel]
            md = (b + a) / 2.0
            sa = a - b
            with np.errstate(divide="ignore", invalid="ignore"):
                sr = np.where(md > 0, sa / md, np.nan)
            blk = np.column_stack([sa, sr, md])
            cs = cell_rth[sel]
            order = np.argsort(cs, kind="stable")
            cs_s = cs[order]
            blk_s = blk[order]
            edges = np.searchsorted(cs_s, np.unique(cs_s), side="left")
            uniq = np.unique(cs_s)
            edges = np.append(edges, len(cs_s))
            for i, c in enumerate(uniq):
                res = acc.census_res.get(int(c))
                if res is None:
                    res = Reservoir(RES_CAP, 3, seed=int(c) * 7919 + year)
                    acc.census_res[int(c)] = res
                res.add(blk_s[edges[i]:edges[i + 1]])
    return n_rows


# ------------------------------------------------------------------ writing

def write_shards(acc: RootAcc, meta: dict) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    r = acc.root

    rows = []
    for y, a in sorted(acc.v1.items()):
        for i, lab in enumerate(ABS_DELTA_LABELS):
            rows.append({"root": r, "year": y, "abs_delta_bucket": lab,
                         "n_rth_rows": int(a[i, 0]), "n_quote_valid": int(a[i, 1]),
                         "frac_quote_valid": (float(a[i, 1]) / a[i, 0]) if a[i, 0] else np.nan})
    pd.DataFrame(rows).to_parquet(SHARD_DIR / f"v1_{r}.parquet", index=False)

    srows = []
    for s, d in sorted(acc.sess.items()):
        srows.append({"root": r, "session": s, **d,
                      "n_contracts": len(acc.sess_contracts.get(s, ())),
                      "n_minutes": len(acc.sess_minutes.get(s, ()))})
    sdf = pd.DataFrame(srows)
    sdf.to_parquet(SHARD_DIR / f"v2sessions_{r}.parquet", index=False)

    rows = []
    for y, a in sorted(acc.v3.items()):
        z = acc.v3_zb[y]
        base = {"root": r, "year": y, "n_rth_rows": int(a[0]),
                "n_crossed": int(a[1]), "n_locked": int(a[2]),
                "n_zero_bid": int(a[3]), "n_quote_valid": int(a[4]),
                "n_nonfinite_quote": int(a[5]),
                "frac_crossed": float(a[1]) / a[0] if a[0] else np.nan,
                "frac_locked": float(a[2]) / a[0] if a[0] else np.nan,
                "frac_zero_bid": float(a[3]) / a[0] if a[0] else np.nan}
        for i, lab in enumerate(MONEYNESS_LABELS):
            base[f"zerobid_frac_{lab}"] = (float(z[i, 1]) / z[i, 0]) if z[i, 0] else np.nan
            base[f"n_{lab}"] = int(z[i, 0])
        rows.append(base)
    pd.DataFrame(rows).to_parquet(SHARD_DIR / f"v3_{r}.parquet", index=False)

    rows = []
    for y, a in sorted(acc.v5.items()):
        n = int(a[0])
        rows.append({"root": r, "year": y, "n_rows": n,
                     "nonnull_implied_vol": float(a[1]) / n if n else np.nan,
                     "nonnull_delta": float(a[2]) / n if n else np.nan,
                     "nonnull_theta": float(a[3]) / n if n else np.nan,
                     "nonnull_vega": float(a[4]) / n if n else np.nan,
                     "plausible_iv_frac": float(a[5]) / n if n else np.nan,
                     "abs_delta_le1_frac": float(a[6]) / n if n else np.nan,
                     "delta_sign_ok_frac": float(a[7]) / n if n else np.nan,
                     "iv_eq_0p5_frac": float(a[8]) / n if n else np.nan,
                     "iv_eq_0p5_n": int(a[8]),
                     "n_invalid_quote": int(a[10]),
                     "iv_eq_0p5_frac_of_invalid_quote":
                         float(a[9]) / a[10] if a[10] else np.nan,
                     "delta_sign_wrong_n": int(a[11])})
    pd.DataFrame(rows).to_parquet(SHARD_DIR / f"v5_{r}.parquet", index=False)

    pd.DataFrame(acc.census_rows).to_parquet(SHARD_DIR / f"v11_{r}.parquet", index=False)
    pd.DataFrame(acc.schema).to_parquet(SHARD_DIR / f"schema_{r}.parquet", index=False)
    (SHARD_DIR / f"meta_{r}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def sweep_root(root: str, batch_size: int, only_year=None, only_month=None,
               force: bool = False) -> dict:
    done = SHARD_DIR / f"meta_{root}.json"
    if done.exists() and not force and only_year is None:
        logger.info(f"[=] {root}: shards exist, skipping (use --force to redo)")
        return {"root": root, "skipped": True}

    parts = root_partitions(root)
    if only_year is not None:
        parts = [p for p in parts if p[0] == only_year and
                 (only_month is None or p[1] == only_month)]
    if not parts:
        logger.warning(f"[!] {root}: no partitions found")
        return {"root": root, "n_partitions": 0}

    acc = RootAcc(root)
    vol = VolProxy()
    t0 = time.time()
    total_rows = 0
    cur_year = None
    with RunStatus(f"vbattery_sweep_{root}",
                   meta={"root": root, "n_partitions": len(parts)}) as st:
        for i, (y, m, path) in enumerate(parts):
            if y != cur_year:
                acc.start_year(y)
                cur_year = y
            total_rows += process_partition(acc, y, m, path, vol, batch_size)
            st.heartbeat(note=f"{root} {y}-{m:02d} ({i+1}/{len(parts)}) rows={total_rows}")
        acc.flush_year()
        meta = {"root": root, "n_partitions": len(parts), "n_rows": total_rows,
                "wall_seconds": round(time.time() - t0, 1),
                "partitions": [f"{y}-{m:02d}" for y, m, _ in parts],
                "unhandled_schema_variants": acc.bad_variants,
                "vol_state_proxy": ("trailing-20-session realized vol of per-session "
                                    "last underlying_px, expanding-percentile ranked, "
                                    "strictly causal; NOT regime_state_daily"),
                "census_percentiles": "reservoir-sampled estimate, cap=%d/cell" % RES_CAP}
        write_shards(acc, meta)
    logger.info(f"[+] {root}: {total_rows:,} rows, {len(parts)} partitions, "
                f"{meta['wall_seconds']}s")
    return meta


# ---------------------------------------------------------------- aggregate

def _expected_sessions() -> pd.DataFrame:
    from src.backtesting.utils.market_calendar import MarketCalendar
    cal = MarketCalendar("NYSE")
    sched = cal.calendar.schedule(start_date="2012-01-01", end_date="2026-12-31")
    d = pd.DataFrame({"session": sched.index.strftime("%Y-%m-%d"),
                      "close_utc": sched["market_close"].values})
    ct = pd.to_datetime(d["close_utc"], utc=True).dt.tz_convert("America/New_York")
    d["expected_close"] = ct.dt.strftime("%H:%M")
    d["is_half_day"] = d["expected_close"] < "16:00"
    return d[["session", "expected_close", "is_half_day"]]


def aggregate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag in ["v1", "v3", "v5", "v11", "schema"]:
        fs = sorted(SHARD_DIR.glob(f"{tag}_*.parquet"))
        if not fs:
            continue
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        df.to_parquet(OUT_DIR / f"{tag}_all.parquet", index=False)
        logger.info(f"[+] {tag}_all.parquet rows={len(df)}")

    fs = sorted(SHARD_DIR.glob("v2sessions_*.parquet"))
    if not fs:
        return
    sdf = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    exp = _expected_sessions()
    sdf = sdf.merge(exp, on="session", how="left")
    sdf["grid_density"] = sdf["n_rows"] / (sdf["n_contracts"] * sdf["n_minutes"])
    sdf["frac_vol0_validquote"] = sdf["n_vol0_qv"] / sdf["n_rows"]
    sdf["last_bar_matches_calendar"] = sdf["last_bar"] == sdf["expected_close"]
    sdf.to_parquet(OUT_DIR / "v2_sessions_all.parquet", index=False)
    logger.info(f"[+] v2_sessions_all.parquet rows={len(sdf)}")

    # V9: root-month coverage vs calendar
    sdf["year"] = sdf["session"].str.slice(0, 4).astype(int)
    sdf["month"] = sdf["session"].str.slice(5, 7).astype(int)
    got = sdf.groupby(["root", "year", "month"]).agg(
        sessions_present=("session", "nunique"),
        n_half_days_present=("is_half_day", "sum")).reset_index()
    exp["year"] = exp["session"].str.slice(0, 4).astype(int)
    exp["month"] = exp["session"].str.slice(5, 7).astype(int)
    expm = exp.groupby(["year", "month"]).agg(
        sessions_expected=("session", "nunique"),
        half_days_expected=("is_half_day", "sum")).reset_index()
    v9 = got.merge(expm, on=["year", "month"], how="left")
    v9["coverage"] = v9["sessions_present"] / v9["sessions_expected"]
    v9["flagged_lt_90pct"] = v9["coverage"] < 0.90
    v9.to_parquet(OUT_DIR / "v9_all.parquet", index=False)
    logger.info(f"[+] v9_all.parquet rows={len(v9)}")

    cf = OUT_DIR / "v11_all.parquet"
    if cf.exists():
        CENSUS_STORE.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(cf)
        df.to_parquet(CENSUS_STORE / "spread_census.parquet", index=False)
        df.to_parquet(OUT_DIR / "spread_census.parquet", index=False)
        logger.info(f"[+] spread_census.parquet rows={len(df)} -> "
                    f"{CENSUS_STORE / 'spread_census.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    ap.add_argument("--batch-size", type=int, default=400_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return
    if not args.root:
        raise SystemExit("--root or --aggregate required")
    roots = ROOTS if args.root == "ALL" else args.root.split(",")
    for r in roots:
        sweep_root(r, args.batch_size, args.year, args.month, args.force)


if __name__ == "__main__":
    main()
