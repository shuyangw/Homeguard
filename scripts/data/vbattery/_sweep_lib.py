"""Shared primitives for the V1/V2/V3/V5/V9/V11 single-pass options sweep.

MEASUREMENT ONLY. No imputation, no smoothing, no recomputation of greeks.
Every excluded row lands in a counted bucket.

Registered definitions (implemented once, here, so a later behavioural
equivalence check against src/data/options/canonical.py has a single target):

    quote_valid := bid_close > 0 AND ask_close >= bid_close AND both finite
    crossed     := bid_close > ask_close
    locked      := bid_close == ask_close AND both > 0
    zero_bid    := bid_close == 0 OR bid_close is null
    mid         := (bid_close + ask_close) / 2, defined ONLY where quote_valid
    moneyness   := log(strike / underlying_px)
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir

BASE = Path(get_local_storage_dir()) / "options" / "options_combined"

ROOTS = ["AAPL", "AMD", "AMZN", "AVGO", "COIN", "DIA", "EEM", "FB", "FXI", "GLD",
         "GOOGL", "IBIT", "IWM", "META", "MSFT", "MSTR", "NVDA", "PLTR", "QQQ",
         "SLV", "SMH", "SPX", "SPY", "TLT", "TSLA", "VIX", "XLE", "XLF", "XLI",
         "XLK", "XLV"]

INDEX_ROOTS = {"SPX", "SPY", "QQQ", "IWM"}

# --- registered bucket edges (declared BEFORE measuring) ---------------------

# V1 gate binds on ABS_DELTA_LABELS[1] == "d_0.05_0.15".
# Half-open on the lower two gate buckets, closed on the top one, so no row is
# double counted. Stated up front, not adjusted after measurement.
ABS_DELTA_LABELS = ["d_0_0.05", "d_0.05_0.15", "d_0.15_0.35", "d_0.35_0.65",
                    "d_0.65_1", "d_null_or_oob"]
V1_GATE_BUCKET = "d_0.05_0.15"

MONEYNESS_LABELS = ["m_le_-0.10", "m_-0.10_-0.05", "m_-0.05_-0.02", "m_atm_-0.02_0.02",
                    "m_0.02_0.05", "m_0.05_0.10", "m_gt_0.10", "m_undefined"]
MONEYNESS_EDGES = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]

DTE_LABELS = ["dte_0", "dte_1_7", "dte_8_30", "dte_31_60", "dte_61_90",
              "dte_91_180", "dte_181p", "dte_negative_or_null"]

VOL_STATE_LABELS = ["vol_low", "vol_mid", "vol_high", "vol_unknown"]

RTH_FIRST = "09:30"
RTH_LAST = "16:00"

READ_COLS = ["timestamp", "expiration", "strike", "right", "volume",
             "bid_close", "ask_close", "implied_vol", "delta", "theta", "vega",
             "underlying_px"]

CANONICAL_DTYPES = {
    "timestamp": "string", "expiration": "string", "strike": "float64",
    "right": "string", "volume": "int64", "bid_close": "float64",
    "ask_close": "float64", "implied_vol": "float64", "delta": "float64",
    "theta": "float64", "vega": "float64", "underlying_px": "float64",
}


# --- partition discovery / dtype-tolerant reading ----------------------------

def root_partitions(root: str) -> List[Tuple[int, int, Path]]:
    rp = BASE / f"root={root}"
    out = []
    if not rp.exists():
        return out
    for yp in sorted(rp.glob("year=*")):
        for mp in sorted(yp.glob("month=*")):
            f = mp / "data.parquet"
            if f.exists():
                out.append((int(yp.name.split("=")[1]), int(mp.name.split("=")[1]), f))
    return out


def observed_dtypes(path: Path) -> Dict[str, str]:
    sch = pq.ParquetFile(path).schema_arrow
    return {n: str(sch.field(n).type) for n in sch.names}


def iter_partition(path: Path, cols: List[str], batch_size: int = 400_000
                   ) -> Iterator[pd.DataFrame]:
    """Read ONE partition file, casting to the canonical dtype set per file.

    A schema fork exists in the store (expiration Date vs String; volume Int32
    vs Int64), so each file is read independently and normalised here rather
    than globbing a whole root-year and assuming a uniform schema.
    """
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    avail = [c for c in cols if c in names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=avail):
        df = batch.to_pandas()
        for c in avail:
            want = CANONICAL_DTYPES.get(c)
            if want == "string":
                df[c] = df[c].astype("string").astype(object)
            elif want == "float64":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            elif want == "int64":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype("int64")
        for c in cols:
            if c not in avail:
                df[c] = np.nan
        yield df


# --- registered predicates (SINGLE definition site) --------------------------

def quote_fields(bid: np.ndarray, ask: np.ndarray) -> Dict[str, np.ndarray]:
    """The registered quote predicates + mid. One function, one definition."""
    finite = np.isfinite(bid) & np.isfinite(ask)
    quote_valid = finite & (bid > 0) & (ask >= bid)
    crossed = finite & (bid > ask)
    locked = finite & (bid == ask) & (bid > 0)
    zero_bid = (~np.isfinite(bid)) | (bid == 0)
    mid = np.where(quote_valid, (bid + ask) / 2.0, np.nan)
    return {"finite": finite, "quote_valid": quote_valid, "crossed": crossed,
            "locked": locked, "zero_bid": zero_bid, "mid": mid}


def abs_delta_code(delta: np.ndarray) -> np.ndarray:
    ad = np.abs(delta)
    code = np.full(ad.shape, 5, dtype=np.int8)
    ok = np.isfinite(ad) & (ad <= 1.0)
    code[ok & (ad < 0.05)] = 0
    code[ok & (ad >= 0.05) & (ad < 0.15)] = 1
    code[ok & (ad >= 0.15) & (ad < 0.35)] = 2
    code[ok & (ad >= 0.35) & (ad <= 0.65)] = 3
    code[ok & (ad > 0.65)] = 4
    return code


def moneyness_code(strike: np.ndarray, und: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log(strike / und)
    ok = np.isfinite(m)
    code = np.full(m.shape, 7, dtype=np.int8)
    idx = np.searchsorted(np.array(MONEYNESS_EDGES), m[ok], side="right")
    code[ok] = idx.astype(np.int8)
    return code, m


def dte_code(dte: np.ndarray) -> np.ndarray:
    code = np.full(dte.shape, 7, dtype=np.int8)
    ok = np.isfinite(dte) & (dte >= 0)
    d = dte
    code[ok & (d == 0)] = 0
    code[ok & (d >= 1) & (d <= 7)] = 1
    code[ok & (d >= 8) & (d <= 30)] = 2
    code[ok & (d >= 31) & (d <= 60)] = 3
    code[ok & (d >= 61) & (d <= 90)] = 4
    code[ok & (d >= 91) & (d <= 180)] = 5
    code[ok & (d >= 181)] = 6
    return code


# --- reservoir for V11 percentile estimation ---------------------------------

class Reservoir:
    """Algorithm-R style reservoir over an (n, k) value block.

    Percentiles for the V11 census are ESTIMATED from this sample, not exact --
    exact percentiles over a root-year cell would require materialising every
    contract-minute. Sample size and the exact cell n are both reported so the
    estimate is auditable.
    """

    __slots__ = ("cap", "k", "buf", "n_filled", "seen", "rng")

    def __init__(self, cap: int, k: int, seed: int):
        self.cap = cap
        self.k = k
        self.buf = np.empty((cap, k), dtype=np.float64)
        self.n_filled = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, block: np.ndarray) -> None:
        m = block.shape[0]
        if m == 0:
            return
        free = self.cap - self.n_filled
        take = min(free, m)
        if take:
            self.buf[self.n_filled:self.n_filled + take] = block[:take]
            self.n_filled += take
        rest = block[take:]
        r = rest.shape[0]
        if r:
            i_arr = np.arange(self.seen + take, self.seen + take + r, dtype=np.int64)
            j = (self.rng.random(r) * (i_arr + 1)).astype(np.int64)
            sel = j < self.cap
            if sel.any():
                self.buf[j[sel]] = rest[sel]
        self.seen += m

    def sample(self) -> np.ndarray:
        return self.buf[:self.n_filled]


PCTS = [10, 25, 50, 75, 90, 95, 99]
