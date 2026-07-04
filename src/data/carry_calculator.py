"""Per-asset-class carry calculator.

Front-month and second-month identified by volume ranking from
futures/databento/per_contract_1min/ on the target date. Outright contracts
only (spreads filtered via month-code regex).
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.data.derivations.futures.sofr import derive_sofr
from src.data.futures.paths import per_contract_1min_dir
from src.data.rates.fred_reader import get_fred_series
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MONTH_CODES = "FGHJKMNQUVXZ"

# Approximate duration in years for each bond root (used for bond carry).
DURATION_BY_ROOT: dict[str, float] = {
    "ZT": 2.0,
    "ZF": 5.0,
    "ZN": 9.0,
    "TN": 9.0,
    "ZB": 17.0,
    "UB": 22.0,
    "2YY": 2.0,
    "5YY": 5.0,
    "10Y": 9.0,
    "30Y": 22.0,
}

# Micro Yield roots: close IS yield in % directly.
MICRO_YIELD_ROOTS = {"2YY", "5YY", "10Y", "30Y"}

# CMT yield FRED series for price-traded bond futures (ZT/ZF/ZN/TN/ZB/UB).
_BOND_CMT_TENOR = {"ZT": "DGS2", "ZF": "DGS5", "ZN": "DGS10",
                   "TN": "DGS10", "ZB": "DGS30", "UB": "DGS30"}
_BOND_FUNDING_SERIES = "DFF"


def _is_outright(symbol: str, root: str) -> bool:
    if "-" in symbol or " " in symbol:
        return False
    if not symbol.startswith(root):
        return False
    suffix = symbol[len(root):]
    if len(suffix) < 2:
        return False
    return suffix[0] in _MONTH_CODES and suffix[1:].isdigit()


class CarryCalculator:
    """Computes per-asset-class carry signals."""

    def _find_front_second_close(
        self, root: str, d: date,
    ) -> tuple[str, float, str, float]:
        """Return (front_symbol, front_close, second_symbol, second_close)
        ranked by daily volume on date d.

        Raises ValueError if fewer than 2 outright contracts have data on d.
        """
        pcm = (
            per_contract_1min_dir()
            / f"year={d.year}" / f"month={d.month}" / "data.parquet"
        )
        if not pcm.exists():
            raise ValueError(f"no per-contract data for {d}: missing {pcm}")
        df = pl.read_parquet(pcm)
        df = df.filter(pl.col("symbol").map_elements(
            lambda s: _is_outright(s, root), return_dtype=pl.Boolean,
        ))
        df = df.filter(pl.col("timestamp").dt.date() == d)
        if df.is_empty():
            raise ValueError(f"no outright {root} data on {d}")
        daily = df.group_by("symbol").agg([
            pl.col("close").last().alias("c"),
            pl.col("volume").sum().alias("v"),
        ]).sort("v", descending=True)
        if daily.shape[0] < 2:
            raise ValueError(
                f"need 2 outright contracts on {d}, found {daily.shape[0]}"
            )
        rows = daily.head(2).to_dicts()
        return (rows[0]["symbol"], rows[0]["c"],
                rows[1]["symbol"], rows[1]["c"])

    def _months_between(self, front_sym: str, second_sym: str, root: str) -> int:
        """Number of calendar months between front and second contract.

        Symbol format: <root><MONTH_CODE><YEAR_DIGIT>. Year is the last digit
        of the year (CME convention). Returns (second_month - front_month).

        Example: front=Q4 (Aug), second=M4 (Jun) -> -2 months (same year, 2 months earlier).
        """
        fs = front_sym[len(root):]
        ss = second_sym[len(root):]
        f_month = _MONTH_CODES.index(fs[0]) + 1
        s_month = _MONTH_CODES.index(ss[0]) + 1
        f_year = int(fs[1:])
        s_year = int(ss[1:])

        # Adjust for year wrapping: if s_year < f_year, assume second is in next decade
        if s_year < f_year:
            s_year += 10

        return (s_year - f_year) * 12 + (s_month - f_month)

    def compute(self, root: str, asset_class: str, d: date) -> float:
        """Compute carry signal for `root` on date `d` per asset class."""
        front_sym, front_c, second_sym, second_c = self._find_front_second_close(root, d)
        months = self._months_between(front_sym, second_sym, root)
        if months == 0:
            raise ValueError(f"front and second contracts in same month: {front_sym}, {second_sym}")
        days_to_second = abs(months) * 30.0

        if asset_class == "commodity":
            return (second_c - front_c) / front_c * (365.0 / days_to_second)
        if asset_class == "crypto":
            # CME crypto futures: annualized calendar roll yield (same convention
            # as commodity). Short history + regime risk -- flagged in the report.
            return (second_c - front_c) / front_c * (365.0 / days_to_second)
        if asset_class == "equity_index":
            return (front_c - second_c) / second_c * (365.0 / days_to_second)
        if asset_class == "fx":
            return (second_c - front_c) / front_c * (365.0 / days_to_second)
        if asset_class == "bond":
            duration = DURATION_BY_ROOT.get(root)
            if duration is None:
                raise ValueError(f"no duration table entry for {root}")
            if root in MICRO_YIELD_ROOTS:
                # Front close is the yield in %; funding rate is SOFR in %
                front_yield = front_c
                funding = derive_sofr(d)
                return duration * (front_yield - funding) / 100.0
            # ZT/ZF/ZN/TN/ZB/UB: price-traded bond futures. Yield is not in the
            # futures price; use the FRED constant-maturity yield for the tenor
            # minus the funding rate, scaled by duration (Carver bond carry).
            cmt_yield = get_fred_series(_BOND_CMT_TENOR[root], d)
            funding = get_fred_series(_BOND_FUNDING_SERIES, d)
            return duration * (cmt_yield - funding) / 100.0
        raise ValueError(f"unknown asset_class: {asset_class}")

    def compute_history(
        self, root: str, asset_class: str, start: date, end: date,
    ) -> pl.DataFrame:
        """Compute carry per trading day between start and end (inclusive).

        Skips dates where the underlying data lookup fails (weekends, holidays,
        missing data). Returns DataFrame with columns [date, carry].
        """
        pcm_dir = per_contract_1min_dir()
        if not pcm_dir.exists():
            raise FileNotFoundError(
                f"per-contract futures store missing: {pcm_dir} "
                f"-- carry cannot be computed"
            )

        records: list[dict] = []
        d = start
        while d <= end:
            try:
                c = self.compute(root, asset_class, d)
                records.append({"date": d, "carry": c})
            except (ValueError, FileNotFoundError, NotImplementedError) as e:
                logger.debug(f"skip {root} carry on {d}: {e}")
            d += timedelta(days=1)
        if not records:
            return pl.DataFrame(schema={"date": pl.Date, "carry": pl.Float64})
        return pl.DataFrame(records)
