"""Options data loader for Hive-partitioned parquet files.

Reads 1-minute options data stored as:
    data_dir/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet
"""

from datetime import date, time, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger()

COLUMN_RENAME_MAP = {
    "bid_close": "bid",
    "ask_close": "ask",
    "gamma_eod": "gamma",
    "open_interest_eod": "open_interest",
    "underlying_px": "underlying_price",
}

RIGHT_TO_OPTION_TYPE = {"PUT": "P", "CALL": "C"}


class OptionsDataLoader:
    """Loads options chain data from Hive-partitioned parquet files.

    Data layout: data_dir/root={SYMBOL}/year={YYYY}/month={MM}/data.parquet
    """

    def __init__(self, data_dir: Path):
        self._data_dir = Path(data_dir)
        self._month_cache: Dict[tuple, pd.DataFrame] = {}

    def get_available_symbols(self) -> List[str]:
        symbols = []
        if not self._data_dir.exists():
            return symbols
        for entry in sorted(self._data_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("root="):
                symbols.append(entry.name.split("=", 1)[1])
        return symbols

    def get_eod_chain(self, symbol: str, target_date: date) -> pd.DataFrame:
        return self.get_chain_at_time(symbol, target_date, time(16, 0))

    def get_chain_at_time(
        self, symbol: str, target_date: date, target_time: time
    ) -> pd.DataFrame:
        month_df = self._load_month(
            symbol, target_date.year, target_date.month, target_time=target_time
        )
        if month_df.empty:
            return pd.DataFrame()

        mask = (month_df["date"] == target_date) & (month_df["time"] == target_time)
        result = month_df.loc[mask].copy()
        result.reset_index(drop=True, inplace=True)
        return result

    def get_date_range(self, symbol: str) -> Tuple[Optional[date], Optional[date]]:
        """Get earliest and latest months available for a symbol using directory listing only."""
        symbol_dir = self._data_dir / f"root={symbol}"
        if not symbol_dir.exists():
            return (None, None)

        year_months: List[Tuple[int, int]] = []
        for year_dir in symbol_dir.iterdir():
            if not (year_dir.is_dir() and year_dir.name.startswith("year=")):
                continue
            year = int(year_dir.name.split("=", 1)[1])
            for month_dir in year_dir.iterdir():
                if not (month_dir.is_dir() and month_dir.name.startswith("month=")):
                    continue
                month = int(month_dir.name.split("=", 1)[1])
                data_file = month_dir / "data.parquet"
                if data_file.exists():
                    year_months.append((year, month))

        if not year_months:
            return (None, None)

        year_months.sort()
        first_y, first_m = year_months[0]
        last_y, last_m = year_months[-1]
        return (date(first_y, first_m, 1), date(last_y, last_m, 1))

    def clear_cache(self):
        self._month_cache.clear()

    def _load_month(
        self,
        symbol: str,
        year: int,
        month: int,
        target_time: Optional[time] = None,
    ) -> pd.DataFrame:
        cache_key = (symbol, year, month, target_time)
        if cache_key in self._month_cache:
            return self._month_cache[cache_key]

        parquet_path = (
            self._data_dir
            / f"root={symbol}"
            / f"year={year}"
            / f"month={month:02d}"
            / "data.parquet"
        )

        if not parquet_path.exists():
            empty = pd.DataFrame()
            self._month_cache[cache_key] = empty
            return empty

        logger.debug(f"Loading options data: {parquet_path}")

        if target_time is not None:
            # Memory-efficient: read in batches, keep only matching timestamps.
            # This avoids OOM on large chains (NVDA ~10M rows, SPX ~16M rows).
            time_suffix = f"T{target_time.hour:02d}:{target_time.minute:02d}:00"
            pf = pq.ParquetFile(parquet_path)
            filtered_batches = []
            for batch in pf.iter_batches(batch_size=200_000):
                ts = batch.column("timestamp").cast(pa.string())
                mask = pc.ends_with(ts, time_suffix)
                filtered = batch.filter(mask)
                if filtered.num_rows > 0:
                    filtered_batches.append(filtered)

            if filtered_batches:
                table = pa.concat_tables(
                    [pa.Table.from_batches([b]) for b in filtered_batches]
                )
                df = table.to_pandas()
            else:
                df = pd.DataFrame()
        else:
            df = pd.read_parquet(parquet_path)

        if df.empty:
            self._month_cache[cache_key] = df
            return df

        df = self._transform(df)

        self._month_cache[cache_key] = df
        return df

    @staticmethod
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=COLUMN_RENAME_MAP)

        df["option_type"] = df["right"].map(RIGHT_TO_OPTION_TYPE)
        df.drop(columns=["right"], inplace=True)

        df["datetime"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["datetime"].dt.date
        df["time"] = df["datetime"].dt.time
        df["expiry"] = pd.to_datetime(df["expiration"]).dt.date
        df["days_to_expiry"] = (
            pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"])
        ).dt.days
        df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

        return df
