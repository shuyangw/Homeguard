"""Databento GLBX.MDP3 futures plugin - downloads OHLCV-1m bars (or raw trades)."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import polars as pl

from src.data.acquisition.aggregators import trades_to_ohlcv_1m
from src.data.acquisition.base import BaseDownloader, DownloadResult
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA, FUTURES_TRADES_SCHEMA
from src.settings import FUTURES_DATABENTO_1MIN, FUTURES_DATABENTO_MBP1, FUTURES_DATABENTO_TRADES
from src.utils.logger import get_logger

try:
    import databento as db
except ImportError:
    db = None  # type: ignore[assignment]

logger = get_logger(__name__)

DEFAULT_FUTURES_UNIVERSE = ["ES", "NQ", "CL", "GC", "ZN", "6E", "ZC", "YM", "RTY"]

# Full Databento bulk-pull universe per docs/strategies/research/DATABENTO_BULK_PULL_PLAN.md
# Section A.1-A.8 continuous OHLCV-1m. Includes GC.n.0 (open-interest roll diagnostic).
BULK_PULL_UNIVERSE_V = [
    # Equity index full size
    "ES", "NQ", "YM", "RTY",
    # Equity index micros
    "MES", "MNQ", "M2K", "MYM",
    # Energy
    "CL", "NG", "HO", "RB", "BZ", "MCL", "MNG",
    # Metals (PRIMARY FIX vs broken c.0)
    "GC", "SI", "HG", "PL", "MGC", "SIL",
    # Rates / bonds
    "ZT", "ZF", "ZN", "TN", "ZB", "UB",
    "SR3", "SR1",
    "10Y", "30Y", "5YY", "2YY",
    # FX
    "6E", "6J", "6B", "6A", "6C", "6S", "6N", "6M",
    # Agriculture (SECONDARY FIX vs broken c.0)
    "ZC", "ZS", "ZW", "KE", "ZL", "ZM", "LE", "HE",
    # Crypto
    "BTC", "MBT", "ETH", "MET",
]

DATASET = "GLBX.MDP3"
DEFAULT_START = "2020-01-01"
BULK_PULL_START = "2010-06-06"  # GLBX.MDP3 dataset floor

VALID_ROLL_RULES = ("v", "n", "c")  # volume, open-interest, calendar

MBP1_CANONICAL_COLUMNS = ["ts_event", "bid_px", "ask_px", "bid_sz", "ask_sz"]


class DatabentoFuturesPlugin(BaseDownloader):
    """Downloads futures data from Databento GLBX.MDP3.

    Supports two modes:
    - schema="ohlcv-1m" (default): Downloads pre-aggregated 1-minute bars directly.
      Much cheaper ($51 vs $2,270 for 9 symbols over 5 years).
    - schema="trades": Downloads raw trades, stores them, then reconstructs OHLCV-1m.
      Two-stage storage for trades mode:
        futures_trades/symbol=ES/year=Y/month=M/data.parquet (raw trades)
        futures_1min/symbol=ES/year=Y/month=M/data.parquet (reconstructed OHLCV)
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        schema: str = "ohlcv-1m",
        roll_rule: str = "v",
        storage_subdir: Optional[str] = None,
        **kwargs,
    ):
        self._api_key = os.getenv("DATABENTO_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DATABENTO_API_KEY not set in environment. "
                "Add it to your .env file."
            )
        if db is None:
            raise ImportError(
                "databento package not installed. Run: pip install databento"
            )
        if schema not in ("ohlcv-1m", "trades", "mbp-1"):
            raise ValueError(f"schema must be 'ohlcv-1m', 'trades', or 'mbp-1', got '{schema}'")
        if roll_rule not in VALID_ROLL_RULES:
            raise ValueError(
                f"roll_rule must be one of {VALID_ROLL_RULES}, got '{roll_rule}'"
            )
        # Set _schema, _roll_rule before super().__init__ because it calls
        # _get_storage_subdir()
        self._schema = schema
        self._roll_rule = roll_rule
        self._storage_subdir_override = storage_subdir
        super().__init__(output_dir=output_dir, **kwargs)

    def _create_client(self) -> Any:
        return db.Historical(self._api_key)

    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        api_symbol = self._to_api_symbol(symbol)
        logger.info(
            f"Fetching {self._schema} for {api_symbol} from {start} to {end}"
        )

        data = client.timeseries.get_range(
            dataset=DATASET,
            schema=self._schema,
            stype_in="continuous",
            symbols=[api_symbol],
            start=start,
            end=end,
        )
        df = data.to_df()

        if df.empty:
            schema_cols = self._get_schema()
            return pd.DataFrame(columns=schema_cols)

        if self._schema == "ohlcv-1m":
            return self._normalize_ohlcv(df)
        elif self._schema == "mbp-1":
            return self._normalize_mbp1(df)
        else:
            return self._normalize_trades(df)

    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Databento ohlcv-1m DataFrame to canonical schema.

        Timestamps are cast to [us, UTC] per the documented canonical schema
        (.claude/data_handling.md). pandas defaults to [ns]; explicit cast
        prevents dtype drift across the dataset.
        """
        # Databento sets ts_event as the index by default
        ts_series = (
            df.index if df.index.name == "ts_event" else df["ts_event"]
        )
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_series, utc=True).astype(
                    "datetime64[us, UTC]"
                ),
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "volume": df["volume"].astype(float),
                "trade_count": (
                    df["trade_count"].astype(float)
                    if "trade_count" in df.columns
                    else 0.0
                ),
                "vwap": (
                    df["vwap"].astype(float)
                    if "vwap" in df.columns
                    else float("nan")
                ),
            }
        )

    def _normalize_mbp1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Databento mbp-1 DataFrame to canonical MBP1 schema.

        Databento mbp-1 includes many columns; we keep only top-of-book snapshot.
        Timestamps cast to [us, UTC].
        """
        ts_series = (
            df.index if df.index.name == "ts_event" else df["ts_event"]
        )
        return pd.DataFrame({
            "ts_event": pd.to_datetime(ts_series, utc=True).astype(
                "datetime64[us, UTC]"
            ),
            "bid_px": df["bid_px_00"].astype(float) if "bid_px_00" in df.columns else df["bid_px"].astype(float),
            "ask_px": df["ask_px_00"].astype(float) if "ask_px_00" in df.columns else df["ask_px"].astype(float),
            "bid_sz": df["bid_sz_00"].astype(int) if "bid_sz_00" in df.columns else df["bid_sz"].astype(int),
            "ask_sz": df["ask_sz_00"].astype(int) if "ask_sz_00" in df.columns else df["ask_sz"].astype(int),
        })

    def _normalize_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Databento trades DataFrame to our trades schema."""
        ts_series = (
            df.index if df.index.name == "ts_event" else df["ts_event"]
        )
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_series, utc=True).astype(
                    "datetime64[us, UTC]"
                ),
                "price": df["price"].astype(float),
                "size": df["size"].astype(float),
            }
        )

    def _get_schema(self) -> list[str]:
        if self._schema == "ohlcv-1m":
            return CANONICAL_OHLCV_SCHEMA
        if self._schema == "mbp-1":
            return MBP1_CANONICAL_COLUMNS
        return FUTURES_TRADES_SCHEMA

    def _save_partitioned(self, df: pd.DataFrame, symbol: str) -> int:
        """Override to handle mbp-1 schema which uses ts_event (not timestamp)."""
        if self._schema != "mbp-1":
            return super()._save_partitioned(df, symbol)

        output_dir = self._get_output_dir()
        fs_symbol = self._normalize_symbol(symbol)

        df = df.copy()
        ts_col = pd.to_datetime(df["ts_event"])
        df["_year"] = ts_col.dt.year
        df["_month"] = ts_col.dt.month

        rows_saved = 0
        for (year, month), group in df.groupby(["_year", "_month"]):
            partition_dir = (
                output_dir
                / f"symbol={fs_symbol}"
                / f"year={year}"
                / f"month={month}"
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            data_to_save = group.drop(columns=["_year", "_month"])
            data_to_save.to_parquet(partition_dir / "data.parquet", index=False)
            rows_saved += len(data_to_save)
        return rows_saved

    def _get_storage_subdir(self) -> str:
        if self._storage_subdir_override:
            return self._storage_subdir_override
        if self._schema == "ohlcv-1m":
            return FUTURES_DATABENTO_1MIN
        if self._schema == "mbp-1":
            return FUTURES_DATABENTO_MBP1
        return FUTURES_DATABENTO_TRADES

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol

    def _to_api_symbol(self, symbol: str) -> str:
        """Convert short symbol (ES) to Databento continuous format (ES.{rule}.0).

        Pass through fully-qualified continuous symbols (e.g. ES.v.0, GC.n.0) unchanged.
        """
        if any(f".{r}." in symbol for r in VALID_ROLL_RULES):
            return symbol
        return f"{symbol}.{self._roll_rule}.0"

    def download(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = DEFAULT_START,
        end_date: Optional[str] = None,
        skip_existing: bool = False,
    ) -> DownloadResult:
        """Download futures data.

        For trades schema: downloads raw trades, then reconstructs OHLCV-1m.
        For ohlcv-1m schema: downloads pre-aggregated bars directly.

        Databento free tier has ~24h data delay, so default end date is T-2
        to avoid dataset_unavailable_range errors.
        """
        symbols = symbols or DEFAULT_FUTURES_UNIVERSE
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

        result = super().download(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            skip_existing=skip_existing,
        )

        # For trades mode, reconstruct OHLCV-1m after raw trade storage
        if self._schema == "trades":
            self._reconstruct_ohlcv(symbols)

        return result

    def _reconstruct_ohlcv(self, symbols: List[str]) -> None:
        """Read raw trades and write OHLCV-1m parquet files."""
        trades_dir = self.base_output_dir / FUTURES_DATABENTO_TRADES
        ohlcv_dir = self.base_output_dir / FUTURES_DATABENTO_1MIN

        for symbol in symbols:
            fs_symbol = self._normalize_symbol(symbol)
            symbol_trades_dir = trades_dir / f"symbol={fs_symbol}"
            if not symbol_trades_dir.exists():
                continue

            for parquet_file in symbol_trades_dir.glob("**/data.parquet"):
                # Extract year/month from path
                parts = parquet_file.parts
                year_part = [p for p in parts if p.startswith("year=")]
                month_part = [p for p in parts if p.startswith("month=")]
                if not year_part or not month_part:
                    continue

                year = year_part[0].replace("year=", "")
                month = month_part[0].replace("month=", "")

                # Read trades and reconstruct
                trades_df = pd.read_parquet(parquet_file)
                if trades_df.empty:
                    continue

                ohlcv_df = trades_to_ohlcv_1m(trades_df)
                if ohlcv_df.empty:
                    continue

                # Write to futures_1min with same partition structure
                out_dir = (
                    ohlcv_dir
                    / f"symbol={fs_symbol}"
                    / f"year={year}"
                    / f"month={month}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                ohlcv_df.to_parquet(out_dir / "data.parquet", index=False)

            logger.info(f"Reconstructed OHLCV-1m for {symbol}")

    def _is_supported_schema(self, schema: str) -> bool:
        return schema in ("ohlcv-1m", "trades", "mbp-1")

    def _write_mbp1_partition(self, df, symbol: str, year: int,
                              month: int, root=None) -> Path:
        """Write MBP-1 data to futures_mbp1/ partition tree."""
        from src.settings import get_local_storage_dir
        root = root if root is not None else get_local_storage_dir()
        out_dir = root / "futures_mbp1" / f"symbol={symbol}" / f"year={year}" / f"month={month}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "data.parquet"

        pl_df = pl.from_pandas(df).select(MBP1_CANONICAL_COLUMNS).with_columns(
            pl.col("ts_event").cast(pl.Datetime(time_unit="ns", time_zone="UTC")),
        ).sort("ts_event")

        tmp = out.with_suffix(out.suffix + ".tmp")
        pl_df.write_parquet(tmp)
        os.replace(tmp, out)
        return out
