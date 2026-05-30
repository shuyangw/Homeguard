"""Canonical asset-class data path layout.

Single source of truth for where each dataset lives on disk under
`get_local_storage_dir()`. Used by:
  - Data acquisition plugins (storage_subdir)
  - Migration script (old -> new mapping)
  - Validators, backtest engines, downstream readers

Layout: <asset_class>/<source>/<frequency>/

  equities/
    iex/1min/                       (Alpaca IEX, raw)
    iex/1min_by_date/
    sip_raw/1min/                   (Alpaca SIP, raw)
    sip_split/1min/                 (Alpaca SIP, split-adjusted)
  crypto/
    alpaca/1min/
    alpaca/1hour/
    alpaca/1day/
    alpaca/archive/1min/
  futures/
    databento/1min/
    databento/mbp1/
    databento/trades/
    databento/staging/
    databento/1min_oi_roll/
    databento/per_contract_1min/
    databento/per_contract_daily/
    databento/options_1min/
    databento/statistics/
    databento/status/
    definitions/
  fx/
    massive/1min/
    massive/quotes_minute_aggregated/
    massive/quotes_raw/
    polygon/1min_backfill/
  news/
    alpaca/
  options/
    thetadata/_logs/
    thetadata/chains/
    thetadata/gex_daily/
    thetadata/options_combined/
"""

from pathlib import Path

from src.settings.settings import get_local_storage_dir


# Canonical subdirs (relative to local_storage_dir)
EQUITIES_IEX_1MIN = "equities/iex/1min"
EQUITIES_IEX_1MIN_BY_DATE = "equities/iex/1min_by_date"
EQUITIES_SIP_RAW_1MIN = "equities/sip_raw/1min"
EQUITIES_SIP_SPLIT_1MIN = "equities/sip_split/1min"

CRYPTO_ALPACA_1MIN = "crypto/alpaca/1min"
CRYPTO_ALPACA_1HOUR = "crypto/alpaca/1hour"
CRYPTO_ALPACA_1DAY = "crypto/alpaca/1day"
CRYPTO_ALPACA_ARCHIVE_1MIN = "crypto/alpaca/archive/1min"

FUTURES_DATABENTO_1MIN = "futures/databento/1min"
FUTURES_DATABENTO_MBP1 = "futures/databento/mbp1"
FUTURES_DATABENTO_TRADES = "futures/databento/trades"
FUTURES_DATABENTO_STAGING = "futures/databento/staging"
FUTURES_DATABENTO_1MIN_OI_ROLL = "futures/databento/1min_oi_roll"
FUTURES_DATABENTO_PER_CONTRACT_1MIN = "futures/databento/per_contract_1min"
FUTURES_DATABENTO_PER_CONTRACT_DAILY = "futures/databento/per_contract_daily"
FUTURES_DATABENTO_OPTIONS_1MIN = "futures/databento/options_1min"
FUTURES_DATABENTO_STATISTICS = "futures/databento/statistics"
FUTURES_DATABENTO_STATUS = "futures/databento/status"
FUTURES_DEFINITIONS = "futures/definitions"

FX_MASSIVE_1MIN = "fx/massive/1min"
FX_MASSIVE_QUOTES_MINUTE_AGGREGATED = "fx/massive/quotes_minute_aggregated"
FX_MASSIVE_QUOTES_RAW = "fx/massive/quotes_raw"
FX_POLYGON_1MIN_BACKFILL = "fx/polygon/1min_backfill"

NEWS_ALPACA = "news/alpaca"

OPTIONS_THETADATA_LOGS = "options/thetadata/_logs"
OPTIONS_THETADATA_CHAINS = "options/thetadata/chains"
OPTIONS_THETADATA_GEX_DAILY = "options/thetadata/gex_daily"
OPTIONS_THETADATA_COMBINED = "options/thetadata/options_combined"


# Mapping of OLD flat subdir -> NEW asset-class subdir.
# Used by migration script + backward-compat reads if needed.
LEGACY_TO_CANONICAL: dict[str, str] = {
    "equities_1min": EQUITIES_IEX_1MIN,
    "equities_1min_by_date": EQUITIES_IEX_1MIN_BY_DATE,
    "equities_1min_sip_raw": EQUITIES_SIP_RAW_1MIN,
    "equities_1min_sip_split": EQUITIES_SIP_SPLIT_1MIN,

    "crypto_1min": CRYPTO_ALPACA_1MIN,
    "crypto_1hour": CRYPTO_ALPACA_1HOUR,
    "crypto_1day": CRYPTO_ALPACA_1DAY,
    "crypto_1min_alpaca_archive": CRYPTO_ALPACA_ARCHIVE_1MIN,

    "futures_1min": FUTURES_DATABENTO_1MIN,
    "futures_mbp1": FUTURES_DATABENTO_MBP1,
    "futures_trades": FUTURES_DATABENTO_TRADES,
    "futures_dbn_staging": FUTURES_DATABENTO_STAGING,
    "futures_1min_oi_roll": FUTURES_DATABENTO_1MIN_OI_ROLL,
    "futures_per_contract_1min": FUTURES_DATABENTO_PER_CONTRACT_1MIN,
    "futures_per_contract_daily": FUTURES_DATABENTO_PER_CONTRACT_DAILY,
    "futures_options_1min": FUTURES_DATABENTO_OPTIONS_1MIN,
    "futures_statistics": FUTURES_DATABENTO_STATISTICS,
    "futures_status": FUTURES_DATABENTO_STATUS,
    "futures_definitions": FUTURES_DEFINITIONS,

    "fx_1min": FX_MASSIVE_1MIN,
    "fx_quotes_minute_aggregated": FX_MASSIVE_QUOTES_MINUTE_AGGREGATED,
    "fx_quotes_raw": FX_MASSIVE_QUOTES_RAW,
    "fx_1min_polygon_backfill": FX_POLYGON_1MIN_BACKFILL,

    "news": NEWS_ALPACA,
}


# Manifest filename mapping: nested subdir is flattened with underscores.
def manifest_filename(subdir: str) -> str:
    """Convert a canonical subdir like 'equities/iex/1min' to 'equities_iex_1min.json'."""
    return subdir.replace("/", "_") + ".json"


# Convenience accessors for full paths.
def _path(subdir: str) -> Path:
    return get_local_storage_dir() / subdir


def get_equities_iex_1min_dir() -> Path:
    return _path(EQUITIES_IEX_1MIN)


def get_equities_sip_raw_1min_dir() -> Path:
    return _path(EQUITIES_SIP_RAW_1MIN)


def get_equities_sip_split_1min_dir() -> Path:
    return _path(EQUITIES_SIP_SPLIT_1MIN)


def get_crypto_alpaca_1min_dir() -> Path:
    return _path(CRYPTO_ALPACA_1MIN)


def get_futures_databento_1min_dir() -> Path:
    return _path(FUTURES_DATABENTO_1MIN)


def get_futures_databento_mbp1_dir() -> Path:
    return _path(FUTURES_DATABENTO_MBP1)


def get_futures_databento_trades_dir() -> Path:
    return _path(FUTURES_DATABENTO_TRADES)


def get_fx_massive_1min_dir() -> Path:
    return _path(FX_MASSIVE_1MIN)


def get_news_alpaca_dir() -> Path:
    return _path(NEWS_ALPACA)


# Generic accessor with explicit asset/source/freq
def get_data_dir(subdir: str) -> Path:
    """Resolve any canonical subdir constant to a Path under local_storage_dir."""
    return _path(subdir)
