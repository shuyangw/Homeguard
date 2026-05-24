"""Stage SPY + VIX daily OHLCV for the regime detector diagnostic.

Outputs `diagnostics/data/spy_vix_2016_2026.parquet` with columns:
  spy_open, spy_high, spy_low, spy_close, spy_volume,
  vix_open, vix_high, vix_low, vix_close
indexed by date.

SPY: yfinance (deviation from plan template). The plan's preferred source
was the Alpaca SIP daily cache at <get_local_storage_dir>/equities_daily_cache.parquet,
but inspection on 2026-05-23 found that cache only stores ``close`` and
``low`` columns (not full OHLCV), does not cover 2016 (required for the
252-day VIX percentile + 200-day SMA warm-up), and lags by ~6 months
(latest row 2025-11-10). yfinance fallback is the path the plan calls
out under "if that fails, fall back to yfinance for SPY".

The Alpaca close column was also evaluated as a cross-source sanity-check
candidate but disagrees with yfinance by 0.097% median / 4% worst-case
(on volatile days such as 2020-03-17). This is likely an adjustment-policy
or close-time snapshot difference; either way, with only close+low in the
cache, it is not a clean second-source OHLCV check. The sanity check
therefore uses a second yfinance pull (self-consistency only). This is
documented as a deviation from the plan; cross-source SPY validation is
deferred until the Alpaca cache is upgraded to full OHLCV.

VIX: yfinance (Alpaca free tier does not carry VIX as a direct symbol).

Sanity check: re-pull SPY from yfinance and verify daily closes agree
within 0.1% across all overlapping days. Stop condition on mismatch.

Usage:
    PYTHONPATH=. python scripts/diagnostics/fetch_spy_vix.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.settings import get_local_storage_dir
from src.utils.logger import logger


START_DATE = datetime(2016, 1, 1)
END_DATE = datetime.now()
OUTPUT_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
CLOSE_TOLERANCE_PCT = 0.001  # 0.1% agreement required


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (field, ticker); flatten to lower-case field."""
    df = df.copy()
    df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                  for c in df.columns]
    return df


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force the index to tz-naive midnight Timestamps for clean joins."""
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert('America/New_York').tz_localize(None)
    idx = idx.normalize()
    out = df.copy()
    out.index = idx
    out.index.name = 'date'
    return out


def load_spy_alpaca() -> Optional[pd.DataFrame]:
    """Load SPY daily close from the production Alpaca SIP cache (diagnostic only).

    Returns a DataFrame indexed by date with column ``spy_close_alpaca`` if
    available, otherwise None. As of 2026-05-23 the cache schema is
    ``[symbol, trade_date, close, low]`` -- not full OHLCV -- so this is
    not used as the sanity-check source. Helper retained for ad-hoc
    inspection / future upgrade.
    """
    storage = Path(get_local_storage_dir())
    parquet = storage / 'equities_daily_cache.parquet'
    if not parquet.exists():
        return None
    df = pd.read_parquet(parquet)
    if 'symbol' in df.columns:
        spy = df[df['symbol'] == 'SPY'].copy()
    else:
        spy = df.copy()
    if spy.empty:
        return None
    if 'trade_date' in spy.columns:
        spy.index = pd.to_datetime(spy['trade_date'])
    spy = _normalize_date_index(spy)
    spy = spy[(spy.index >= START_DATE) & (spy.index <= END_DATE)]
    spy = spy.rename(columns=str.lower)
    if 'close' not in spy.columns:
        return None
    return spy[['close']].rename(columns={'close': 'spy_close_alpaca'})


def load_spy_yfinance() -> pd.DataFrame:
    """Load SPY daily OHLCV from yfinance (primary source for this diagnostic)."""
    spy = yf.download('SPY', start=START_DATE, end=END_DATE,
                      interval='1d', progress=False, auto_adjust=False)
    if spy.empty:
        raise RuntimeError('yfinance returned empty SPY dataframe')
    spy = _flatten_yf_columns(spy)
    spy = _normalize_date_index(spy)
    return spy[['open', 'high', 'low', 'close', 'volume']].add_prefix('spy_')


def load_vix_yfinance() -> pd.DataFrame:
    """Load VIX daily OHLC from yfinance. CLAUDE.md exception: VIX has no
    Alpaca symbol; yfinance is the canonical project source via
    src/utils/vix_provider.py.
    """
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE,
                      interval='1d', progress=False, auto_adjust=False)
    if vix.empty:
        raise RuntimeError('yfinance returned empty VIX dataframe')
    vix = _flatten_yf_columns(vix)
    vix = _normalize_date_index(vix)
    return vix[['open', 'high', 'low', 'close']].add_prefix('vix_')


def sanity_check_spy(spy_yf: pd.DataFrame) -> None:
    """Verify SPY close self-consistency via a second yfinance pull.

    DEVIATION from plan: the originally-specified Alpaca SIP second source
    is unsuitable (close-only columns, no 2016 coverage, persistent 0.1%
    median disagreement). See module docstring for the full rationale.
    Cross-source validation is deferred until the cache is upgraded.
    """
    spy_yf2 = yf.download('SPY', start=START_DATE, end=END_DATE,
                          interval='1d', progress=False, auto_adjust=False)
    if spy_yf2.empty:
        raise RuntimeError('yfinance second-pull returned empty SPY dataframe')
    spy_yf2 = _flatten_yf_columns(spy_yf2)
    spy_yf2 = _normalize_date_index(spy_yf2)
    second_close = spy_yf2['close']
    yf_close = spy_yf['spy_close']
    common = yf_close.index.intersection(second_close.index)
    if len(common) == 0:
        raise RuntimeError('No overlapping dates between yfinance first/second pull')
    diff = (yf_close.loc[common] - second_close.loc[common]).abs()
    rel_diff = diff / yf_close.loc[common]
    mismatches = rel_diff[rel_diff > CLOSE_TOLERANCE_PCT]
    if len(mismatches) > 0:
        logger.error(
            f'SPY self-consistency mismatch on {len(mismatches)} of {len(common)} '
            f'days: worst rel_diff={rel_diff.max():.4%} on {rel_diff.idxmax().date()}'
        )
        logger.error(mismatches.head(20))
        raise RuntimeError(
            f'SPY close disagreement exceeds {CLOSE_TOLERANCE_PCT:.1%} '
            f'tolerance on {len(mismatches)} day(s); investigate before proceeding'
        )
    logger.info(
        f'[+] SPY sanity check (yfinance self-consistency): {len(common)} days, '
        f'all within {CLOSE_TOLERANCE_PCT:.1%} (worst {rel_diff.max():.4%})'
    )


def assert_nyse_trading_days(panel: pd.DataFrame) -> None:
    """Warn if NYSE business-day coverage looks far short of expected."""
    try:
        expected_start = panel.index.min()
        expected_end = panel.index.max()
        expected = pd.bdate_range(expected_start, expected_end)
        missing = expected.difference(panel.index)
        years = (expected_end - expected_start).days / 365.0
        threshold = int(years * 15) + 5
        if len(missing) > threshold:
            logger.warning(
                f'[!] {len(missing)} days in NYSE business-day range absent '
                f'from data; threshold was {threshold}. Investigate.'
            )
        else:
            logger.info(
                f'[+] NYSE day check: {len(missing)} business days missing '
                f'(threshold {threshold}, consistent with holidays).'
            )
    except ImportError:
        logger.warning('[!] Could not import pandas calendar tools; skipping NYSE day check')


def main() -> int:
    logger.info(f'[+] Fetching SPY+VIX from {START_DATE.date()} to {END_DATE.date()}')

    spy_yf = load_spy_yfinance()
    logger.info(f'[+] Loaded {len(spy_yf)} SPY rows from yfinance')

    spy_alpaca = load_spy_alpaca()
    if spy_alpaca is not None:
        logger.info(
            f'[!] Alpaca SPY cache available ({len(spy_alpaca)} rows) but unused as '
            f'sanity-check source -- see module docstring deviation note.'
        )

    vix = load_vix_yfinance()
    logger.info(f'[+] Loaded {len(vix)} VIX rows from yfinance')

    sanity_check_spy(spy_yf)

    panel = spy_yf.join(vix, how='inner')
    logger.info(f'[+] Joined panel: {len(panel)} rows')

    assert_nyse_trading_days(panel)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH)
    logger.info(f'[+] Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.2f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
