"""Day-by-day replay of MarketRegimeDetector across 2017-2026.

Reads diagnostics/data/spy_vix_2016_2026.parquet, calls
MarketRegimeDetector.classify_regime for each trading day in 2017-2026,
and emits a Parquet with detector outputs + intermediate values +
parametrized alternatives, suitable for ad-hoc analysis.

Usage:
    PYTHONPATH=. python scripts/diagnostics/regime_detector_replay.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.strategies.advanced.market_regime_detector import (
    DataInsufficientError, MarketRegimeDetector,
)
from src.utils.logger import logger


INPUT_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
OUTPUT_PATH = Path('diagnostics/regime/v0/labels.parquet')
REPLAY_START = datetime(2017, 1, 1)
WARMUP_DAYS = 400  # 252 VIX pct lookback + 200 SMA + buffer
ALT_LOOKBACKS = (63, 126, 252, 504)


def compute_alternative_vix_percentiles(
    panel: pd.DataFrame, t: pd.Timestamp,
) -> Dict[int, float]:
    """For each lookback window, compute the percentile of VIX at t against
    the prior `window` days of VIX closes.
    """
    out = {}
    vix_close = panel['vix_close']
    current_vix = vix_close.loc[t]
    history = vix_close.loc[vix_close.index < t]
    for window in ALT_LOOKBACKS:
        sample = history.iloc[-window:]
        if len(sample) < window // 2:
            out[window] = float('nan')
            continue
        pct = float((sample < current_vix).sum() / len(sample) * 100.0)
        out[window] = pct
    return out


def _identify_branch(scores: Dict[str, float]) -> str:
    """Which regime won, formatted for the branch_taken column."""
    if not scores:
        return 'NO_SCORES'
    return max(scores, key=scores.get)


def replay_one_day(panel: pd.DataFrame, t: pd.Timestamp) -> Dict:
    """Replay the detector on a single date t.

    Args:
        panel: Full SPY+VIX panel indexed by date.
        t: The classification date (must be in panel.index).

    Returns:
        Dict matching the labels.parquet schema for this date.

    Strict point-in-time: slices panel to [t-400d, t] inclusive of t.
    """
    if t not in panel.index:
        raise KeyError(f'{t} not in panel.index')

    slice_start = t - timedelta(days=WARMUP_DAYS)
    pt_panel = panel.loc[slice_start:t]

    spy_df = pt_panel[['spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume']].copy()
    spy_df.columns = ['open', 'high', 'low', 'close', 'volume']
    vix_df = pt_panel[['vix_open', 'vix_high', 'vix_low', 'vix_close']].copy()
    vix_df.columns = ['open', 'high', 'low', 'close']

    detector = MarketRegimeDetector(lookback_window=252)
    try:
        regime, confidence = detector.classify_regime(spy_df, vix_df, t.to_pydatetime())
    except DataInsufficientError:
        # Insufficient data this early in the window; mark as SAFE_MODE so
        # the downstream notebook can filter or treat it as a sentinel.
        regime, confidence = 'SAFE_MODE', float('nan')

    indicators = detector.last_indicators or {}
    scores = detector.last_regime_scores or {}

    alt_pcts = compute_alternative_vix_percentiles(panel, t)

    # Realized vol and VIX MA ratio (computed independently, NOT from detector).
    spy_close = panel['spy_close']
    returns = spy_close.pct_change()
    rv20 = float(returns.loc[:t].iloc[-20:].std() * np.sqrt(252)) if returns.loc[:t].size >= 20 else float('nan')
    rv60 = float(returns.loc[:t].iloc[-60:].std() * np.sqrt(252)) if returns.loc[:t].size >= 60 else float('nan')

    vix_close = panel['vix_close']
    vix_5d_ma = vix_close.loc[:t].iloc[-5:].mean()
    vix_ratio = float(vix_close.loc[t] / vix_5d_ma) if vix_5d_ma > 0 else float('nan')

    spy_history = spy_close.loc[:t]
    if len(spy_history) >= 252:
        peak = spy_history.iloc[-252:].max()
        dd = float(spy_history.iloc[-1] / peak - 1.0)
    else:
        dd = float('nan')

    return {
        'date': t.date() if hasattr(t, 'date') else t,
        'regime': regime,
        'confidence': float(confidence) if confidence == confidence else float('nan'),
        'above_20': bool(indicators.get('above_20', False)),
        'above_50': bool(indicators.get('above_50', False)),
        'above_200': bool(indicators.get('above_200', False)),
        'momentum_slope': float(indicators.get('momentum_slope', float('nan'))),
        'vix_close': float(panel['vix_close'].loc[t]),
        'vix_percentile_252d': float(indicators.get('vix_percentile', float('nan'))),
        'vix_percentile_63d': float(alt_pcts.get(63, float('nan'))),
        'vix_percentile_126d': float(alt_pcts.get(126, float('nan'))),
        'vix_percentile_504d': float(alt_pcts.get(504, float('nan'))),
        'realized_vol_20d': rv20,
        'realized_vol_60d': rv60,
        'vix_5d_ma_ratio': vix_ratio,
        'branch_taken': _identify_branch(scores),
        'spy_close': float(panel['spy_close'].loc[t]),
        'spy_drawdown_from_252d_high': dd,
    }


def replay_range(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 output: Path) -> pd.DataFrame:
    """Replay the detector across [start, end] inclusive and write Parquet."""
    if hasattr(start, 'to_pydatetime'):
        start_ts = start
    else:
        start_ts = pd.Timestamp(start)
    if hasattr(end, 'to_pydatetime'):
        end_ts = end
    else:
        end_ts = pd.Timestamp(end)

    dates_in_range = panel.index[(panel.index >= start_ts) & (panel.index <= end_ts)]
    records = []
    for i, t in enumerate(dates_in_range):
        if i % 250 == 0:
            logger.info(f'[+] Replaying day {i + 1}/{len(dates_in_range)}: {t.date()}')
        records.append(replay_one_day(panel, t))

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df['year'] = df['date'].dt.year
    df.to_parquet(output, partition_cols=['year'])
    logger.info(f'[+] Wrote {output} ({len(df)} rows)')
    return df


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f'{INPUT_PATH} not found. Run scripts/diagnostics/fetch_spy_vix.py first.'
        )
    panel = pd.read_parquet(INPUT_PATH)
    logger.info(f'[+] Loaded {len(panel)} rows from {INPUT_PATH}')

    end = panel.index.max()
    df = replay_range(panel, pd.Timestamp(REPLAY_START), end, OUTPUT_PATH)

    logger.info(f'[+] Done. {len(df)} replay days. Regime distribution:')
    logger.info(df['regime'].value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
