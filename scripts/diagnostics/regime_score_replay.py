"""Day-by-day replay of MarketRegimeDetector emitting per-regime soft scores.

Extends scripts/diagnostics/regime_detector_replay.py by augmenting each
replay record with the full 5-element score vector from
``MarketRegimeDetector.last_regime_scores`` (populated after every
``classify_regime`` call). The v0 replay only captured the argmax winner;
this v0_scores replay captures all five so downstream analyses can ask
"how early did BEAR's score rise above tau?" rather than only "when did
BEAR become the argmax winner?".

Usage:
    PYTHONPATH=. python scripts/diagnostics/regime_score_replay.py

Output:
    diagnostics/regime/v0_scores/labels.parquet (partitioned by year).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.diagnostics.regime_detector_replay import (
    INPUT_PATH,
    replay_one_day as _v0_replay_one_day,
)
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import logger


OUTPUT_PATH = Path('diagnostics/regime/v0_scores/labels.parquet')
REPLAY_START = datetime(2017, 1, 1)
REGIME_KEYS = ('STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR')


def replay_one_day_with_scores(panel: pd.DataFrame, t: pd.Timestamp) -> Dict:
    """Run the v0 replay then attach per-regime soft scores.

    The v0 helper instantiates its own detector and calls classify_regime
    internally, so we re-run the same call here to harvest
    last_regime_scores rather than restructuring the v0 helper.

    Both calls use identical inputs / settings, so the v0 record's
    ``regime`` is the argmax of the scores recorded here -- the test suite
    asserts this invariance directly.
    """
    record = _v0_replay_one_day(panel, t)

    # Re-run classification to capture last_regime_scores. The v0 helper
    # uses a fresh detector each call and discards it, so we cannot read
    # the scores from there. The detector is deterministic on the same
    # slice + settings so the scores match the v0 invocation exactly.
    slice_start = t - pd.Timedelta(days=400)
    pt_panel = panel.loc[slice_start:t]
    spy_df = pt_panel[['spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume']].copy()
    spy_df.columns = ['open', 'high', 'low', 'close', 'volume']
    vix_df = pt_panel[['vix_open', 'vix_high', 'vix_low', 'vix_close']].copy()
    vix_df.columns = ['open', 'high', 'low', 'close']

    detector = MarketRegimeDetector(lookback_window=252)
    try:
        detector.classify_regime(spy_df, vix_df, t.to_pydatetime())
    except Exception:
        pass  # SAFE_MODE record already produced by v0 helper; scores stay NaN.

    scores = detector.last_regime_scores or {}
    for regime in REGIME_KEYS:
        record[f'score_{regime}'] = float(scores.get(regime, float('nan')))
    return record


def replay_range_with_scores(panel: pd.DataFrame, start: pd.Timestamp,
                             end: pd.Timestamp, output: Path) -> pd.DataFrame:
    """Replay the detector across [start, end] inclusive, writing Parquet."""
    start_ts = start if hasattr(start, 'to_pydatetime') else pd.Timestamp(start)
    end_ts = end if hasattr(end, 'to_pydatetime') else pd.Timestamp(end)

    dates_in_range = panel.index[(panel.index >= start_ts) & (panel.index <= end_ts)]
    records = []
    for i, t in enumerate(dates_in_range):
        if i % 250 == 0:
            logger.info(f'[+] Replaying day {i + 1}/{len(dates_in_range)}: {t.date()}')
        records.append(replay_one_day_with_scores(panel, t))

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
    df = replay_range_with_scores(panel, pd.Timestamp(REPLAY_START), end, OUTPUT_PATH)

    logger.info(f'[+] Done. {len(df)} replay days. Per-regime mean scores:')
    for regime in REGIME_KEYS:
        col = f'score_{regime}'
        logger.info(f'  {regime:14s} mean={df[col].mean():.3f} max={df[col].max():.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
