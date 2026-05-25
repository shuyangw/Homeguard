"""Day-by-day replay of MarketRegimeDetectorV1 across 2017-2026.

Reads diagnostics/data/spy_vix_2016_2026.parquet, loads the trained LightGBM
artifact via MarketRegimeDetectorV1, and emits a Parquet with the v1 regime
labels and the underlying BEAR probability.

Output schema mirrors v0 labels.parquet plus a `bear_proba` column.

Usage:
    PYTHONPATH=. python scripts/diagnostics/regime_detector_v1_replay.py [--model-path <path>]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.leading_indicators import load_leading_indicators
from src.settings import get_local_storage_dir
from src.strategies.advanced.market_regime_detector import DataInsufficientError
from src.strategies.advanced.market_regime_detector_v1 import MarketRegimeDetectorV1
from src.utils.logger import get_logger

logger = get_logger(__name__)


INPUT_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
OUTPUT_PATH = Path('diagnostics/regime/v1/labels.parquet')
REPLAY_START = datetime(2017, 1, 1)
REPLAY_END = datetime(2026, 5, 22)
WARMUP_DAYS = 400


def _identify_branch(scores: Dict[str, float]) -> str:
    if not scores:
        return 'NO_SCORES'
    return max(scores, key=scores.get)


def replay_one_day(
    panel: pd.DataFrame,
    t: pd.Timestamp,
    detector: MarketRegimeDetectorV1,
) -> Dict:
    """Replay the v1 detector on a single date."""
    if t not in panel.index:
        raise KeyError(f'{t} not in panel.index')

    slice_start = t - timedelta(days=WARMUP_DAYS)
    pt_panel = panel.loc[slice_start:t]

    spy_df = pt_panel[['spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume']].copy()
    spy_df.columns = ['open', 'high', 'low', 'close', 'volume']
    vix_df = pt_panel[['vix_open', 'vix_high', 'vix_low', 'vix_close']].copy()
    vix_df.columns = ['open', 'high', 'low', 'close']

    try:
        regime, confidence = detector.classify_regime(
            spy_df, vix_df, t.to_pydatetime()
        )
    except DataInsufficientError:
        regime, confidence = 'SAFE_MODE', float('nan')

    indicators = detector.last_indicators or {}
    scores = detector.last_regime_scores or {}
    bear_proba = detector.last_bear_probability
    bear_proba = float(bear_proba) if bear_proba is not None else float('nan')

    spy_close = panel['spy_close']
    returns = spy_close.pct_change()
    rv20 = (
        float(returns.loc[:t].iloc[-20:].std() * np.sqrt(252))
        if returns.loc[:t].size >= 20 else float('nan')
    )
    rv60 = (
        float(returns.loc[:t].iloc[-60:].std() * np.sqrt(252))
        if returns.loc[:t].size >= 60 else float('nan')
    )

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
        'bear_proba': bear_proba,
        'above_20': bool(indicators.get('above_20', False)),
        'above_50': bool(indicators.get('above_50', False)),
        'above_200': bool(indicators.get('above_200', False)),
        'momentum_slope': float(indicators.get('momentum_slope', float('nan'))),
        'vix_close': float(panel['vix_close'].loc[t]),
        'vix_percentile': float(indicators.get('vix_percentile', float('nan'))),
        'vix_term_ratio': float(indicators.get('vix_term_ratio', float('nan'))),
        'hy_proxy_ratio': float(indicators.get('hy_proxy_ratio', float('nan'))),
        'breadth_pct': float(indicators.get('breadth_pct', float('nan'))),
        'skew_close': float(indicators.get('skew_close', float('nan'))),
        'realized_vol_20d': rv20,
        'realized_vol_60d': rv60,
        'branch_taken': _identify_branch(scores),
        'spy_close': float(panel['spy_close'].loc[t]),
        'spy_drawdown_from_252d_high': dd,
        'score_strong_bull': float(scores.get('STRONG_BULL', float('nan'))),
        'score_weak_bull': float(scores.get('WEAK_BULL', float('nan'))),
        'score_sideways': float(scores.get('SIDEWAYS', float('nan'))),
        'score_unpredictable': float(scores.get('UNPREDICTABLE', float('nan'))),
        'score_bear': float(scores.get('BEAR', float('nan'))),
    }


def replay_range(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    detector: MarketRegimeDetectorV1,
    output: Path,
) -> pd.DataFrame:
    """Replay the v1 detector across [start, end] inclusive."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    dates_in_range = panel.index[(panel.index >= start_ts) & (panel.index <= end_ts)]
    records = []
    for i, t in enumerate(dates_in_range):
        if i % 250 == 0:
            logger.info(f'[+] v1 replay day {i + 1}/{len(dates_in_range)}: {t.date()}')
        records.append(replay_one_day(panel, t, detector))

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df['year'] = df['date'].dt.year
    df.to_parquet(output, partition_cols=['year'])
    logger.info(f'[+] Wrote {output} ({len(df)} rows)')
    return df


def _find_latest_model() -> Path:
    """Discover the most recently saved v20 detector model artifact."""
    base = get_local_storage_dir() / 'alt_data' / 'models' / 'v20_detector'
    if not base.exists():
        raise FileNotFoundError(f'No v20_detector model directory at {base}')
    candidates = sorted(
        base.glob('*/model.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f'No model.pkl artifacts under {base}; run train_detector_v1.py first'
        )
    return candidates[0]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Override model artifact path (default: most recent under v20_detector/)',
    )
    args = parser.parse_args(argv)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f'{INPUT_PATH} not found. Run scripts/diagnostics/fetch_spy_vix.py first.'
        )
    panel = pd.read_parquet(INPUT_PATH)
    logger.info(f'[+] Loaded {len(panel)} rows from {INPUT_PATH}')

    model_path = Path(args.model_path) if args.model_path else _find_latest_model()
    logger.info(f'[+] Using model: {model_path}')

    leading = load_leading_indicators(REPLAY_START, REPLAY_END, cache=True)
    detector = MarketRegimeDetectorV1(
        model_path=model_path,
        leading_panel=leading,
    )

    end = min(panel.index.max(), pd.Timestamp(REPLAY_END))
    df = replay_range(
        panel, pd.Timestamp(REPLAY_START), end, detector, OUTPUT_PATH
    )

    logger.info(f'[+] Done. {len(df)} replay days. Regime distribution:')
    logger.info(df['regime'].value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
