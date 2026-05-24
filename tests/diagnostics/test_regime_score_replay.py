"""TDD tests for scripts/diagnostics/regime_score_replay.py.

Mirrors tests/diagnostics/test_regime_detector_replay.py. Verifies the v0
record is preserved AND the 5 per-regime soft score columns are populated
correctly (within [0, 1] and consistent with the argmax winner).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.regime_score_replay import (
    REGIME_KEYS,
    replay_one_day_with_scores,
    replay_range_with_scores,
)


def _synthetic_panel(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2016-01-04', periods=n_days)
    spy = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days))
    vix = np.clip(15 + rng.normal(0, 4, n_days), 10, 50)
    return pd.DataFrame({
        'spy_open': spy * 0.999, 'spy_high': spy * 1.005,
        'spy_low': spy * 0.995, 'spy_close': spy, 'spy_volume': 1e8,
        'vix_open': vix, 'vix_high': vix * 1.02,
        'vix_low': vix * 0.98, 'vix_close': vix,
    }, index=dates)


def test_replay_one_day_with_scores_has_five_score_columns():
    """replay_one_day_with_scores produces 5 per-regime score columns."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    record = replay_one_day_with_scores(panel, t)
    for regime in REGIME_KEYS:
        col = f'score_{regime}'
        assert col in record, f'missing {col}'


def test_replay_one_day_with_scores_preserves_v0_columns():
    """v0 schema columns must still be present alongside the new score columns."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    record = replay_one_day_with_scores(panel, t)
    v0_required = {
        'date', 'regime', 'confidence',
        'above_20', 'above_50', 'above_200', 'momentum_slope',
        'vix_close', 'vix_percentile_252d',
        'vix_percentile_63d', 'vix_percentile_126d', 'vix_percentile_504d',
        'realized_vol_20d', 'realized_vol_60d', 'vix_5d_ma_ratio',
        'branch_taken', 'spy_close', 'spy_drawdown_from_252d_high',
    }
    missing = v0_required - set(record.keys())
    assert not missing, f'v0 columns missing: {missing}'


def test_replay_one_day_scores_in_unit_interval():
    """All 5 per-regime scores must lie within [0, 1] (or be NaN in SAFE_MODE)."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    record = replay_one_day_with_scores(panel, t)
    for regime in REGIME_KEYS:
        score = record[f'score_{regime}']
        if pd.isna(score):
            continue
        assert 0.0 <= score <= 1.0, (
            f'score_{regime} = {score} outside [0, 1]'
        )


def test_replay_one_day_argmax_matches_regime_column():
    """The argmax of the 5 score columns must equal the regime label.

    `regime` is by construction `max(regime_scores, key=...)`; this test
    guards against the score replay drifting out of sync with the v0
    classification (e.g. if the re-classification call ever diverged).
    """
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    record = replay_one_day_with_scores(panel, t)
    if record['regime'] == 'SAFE_MODE':
        pytest.skip('SAFE_MODE record -- no scores to argmax')
    scores = {r: record[f'score_{r}'] for r in REGIME_KEYS}
    if any(pd.isna(v) for v in scores.values()):
        pytest.skip('NaN scores -- cannot argmax')
    argmax = max(scores, key=scores.get)
    assert argmax == record['regime'], (
        f'argmax={argmax} but record["regime"]={record["regime"]}; '
        f'scores={scores}'
    )


def test_replay_range_with_scores_writes_parquet(tmp_path: Path):
    """replay_range_with_scores writes Parquet with all 5 score columns."""
    panel = _synthetic_panel(800)
    output = tmp_path / 'labels.parquet'
    replay_range_with_scores(panel, panel.index[300], panel.index[-1], output)
    df = pd.read_parquet(output)
    assert len(df) > 400
    for regime in REGIME_KEYS:
        col = f'score_{regime}'
        assert col in df.columns, f'missing {col} in written parquet'


def test_replay_range_argmax_consistency_in_written_parquet(tmp_path: Path):
    """For every non-SAFE_MODE row in the written parquet, the argmax of
    scores must equal the regime column."""
    panel = _synthetic_panel(800)
    output = tmp_path / 'labels.parquet'
    replay_range_with_scores(panel, panel.index[300], panel.index[-1], output)
    df = pd.read_parquet(output)

    score_cols = [f'score_{r}' for r in REGIME_KEYS]
    df_valid = df[df['regime'] != 'SAFE_MODE'].dropna(subset=score_cols)
    if df_valid.empty:
        pytest.skip('No valid rows to test')

    argmax_regime = df_valid[score_cols].idxmax(axis=1).str.replace('score_', '', regex=False)
    mismatches = (argmax_regime.values != df_valid['regime'].values).sum()
    assert mismatches == 0, (
        f'{mismatches} of {len(df_valid)} rows have argmax-vs-regime mismatch'
    )
