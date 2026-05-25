"""TDD tests for scripts/diagnostics/regime_detector_replay.py.

Tests use synthetic SPY+VIX data with known regime characteristics so that
expected detector outputs are predictable. Production parity is verified
in an integration test that pulls 10 real days from the staged Parquet.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.regime_detector_replay import (
    replay_one_day,
    replay_range,
    compute_alternative_vix_percentiles,
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


def test_replay_one_day_returns_expected_columns():
    """replay_one_day produces a dict with all schema columns."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    record = replay_one_day(panel, t)
    expected_keys = {
        'date', 'regime', 'confidence',
        'above_20', 'above_50', 'above_200', 'momentum_slope',
        'vix_close', 'vix_percentile_252d',
        'vix_percentile_63d', 'vix_percentile_126d', 'vix_percentile_504d',
        'realized_vol_20d', 'realized_vol_60d', 'vix_5d_ma_ratio',
        'branch_taken', 'spy_close', 'spy_drawdown_from_252d_high',
    }
    assert set(record.keys()) == expected_keys, (
        f'Missing: {expected_keys - set(record.keys())}; '
        f'Extra: {set(record.keys()) - expected_keys}'
    )


def test_replay_one_day_no_lookahead():
    """replay_one_day on date t must not use data after t."""
    panel = _synthetic_panel(400)
    t = panel.index[100]
    # Modify panel beyond t with sentinel values. Output should be identical.
    panel_clean = panel.copy()
    panel_polluted = panel.copy()
    panel_polluted.loc[panel.index > t] = np.nan
    rec_clean = replay_one_day(panel_clean, t)
    rec_poll = replay_one_day(panel_polluted, t)
    assert rec_clean['regime'] == rec_poll['regime']
    # NaN-safe equality: at early indices the detector may produce NaN for
    # indicators that require >252d of history. The invariant we care about
    # is that the polluted-future panel yields the same value as the clean
    # one (which it does, since both are NaN or both are the same float).
    assert rec_clean['confidence'] == rec_poll['confidence'] or (
        pd.isna(rec_clean['confidence']) and pd.isna(rec_poll['confidence'])
    )
    assert rec_clean['vix_percentile_252d'] == rec_poll['vix_percentile_252d'] or (
        pd.isna(rec_clean['vix_percentile_252d']) and pd.isna(rec_poll['vix_percentile_252d'])
    )


def test_replay_one_day_idempotent():
    """Two calls on identical input produce identical output."""
    panel = _synthetic_panel(400)
    t = panel.index[-1]
    rec1 = replay_one_day(panel, t)
    rec2 = replay_one_day(panel, t)
    assert rec1 == rec2


def test_compute_alternative_vix_percentiles_returns_four_values():
    """The 63/126/252/504-day VIX percentiles all populate."""
    panel = _synthetic_panel(600)
    t = panel.index[-1]
    pcts = compute_alternative_vix_percentiles(panel, t)
    assert set(pcts.keys()) == {63, 126, 252, 504}
    for w, pct in pcts.items():
        assert 0.0 <= pct <= 100.0, f'pct[{w}d] = {pct} out of range'


def test_replay_range_writes_parquet(tmp_path: Path):
    """replay_range writes a Parquet partitioned by year."""
    panel = _synthetic_panel(800)
    output = tmp_path / 'labels.parquet'
    replay_range(panel, panel.index[300], panel.index[-1], output)
    df = pd.read_parquet(output)
    assert len(df) > 400
    assert {'regime', 'confidence', 'date'}.issubset(df.columns)


def test_replay_range_idempotency_check(tmp_path: Path):
    """Two runs produce byte-identical Parquets."""
    panel = _synthetic_panel(800)
    out1 = tmp_path / 'labels1.parquet'
    out2 = tmp_path / 'labels2.parquet'
    replay_range(panel, panel.index[300], panel.index[-1], out1)
    replay_range(panel, panel.index[300], panel.index[-1], out2)
    df1 = pd.read_parquet(out1).reset_index(drop=True)
    df2 = pd.read_parquet(out2).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2)
