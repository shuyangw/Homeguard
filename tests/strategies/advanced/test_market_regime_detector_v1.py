"""Tests for MarketRegimeDetectorV1 -- WS-3d LightGBM detector.

Covers:
- Schema compatibility with v0 (5 regime keys, last_indicators, freshness).
- Load-from-disk behavior + FileNotFoundError on missing artifact.
- Insufficient-data error path (DataInsufficientError).
- Argmax-flip-on-BEAR-probability behavior.
- v0-style feature computation matches what classify_regime emits.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest

from src.strategies.advanced.market_regime_detector import DataInsufficientError
from src.strategies.advanced.market_regime_detector_v1 import (
    BEAR_PROB_THRESHOLD,
    FEATURE_COLUMNS,
    MarketRegimeDetectorV1,
    _v0_style_nonbear_scores,
    build_feature_row,
    compute_v0_features,
)


class _FakeModel:
    """Deterministic stub with predict_proba returning a fixed probability."""

    def __init__(self, bear_prob: float):
        self._p = bear_prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self._p), np.full(n, self._p)])


def _build_panel(n_days: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range('2024-01-02', periods=n_days, freq='B')
    spy = pd.DataFrame({
        'open': 400.0 + np.arange(n_days) * 0.1,
        'high': 401.0 + np.arange(n_days) * 0.1,
        'low': 399.0 + np.arange(n_days) * 0.1,
        'close': 400.0 + np.arange(n_days) * 0.1,
        'volume': 1_000_000.0,
    }, index=dates)
    vix = pd.DataFrame({
        'open': 15.0, 'high': 16.0, 'low': 14.0, 'close': 15.0,
    }, index=dates)
    return spy, vix


def _build_leading_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        'vix_close': np.full(n, 15.0),
        'vix3m_close': np.full(n, 18.0),
        'vix_term_ratio': np.full(n, 0.83),
        'hyg_close': np.full(n, 80.0),
        'ief_close': np.full(n, 100.0),
        'hy_proxy_ratio': np.full(n, 0.80),
        'breadth_pct': np.full(n, 65.0),
        'n_constituents': np.full(n, 500),
        'skew_close': np.full(n, 140.0),
    }, index=dates)


def _make_detector(
    tmp_path: Path,
    bear_prob: float,
    leading_panel: pd.DataFrame,
) -> MarketRegimeDetectorV1:
    fake = _FakeModel(bear_prob)
    model_path = tmp_path / 'fake_model.pkl'
    joblib.dump(fake, model_path)
    return MarketRegimeDetectorV1(
        model_path=model_path, leading_panel=leading_panel,
    )


def test_feature_columns_exposed():
    """FEATURE_COLUMNS lists the 8 inputs (4 leading + 4 v0-style)."""
    assert len(FEATURE_COLUMNS) == 8
    assert 'vix_term_ratio' in FEATURE_COLUMNS
    assert 'hy_proxy_ratio' in FEATURE_COLUMNS
    assert 'breadth_pct' in FEATURE_COLUMNS
    assert 'skew_close' in FEATURE_COLUMNS
    assert 'above_20' in FEATURE_COLUMNS
    assert 'above_50' in FEATURE_COLUMNS
    assert 'above_200' in FEATURE_COLUMNS
    assert 'vix_percentile' in FEATURE_COLUMNS


def test_missing_model_raises(tmp_path):
    """Constructor raises FileNotFoundError when artifact does not exist."""
    with pytest.raises(FileNotFoundError):
        MarketRegimeDetectorV1(model_path=tmp_path / 'nope.pkl')


def test_classify_5_regime_keys(tmp_path):
    """last_regime_scores has exactly 5 keys (v0 schema compat)."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    detector = _make_detector(tmp_path, bear_prob=0.10, leading_panel=leading)

    ts = spy.index[-1].to_pydatetime()
    regime, conf = detector.classify_regime(spy, vix, ts)
    assert detector.last_regime_scores is not None
    assert set(detector.last_regime_scores.keys()) == {
        'STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE', 'BEAR',
    }
    assert detector.last_classification_timestamp == ts


def test_bear_argmax_flip_above_threshold(tmp_path):
    """When P(BEAR) >= threshold the argmax regime is BEAR."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    detector = _make_detector(
        tmp_path, bear_prob=BEAR_PROB_THRESHOLD + 0.05, leading_panel=leading,
    )
    ts = spy.index[-1].to_pydatetime()
    regime, conf = detector.classify_regime(spy, vix, ts)
    assert regime == 'BEAR'
    assert conf == pytest.approx(BEAR_PROB_THRESHOLD + 0.05, abs=1e-6)


def test_no_bear_flip_below_threshold(tmp_path):
    """When P(BEAR) < threshold the argmax regime is NOT BEAR."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    detector = _make_detector(
        tmp_path, bear_prob=0.10, leading_panel=leading,
    )
    ts = spy.index[-1].to_pydatetime()
    regime, conf = detector.classify_regime(spy, vix, ts)
    assert regime != 'BEAR'


def test_last_indicators_has_v0_and_leading_fields(tmp_path):
    """last_indicators carries v0-style fields + leading-indicator fields."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    detector = _make_detector(tmp_path, bear_prob=0.20, leading_panel=leading)
    ts = spy.index[-1].to_pydatetime()
    detector.classify_regime(spy, vix, ts)
    ind = detector.last_indicators
    for k in (
        'above_20', 'above_50', 'above_200', 'momentum_slope', 'vix_percentile',
        'vix_term_ratio', 'hy_proxy_ratio', 'breadth_pct', 'skew_close',
        'bear_probability',
    ):
        assert k in ind, f'missing key in last_indicators: {k}'


def test_insufficient_coverage_raises_hard_block(tmp_path):
    """SPY coverage below hard_block_pct raises DataInsufficientError(hard)."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    detector = _make_detector(tmp_path, bear_prob=0.10, leading_panel=leading)
    # 70% coverage -> below 80% hard-block threshold
    n = len(spy)
    spy.iloc[: int(n * 0.3), spy.columns.get_loc('close')] = np.nan
    with pytest.raises(DataInsufficientError) as exc_info:
        detector.classify_regime(spy, vix, spy.index[-1].to_pydatetime())
    assert exc_info.value.hard_block is True


def test_double_call_idempotent(tmp_path):
    """Two calls with identical inputs return identical outputs."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    detector = _make_detector(tmp_path, bear_prob=0.25, leading_panel=leading)
    ts = spy.index[-1].to_pydatetime()
    r1 = detector.classify_regime(spy, vix, ts)
    s1 = dict(detector.last_regime_scores)
    r2 = detector.classify_regime(spy, vix, ts)
    s2 = dict(detector.last_regime_scores)
    assert r1 == r2
    assert s1 == s2


def test_compute_v0_features_basic():
    """compute_v0_features returns expected keys on the minimal panel."""
    spy, vix = _build_panel(n_days=260)
    feats = compute_v0_features(spy, vix, lookback_window=252)
    assert set(feats.keys()) == {
        'above_20', 'above_50', 'above_200', 'momentum_slope', 'vix_percentile',
    }
    # Monotone-up SPY: above all SMAs.
    assert feats['above_20'] is True
    assert feats['above_50'] is True
    assert feats['above_200'] is True


def test_build_feature_row_shape():
    """build_feature_row returns a 1-row DataFrame with FEATURE_COLUMNS."""
    spy, vix = _build_panel()
    leading = _build_leading_panel(spy.index)
    ts = spy.index[-1].to_pydatetime()
    X_row, v0_feats = build_feature_row(spy, vix, ts, leading_panel=leading)
    assert X_row.shape == (1, len(FEATURE_COLUMNS))
    assert list(X_row.columns) == FEATURE_COLUMNS


def test_nonbear_scores_4_keys():
    """_v0_style_nonbear_scores returns 4 keys (BEAR excluded)."""
    feats = {
        'above_20': True, 'above_50': True, 'above_200': True,
        'momentum_slope': 0.005, 'vix_percentile': 35.0,
    }
    scores = _v0_style_nonbear_scores(feats)
    assert set(scores.keys()) == {
        'STRONG_BULL', 'WEAK_BULL', 'SIDEWAYS', 'UNPREDICTABLE',
    }
    for s in scores.values():
        assert 0.0 <= s <= 1.0
