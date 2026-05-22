"""Test that MarketRegimeDetector exposes last_regime_scores after classify_regime."""
from datetime import datetime
import pandas as pd
import numpy as np
import pytest

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector


def test_last_regime_scores_populated_after_classify_regime():
    detector = MarketRegimeDetector()
    # 300 days of synthetic SPY + VIX data sufficient for the detector's lookback.
    idx = pd.date_range('2023-01-02', periods=300, freq='B')
    spy_data = pd.DataFrame({
        'open':  400 + np.arange(300) * 0.1,
        'high':  400 + np.arange(300) * 0.1 + 1,
        'low':   400 + np.arange(300) * 0.1 - 1,
        'close': 400 + np.arange(300) * 0.1,
        'volume': [1e6] * 300,
    }, index=idx)
    vix_data = pd.DataFrame({'close': 15 + np.sin(np.arange(300) / 20)}, index=idx)
    detector.classify_regime(spy_data, vix_data, idx[-1])
    assert hasattr(detector, 'last_regime_scores')
    assert detector.last_regime_scores is not None
    assert set(detector.last_regime_scores.keys()) == set(MarketRegimeDetector.REGIMES.keys())
