"""WS-3d replacement detector: LightGBM on leading indicators.

Schema-compatible with src/strategies/advanced/market_regime_detector.py:
- classify_regime(spy_data, vix_data, timestamp) -> Tuple[str, float]
- last_regime_scores: Dict[str, float]
- last_classification_timestamp: Optional[datetime]
- last_indicators: Dict (with at least the v0 fields above_20, above_50,
  above_200, vix_percentile for downstream compatibility)

Trained on G1_BEAR labels via purged combinatorial cross-validation;
output is P(G1_BEAR | indicators) in [0, 1]. The 5 regime labels
(STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR) are mapped from
a combination of BEAR probability + retained v0-style features.

Loads its trained LightGBM model artifact from disk at construction.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data.leading_indicators import load_leading_indicators
from src.strategies.advanced.market_regime_detector import DataInsufficientError
from src.utils.logger import get_logger

logger = get_logger(__name__)


FEATURE_COLUMNS: List[str] = [
    # 4 canonical leading indicators (WS-3d input set, Amendment 5)
    'vix_term_ratio',
    'hy_proxy_ratio',
    'breadth_pct',
    'skew_close',
    # 4 v0-style backwards-compat features for non-BEAR regime mapping
    'above_20',
    'above_50',
    'above_200',
    'vix_percentile',
]

# BEAR probability threshold used to flip the argmax to BEAR. Pre-spec value;
# downstream consumers (V20) Schmitt-trigger on the raw probability, not on
# the argmax regime label.
BEAR_PROB_THRESHOLD: float = 0.5

# Minimum trading days needed for SMA / VIX percentile computation.
_MIN_SPY_ROWS: int = 200
_MIN_VIX_ROWS: int = 252

# Leading-indicator panel cache: avoid reloading per classify_regime call
# (which would hit yfinance once per day in replay). Keyed by (start, end).
_PANEL_CACHE: Dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}


def _default_panel_window() -> Tuple[datetime, datetime]:
    """Default leading-indicator panel window: 2017-01-01 to today."""
    return datetime(2017, 1, 1), datetime.now()


def _get_leading_panel(
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Cached load of the leading-indicator joined panel."""
    key = (pd.Timestamp(start), pd.Timestamp(end))
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]
    df = load_leading_indicators(start, end, cache=True)
    _PANEL_CACHE[key] = df
    return df


def compute_v0_features(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    lookback_window: int = 252,
) -> Dict[str, float]:
    """Compute the 4 v0-style features the v1 detector consumes for non-BEAR
    regime mapping: above_20, above_50, above_200, vix_percentile.

    Also returns momentum_slope for last_indicators compatibility.
    """
    spy_close = spy_data['close']
    if len(spy_close) < _MIN_SPY_ROWS:
        raise ValueError(
            f'compute_v0_features: need >= {_MIN_SPY_ROWS} SPY rows, '
            f'got {len(spy_close)}'
        )

    sma_20 = spy_close.rolling(20).mean()
    sma_50 = spy_close.rolling(50).mean()
    sma_200 = spy_close.rolling(200).mean()

    current_price = float(spy_close.iloc[-1])
    above_20 = bool(current_price > sma_20.iloc[-1])
    above_50 = bool(current_price > sma_50.iloc[-1])
    above_200 = bool(current_price > sma_200.iloc[-1])
    momentum_slope = float(
        (sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20]
    )

    vix_close = vix_data['close']
    current_vix = float(vix_close.iloc[-1])
    lookback = vix_close.iloc[-lookback_window:]
    vix_percentile = float((lookback < current_vix).sum() / len(lookback) * 100.0)

    return {
        'above_20': above_20,
        'above_50': above_50,
        'above_200': above_200,
        'momentum_slope': momentum_slope,
        'vix_percentile': vix_percentile,
    }


def build_feature_row(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    timestamp: datetime,
    leading_panel: Optional[pd.DataFrame] = None,
    lookback_window: int = 252,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build a single-row feature DataFrame matching FEATURE_COLUMNS.

    Joins the 4 leading indicators (as-of `timestamp`) with the 4 v0-style
    features computed from spy_data / vix_data.

    Returns (X_row_df, v0_features_dict).
    """
    v0_feats = compute_v0_features(spy_data, vix_data, lookback_window=lookback_window)

    if leading_panel is None:
        panel_start, panel_end = _default_panel_window()
        leading_panel = _get_leading_panel(panel_start, panel_end)

    ts = pd.Timestamp(timestamp).normalize()
    available = leading_panel.loc[leading_panel.index <= ts]
    if len(available) == 0:
        raise DataInsufficientError(
            f'No leading_indicator data <= {ts.date()}',
            coverage_pct=0.0,
            threshold_pct=1.0,
            hard_block=True,
        )
    leading_row = available.iloc[-1]

    row = {
        'vix_term_ratio': float(leading_row['vix_term_ratio']),
        'hy_proxy_ratio': float(leading_row['hy_proxy_ratio']),
        'breadth_pct': float(leading_row['breadth_pct']),
        'skew_close': float(leading_row['skew_close']),
        'above_20': float(v0_feats['above_20']),
        'above_50': float(v0_feats['above_50']),
        'above_200': float(v0_feats['above_200']),
        'vix_percentile': float(v0_feats['vix_percentile']),
    }
    X_row = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return X_row, v0_feats


def _v0_style_nonbear_scores(v0_feats: Dict[str, float]) -> Dict[str, float]:
    """Produce v0-style scores for the 4 non-BEAR regimes.

    Mirrors src/strategies/advanced/market_regime_detector.py::_score_regime
    using only the subset of features the v1 detector retains. The BEAR score
    is filled in separately by classify_regime from the LightGBM probability.
    """
    above_20 = bool(v0_feats['above_20'])
    above_50 = bool(v0_feats['above_50'])
    above_200 = bool(v0_feats['above_200'])
    vix_pct = float(v0_feats['vix_percentile'])
    momentum = float(v0_feats['momentum_slope'])

    # STRONG_BULL: momentum >= 0.02, vix_pct <= 30, above 20/50/200
    sb_score = 0.0
    sb_score += 1.0 if momentum >= 0.02 else 0.0
    sb_score += 1.0 if vix_pct <= 30 else 0.0
    sb_score += (int(above_20) + int(above_50) + int(above_200)) / 3.0
    sb_score /= 3.0

    # WEAK_BULL: 0 <= momentum <= 0.02, vix_pct <= 50, above 20/50
    wb_score = 0.0
    wb_score += 1.0 if momentum >= 0.0 else 0.0
    wb_score += 1.0 if momentum <= 0.02 else 0.0
    wb_score += 1.0 if vix_pct <= 50 else 0.0
    wb_score += (int(above_20) + int(above_50)) / 2.0
    wb_score /= 4.0

    # SIDEWAYS: -0.01 <= momentum <= 0.01, 30 <= vix_pct <= 60
    sw_score = 0.0
    sw_score += 1.0 if momentum >= -0.01 else 0.0
    sw_score += 1.0 if momentum <= 0.01 else 0.0
    sw_score += 1.0 if vix_pct >= 30 else 0.0
    sw_score += 1.0 if vix_pct <= 60 else 0.0
    sw_score /= 4.0

    # UNPREDICTABLE: vix_pct >= 60 (volatility_spike not computable here;
    # mapped to vix_pct >= 75 as a stricter proxy)
    up_score = 0.0
    up_score += 1.0 if vix_pct >= 60 else 0.0
    up_score += 1.0 if vix_pct >= 75 else 0.0
    up_score /= 2.0

    return {
        'STRONG_BULL': sb_score,
        'WEAK_BULL': wb_score,
        'SIDEWAYS': sw_score,
        'UNPREDICTABLE': up_score,
    }


class MarketRegimeDetectorV1:
    """WS-3d LightGBM-backed regime detector.

    Schema-compatible with MarketRegimeDetector (v0):
    - classify_regime(spy_data, vix_data, timestamp) -> Tuple[str, float]
    - last_regime_scores: Dict[str, float]  (5 keys)
    - last_classification_timestamp: Optional[datetime]
    - last_indicators: Dict (above_20, above_50, above_200, momentum_slope,
      vix_percentile, plus the 4 leading indicators)

    The BEAR score is P(G1_BEAR | indicators) from the LightGBM classifier;
    the 4 non-BEAR scores are derived from v0-style features for downstream
    compatibility.
    """

    REGIMES = {
        'STRONG_BULL': 1,
        'WEAK_BULL': 2,
        'SIDEWAYS': 3,
        'UNPREDICTABLE': 4,
        'BEAR': 5,
    }

    def __init__(
        self,
        model_path: Path,
        lookback_window: int = 252,
        leading_panel: Optional[pd.DataFrame] = None,
        bear_prob_threshold: float = BEAR_PROB_THRESHOLD,
    ):
        """Load the trained LightGBM model from disk.

        Args:
            model_path: Path to the joblib-serialized LightGBM model artifact.
            lookback_window: VIX percentile window (default 252 trading days).
            leading_panel: Optional pre-loaded leading-indicator DataFrame.
                If None, will load on-demand via the unified loader cache.
            bear_prob_threshold: Probability >= this maps to argmax BEAR.
        """
        self.lookback_window = lookback_window
        self.bear_prob_threshold = bear_prob_threshold
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f'MarketRegimeDetectorV1: model artifact not found at '
                f'{self._model_path}'
            )
        self._model = joblib.load(self._model_path)
        self._leading_panel = leading_panel

        self.last_indicators: Optional[Dict] = None
        self.last_regime_scores: Optional[Dict[str, float]] = None
        self.last_classification_timestamp: Optional[datetime] = None
        self.last_bear_probability: Optional[float] = None

        logger.info(
            f'[+] MarketRegimeDetectorV1: loaded model from {self._model_path}'
        )

    def _predict_bear_prob(self, X_row: pd.DataFrame) -> float:
        """Predict P(G1_BEAR | features) for a single feature row.

        Supports both sklearn-API LightGBM (predict_proba) and native Booster.
        """
        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(X_row)
            return float(proba[0, 1])
        # Native Booster: predict returns probability directly for binary obj.
        raw = self._model.predict(X_row)
        return float(np.asarray(raw).ravel()[0])

    def classify_regime(
        self,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        timestamp: datetime,
        *,
        min_coverage_pct: float = 0.95,
        hard_block_pct: float = 0.80,
    ) -> Tuple[str, float]:
        """Classify current market regime.

        Schema-compatible with MarketRegimeDetector.classify_regime.
        """
        spy_close = spy_data['close']
        spy_coverage = (
            float(spy_close.notna().mean()) if len(spy_close) > 0 else 0.0
        )
        if spy_coverage < hard_block_pct:
            raise DataInsufficientError(
                f'SPY coverage {spy_coverage:.1%} below hard-block '
                f'{hard_block_pct:.0%}',
                coverage_pct=spy_coverage,
                threshold_pct=hard_block_pct,
                hard_block=True,
            )
        if spy_coverage < min_coverage_pct:
            raise DataInsufficientError(
                f'SPY coverage {spy_coverage:.1%} below safe-mode '
                f'{min_coverage_pct:.0%}',
                coverage_pct=spy_coverage,
                threshold_pct=min_coverage_pct,
                hard_block=False,
            )

        if len(spy_data) < _MIN_SPY_ROWS or len(vix_data) < _MIN_VIX_ROWS:
            logger.warning(
                f'MarketRegimeDetectorV1: insufficient rows '
                f'(spy={len(spy_data)}, vix={len(vix_data)})'
            )
            return 'SIDEWAYS', 0.5

        X_row, v0_feats = build_feature_row(
            spy_data,
            vix_data,
            timestamp,
            leading_panel=self._leading_panel,
            lookback_window=self.lookback_window,
        )

        bear_prob = self._predict_bear_prob(X_row)
        self.last_bear_probability = bear_prob

        nonbear_scores = _v0_style_nonbear_scores(v0_feats)
        scores: Dict[str, float] = dict(nonbear_scores)
        scores['BEAR'] = bear_prob

        # Argmax mapping: if BEAR prob >= threshold, force BEAR; otherwise
        # take argmax of the 4 non-BEAR scores. This preserves the v0 5-key
        # contract for downstream consumers.
        if bear_prob >= self.bear_prob_threshold:
            best_regime = 'BEAR'
            confidence = bear_prob
        else:
            nonbear_best = max(nonbear_scores, key=nonbear_scores.get)
            best_regime = nonbear_best
            confidence = nonbear_scores[nonbear_best]

        # Populate last_indicators with v0-compat fields plus leading indicators
        leading_vals = X_row.iloc[0].to_dict()
        self.last_indicators = {
            'above_20': v0_feats['above_20'],
            'above_50': v0_feats['above_50'],
            'above_200': v0_feats['above_200'],
            'momentum_slope': v0_feats['momentum_slope'],
            'vix_percentile': v0_feats['vix_percentile'],
            'vix_term_ratio': leading_vals['vix_term_ratio'],
            'hy_proxy_ratio': leading_vals['hy_proxy_ratio'],
            'breadth_pct': leading_vals['breadth_pct'],
            'skew_close': leading_vals['skew_close'],
            'bear_probability': bear_prob,
        }
        self.last_regime_scores = dict(scores)
        self.last_classification_timestamp = timestamp

        return best_regime, float(confidence)
