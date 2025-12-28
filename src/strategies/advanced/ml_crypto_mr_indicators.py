"""
ML Crypto Mean Reversion Indicators.

Implements indicators and ML regime filter for the Mean Reversion crypto strategy.
The key insight: mean reversion works great in ranging markets but fails in trending
markets. The ML filter identifies ranging conditions to maximize strategy edge.

Indicators:
- Z-score (mean reversion signal)
- RSI (momentum confirmation)
- ATR (risk management)
- ADX (trend strength)
- Choppiness Index (range detection)
- Efficiency Ratio (trend quality)
- Bollinger Width (volatility)

ML Regime Filter:
- GradientBoostingClassifier to predict ranging vs trending regimes
- Features: ADX, Choppiness, Efficiency Ratio, Bollinger Width, Z-score
- Trained inline during backtest (no lookahead bias)
- Rule-based fallback when ML unavailable
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional Hurst exponent import
try:
    from hurst import compute_Hc
    HAS_HURST = True
except ImportError:
    HAS_HURST = False
    logger.info("hurst package not installed - Hurst exponent will use fallback")

# Optional sklearn import with fallback
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not installed - ML filter will use rule-based fallback")

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("xgboost not installed - XGBoost model type unavailable")


@dataclass
class RegimeFeatures:
    """Feature set for regime classification at a single point in time."""
    adx: float
    choppiness: float
    efficiency_ratio: float
    bollinger_width: float
    zscore_abs: float
    volatility_ratio: float
    hurst_exponent: float = 0.5  # Default to random walk

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model input."""
        return np.array([
            self.adx,
            self.choppiness,
            self.efficiency_ratio,
            self.bollinger_width,
            self.zscore_abs,
            self.volatility_ratio,
            self.hurst_exponent
        ])


@dataclass
class MLCryptoMRSignal:
    """Signal state for ML Crypto Mean Reversion strategy."""
    timestamp: pd.Timestamp
    close: float
    zscore: float
    rsi: float
    atr: float
    is_ranging: bool
    regime_prob: float
    entry_long: bool
    entry_short: bool
    stop_long: float
    stop_short: float
    target_long: float
    target_short: float


class MLCryptoMRIndicators:
    """
    Indicator calculations for ML Crypto Mean Reversion strategy.

    Reuses existing indicators from src.backtesting.utils.indicators where
    possible and adds new ones (ADX, Choppiness, Efficiency Ratio).
    """

    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling Z-score for mean reversion detection.

        Z-score measures how many standard deviations price is from its mean.
        High absolute Z-score = price is extended = mean reversion opportunity.

        Args:
            series: Price series (typically close)
            window: Rolling window for mean/std (default 20)

        Returns:
            Z-score series
        """
        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        rolling_std = series.rolling(window=window, min_periods=window).std()
        zscore = (series - rolling_mean) / (rolling_std + 1e-10)
        return zscore

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI < 30 = oversold (confirms long entry)
        RSI > 70 = overbought (confirms short entry)

        Args:
            series: Price series
            period: RSI period (default 14)

        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for volatility-based stops.

        Args:
            df: OHLCV DataFrame
            period: ATR period (default 14)

        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength:
        - ADX > 25 = trending market (avoid mean reversion)
        - ADX < 20 = ranging market (good for mean reversion)

        Args:
            df: OHLCV DataFrame
            period: ADX period (default 14)

        Returns:
            ADX series (0-100)
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed values (Wilder's smoothing)
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
            span=period, adjust=False, min_periods=period
        ).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
            span=period, adjust=False, min_periods=period
        ).mean() / (atr + 1e-10)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()

        return adx

    @staticmethod
    def calculate_choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Choppiness Index.

        Measures whether market is trending or ranging:
        - CI > 61.8 = choppy/ranging (good for mean reversion)
        - CI < 38.2 = trending (avoid mean reversion)

        Formula: 100 * log10(sum(ATR, n) / (highest - lowest)) / log10(n)

        Args:
            df: OHLCV DataFrame
            period: Choppiness period (default 14)

        Returns:
            Choppiness Index series (0-100)
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Sum of True Range over period
        atr_sum = true_range.rolling(window=period, min_periods=period).sum()

        # Highest high - Lowest low over period
        highest = high.rolling(window=period, min_periods=period).max()
        lowest = low.rolling(window=period, min_periods=period).min()
        hl_range = highest - lowest

        # Choppiness Index
        choppiness = 100 * np.log10(atr_sum / (hl_range + 1e-10)) / np.log10(period)

        return choppiness

    @staticmethod
    def calculate_efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Kaufman's Efficiency Ratio.

        Measures trend efficiency (directional movement vs noise):
        - High ER (near 1) = strong trend
        - Low ER (near 0) = choppy/ranging (good for mean reversion)

        Formula: |Direction| / Volatility = |Close - Close[n]| / Sum(|Close - Close[1]|)

        Args:
            series: Price series
            period: Lookback period (default 10)

        Returns:
            Efficiency Ratio series (0-1)
        """
        # Direction: net price change over period
        direction = (series - series.shift(period)).abs()

        # Volatility: sum of absolute price changes
        volatility = series.diff().abs().rolling(window=period, min_periods=period).sum()

        # Efficiency Ratio
        efficiency = direction / (volatility + 1e-10)

        return efficiency

    @staticmethod
    def calculate_bollinger_width(
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Calculate Bollinger Band width as percentage of price.

        Narrow bands = low volatility consolidation
        Wide bands = high volatility trending

        Args:
            series: Price series
            window: Bollinger window (default 20)
            num_std: Standard deviations (default 2.0)

        Returns:
            Band width as fraction of price
        """
        middle = series.rolling(window=window, min_periods=window).mean()
        std = series.rolling(window=window, min_periods=window).std()

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        width = (upper - lower) / (middle + 1e-10)

        return width

    @staticmethod
    def calculate_hurst(series: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculate rolling Hurst exponent using R/S analysis.

        The Hurst exponent measures long-term memory/persistence in time series:
        - H < 0.5: Anti-persistent (mean-reverting) - GOOD for mean reversion trades
        - H = 0.5: Random walk (geometric Brownian motion)
        - H > 0.5: Persistent (trending) - AVOID mean reversion trades

        Args:
            series: Price series (will be converted to returns internally)
            window: Rolling window size (default 100 bars)

        Returns:
            Series of Hurst exponent values (0-1 range)
        """
        if not HAS_HURST:
            # Fallback: return 0.5 (random walk assumption)
            logger.warning("Hurst package not available, using 0.5 fallback")
            return pd.Series(0.5, index=series.index)

        # Calculate returns (Hurst is computed on returns, not prices)
        returns = series.pct_change().dropna()

        def calc_hurst_single(x: np.ndarray) -> float:
            """Calculate Hurst for a single window."""
            if len(x) < 20:  # Minimum for reliable estimate
                return 0.5
            try:
                # compute_Hc returns (H, c, data) tuple
                # kind='change' for returns, 'price' for cumulative series
                H, _, _ = compute_Hc(x, kind='change', simplified=True)
                # Clamp to valid range
                return max(0.0, min(1.0, H))
            except Exception:
                return 0.5

        # Rolling Hurst calculation
        hurst_values = returns.rolling(
            window=window,
            min_periods=max(20, window // 2)
        ).apply(calc_hurst_single, raw=True)

        # Reindex to match original series (first value is NaN due to pct_change)
        result = pd.Series(0.5, index=series.index)
        result.loc[hurst_values.index] = hurst_values

        return result

    @staticmethod
    def calculate_all_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate all indicators and features needed for the strategy.

        Args:
            df: OHLCV DataFrame
            config: Dictionary with indicator parameters

        Returns:
            DataFrame with all calculated features
        """
        close = df['close']

        features = pd.DataFrame(index=df.index)

        # Core indicators
        features['close'] = close
        features['zscore'] = MLCryptoMRIndicators.calculate_zscore(
            close, config.get('zscore_window', 20)
        )
        features['rsi'] = MLCryptoMRIndicators.calculate_rsi(
            close, config.get('rsi_period', 14)
        )
        features['atr'] = MLCryptoMRIndicators.calculate_atr(
            df, config.get('atr_period', 14)
        )

        # Regime detection features
        features['adx'] = MLCryptoMRIndicators.calculate_adx(
            df, config.get('adx_period', 14)
        )
        features['choppiness'] = MLCryptoMRIndicators.calculate_choppiness(
            df, config.get('choppiness_period', 14)
        )
        features['efficiency_ratio'] = MLCryptoMRIndicators.calculate_efficiency_ratio(
            close, config.get('efficiency_period', 10)
        )
        features['bollinger_width'] = MLCryptoMRIndicators.calculate_bollinger_width(
            close, config.get('bollinger_window', 20)
        )

        # Derived features
        features['zscore_abs'] = features['zscore'].abs()

        # Volatility ratio (current vs average)
        vol_window = config.get('volatility_window', 20)
        returns = close.pct_change()
        current_vol = returns.rolling(window=vol_window, min_periods=vol_window).std()
        avg_vol = current_vol.rolling(window=vol_window, min_periods=vol_window).mean()
        features['volatility_ratio'] = current_vol / (avg_vol + 1e-10)

        # Hurst exponent for mean reversion detection
        features['hurst_exponent'] = MLCryptoMRIndicators.calculate_hurst(
            close, config.get('hurst_window', 100)
        )

        return features


class MLRegimeFilter:
    """
    ML-based regime filter for identifying ranging vs trending conditions.

    The filter uses a GradientBoostingClassifier trained on historical data
    to predict whether current market conditions are favorable for mean
    reversion (ranging) or unfavorable (trending).

    Training is performed inline during backtest using a walk-forward approach
    to avoid lookahead bias.
    """

    # Feature columns for ML model (7 features including Hurst)
    FEATURE_COLS = [
        'adx', 'choppiness', 'efficiency_ratio',
        'bollinger_width', 'zscore_abs', 'volatility_ratio',
        'hurst_exponent'
    ]

    # Supported model types
    MODEL_TYPES = ['gradient_boosting', 'xgboost', 'random_forest', 'ensemble']

    def __init__(
        self,
        lookback_days: int = 252,
        retrain_frequency: int = 20,
        min_train_samples: int = 100,
        use_ml: bool = True,
        adx_threshold: float = 25.0,
        choppiness_threshold: float = 61.8,
        regime_prob_threshold: float = 0.6,
        model_type: str = 'gradient_boosting'
    ):
        """
        Initialize the regime filter.

        Args:
            lookback_days: Number of days of history for training
            retrain_frequency: Retrain model every N bars
            min_train_samples: Minimum samples required for training
            use_ml: If False, use rule-based filter only
            adx_threshold: ADX threshold for rule-based filter
            choppiness_threshold: Choppiness threshold for rule-based filter
            regime_prob_threshold: Probability threshold for ML classification
            model_type: ML model type - 'gradient_boosting', 'xgboost',
                       'random_forest', or 'ensemble' (XGB + RF voting)
        """
        self.lookback_days = lookback_days
        self.retrain_frequency = retrain_frequency
        self.min_train_samples = min_train_samples
        self.use_ml = use_ml and HAS_SKLEARN
        self.adx_threshold = adx_threshold
        self.choppiness_threshold = choppiness_threshold
        self.regime_prob_threshold = regime_prob_threshold
        self.model_type = model_type

        # Validate model type
        if model_type not in self.MODEL_TYPES:
            logger.warning(f"Unknown model_type '{model_type}', using 'gradient_boosting'")
            self.model_type = 'gradient_boosting'

        # Check XGBoost availability
        if model_type in ['xgboost', 'ensemble'] and not HAS_XGBOOST:
            logger.warning("XGBoost not installed, falling back to gradient_boosting")
            self.model_type = 'gradient_boosting'

        # Model state
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self._last_train_idx: int = -1
        self._is_trained: bool = False

    def _create_model(self) -> Any:
        """Create ML model based on model_type setting."""
        if self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_child_weight=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'ensemble':
            # Voting ensemble of XGBoost + Random Forest
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_child_weight=20,
                subsample=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )
            return VotingClassifier(
                estimators=[('xgb', xgb), ('rf', rf)],
                voting='soft'  # Use probability averaging
            )
        else:  # gradient_boosting (default)
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42
            )

    def _generate_labels(
        self,
        features: pd.DataFrame,
        forward_window: int = 10,
        return_threshold: float = 0.02
    ) -> pd.Series:
        """
        Generate training labels based on forward-looking returns.

        Ranging (1): Absolute forward return < threshold (mean reversion worked)
        Trending (0): Absolute forward return >= threshold (trend continued)

        NOTE: This uses future data - only for training!

        Args:
            features: DataFrame with 'close' column
            forward_window: Bars to look forward
            return_threshold: Threshold for ranging classification

        Returns:
            Binary labels (1 = ranging, 0 = trending)
        """
        close = features['close']

        # Forward return
        forward_return = close.shift(-forward_window) / close - 1

        # Ranging: small absolute forward return
        labels = (forward_return.abs() < return_threshold).astype(int)

        return labels

    def fit(
        self,
        features: pd.DataFrame,
        end_idx: int = None
    ) -> Dict[str, Any]:
        """
        Train the ML model on historical data.

        Args:
            features: DataFrame with feature columns
            end_idx: Train only on data up to this index (for walk-forward)

        Returns:
            Dictionary with training statistics
        """
        if not self.use_ml or not HAS_SKLEARN:
            return {'method': 'rule_based', 'trained': False}

        # Select training data
        if end_idx is not None:
            train_start = max(0, end_idx - self.lookback_days)
            train_features = features.iloc[train_start:end_idx].copy()
        else:
            train_features = features.copy()

        # Generate labels
        labels = self._generate_labels(train_features)

        # Align features and labels (drop NaN)
        X = train_features[self.FEATURE_COLS].copy()
        y = labels

        # Drop rows with NaN in features or labels
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < self.min_train_samples:
            logger.warning(
                f"Insufficient training samples: {len(X)} < {self.min_train_samples}"
            )
            return {'method': 'rule_based', 'trained': False, 'samples': len(X)}

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self._is_trained = True

        # Feature importance (handle ensemble case)
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = pd.Series(
                    self.model.feature_importances_,
                    index=self.FEATURE_COLS
                ).sort_values(ascending=False)
            elif hasattr(self.model, 'estimators_'):
                # VotingClassifier - average feature importance from estimators
                importances = []
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    importance = pd.Series(
                        avg_importance,
                        index=self.FEATURE_COLS
                    ).sort_values(ascending=False)
                else:
                    importance = pd.Series(dtype=float)
            else:
                importance = pd.Series(dtype=float)
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            importance = pd.Series(dtype=float)

        return {
            'method': 'ml',
            'trained': True,
            'samples': len(X),
            'class_balance': y.value_counts().to_dict(),
            'feature_importance': importance.to_dict()
        }

    def predict(
        self,
        features: pd.DataFrame,
        current_idx: int = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict regime for each bar.

        Args:
            features: DataFrame with feature columns
            current_idx: Current bar index (for retraining check)

        Returns:
            Tuple of (is_ranging, prob_ranging) series
        """
        n = len(features)

        # Initialize output
        is_ranging = pd.Series(False, index=features.index)
        prob_ranging = pd.Series(0.5, index=features.index)

        # Check if retraining needed
        if current_idx is not None and self.use_ml:
            if (current_idx - self._last_train_idx) >= self.retrain_frequency:
                self.fit(features, end_idx=current_idx)
                self._last_train_idx = current_idx

        # Use ML or rule-based prediction
        if self._is_trained and self.use_ml:
            return self._predict_ml(features)
        else:
            return self._predict_rules(features)

    def _predict_ml(self, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """ML-based prediction."""
        X = features[self.FEATURE_COLS].copy()

        # Find valid rows
        valid_mask = X.notna().all(axis=1)
        valid_idx = features.index[valid_mask]

        # Initialize output
        is_ranging = pd.Series(False, index=features.index)
        prob_ranging = pd.Series(0.5, index=features.index)

        if len(valid_idx) == 0:
            return is_ranging, prob_ranging

        # Scale and predict
        X_valid = self.scaler.transform(X.loc[valid_idx])
        probs = self.model.predict_proba(X_valid)

        # Probability of ranging (class 1)
        prob_ranging.loc[valid_idx] = probs[:, 1]
        is_ranging.loc[valid_idx] = prob_ranging.loc[valid_idx] >= self.regime_prob_threshold

        return is_ranging, prob_ranging

    def _predict_rules(self, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Rule-based regime prediction (fallback).

        Ranging: ADX < threshold AND Choppiness > threshold
        """
        adx = features.get('adx', pd.Series(25.0, index=features.index))
        chop = features.get('choppiness', pd.Series(50.0, index=features.index))

        # Ranging condition
        is_ranging = (adx < self.adx_threshold) & (chop > self.choppiness_threshold)

        # Approximate probability based on ADX
        prob_ranging = 1 - (adx / 50).clip(0, 1)

        return is_ranging, prob_ranging

    def predict_single(self, features: RegimeFeatures) -> Tuple[bool, float]:
        """
        Predict regime for a single bar (for live trading).

        Args:
            features: RegimeFeatures dataclass

        Returns:
            Tuple of (is_ranging, prob_ranging)
        """
        if self._is_trained and self.use_ml:
            X = features.to_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)
            prob_ranging = probs[0, 1]
            is_ranging = prob_ranging >= self.regime_prob_threshold
            return is_ranging, prob_ranging
        else:
            # Rule-based
            is_ranging = (
                features.adx < self.adx_threshold and
                features.choppiness > self.choppiness_threshold
            )
            prob_ranging = 1 - (features.adx / 50)
            return is_ranging, max(0, min(1, prob_ranging))

    def reset(self) -> None:
        """Reset model state for new backtest run."""
        self.model = None
        self.scaler = None
        self._last_train_idx = -1
        self._is_trained = False
