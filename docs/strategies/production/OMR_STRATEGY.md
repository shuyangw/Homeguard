# Regime-Based Overnight Mean Reversion Strategy Implementation

## Strategy Overview

This strategy exploits predictable overnight mean reversion patterns in leveraged ETFs, with signals varying based on market regime classification.

### Key Components

1. **Market Regime Classification** (5 regimes)
2. **Leveraged ETF Universe** (high volatility instruments)
3. **Bayesian Probability Model** (10 years historical data)
4. **Overnight Holding Period** (3:50 PM -> Market Open)

---

## Implementation Architecture

### 1. Market Regime Detector

```python
class MarketRegimeDetector:
    """
    Classifies market into one of 5 regimes based on momentum and volatility
    """

    REGIMES = {
        'STRONG_BULL': 1,
        'WEAK_BULL': 2,
        'SIDEWAYS': 3,
        'UNPREDICTABLE': 4,
        'BEAR': 5
    }

    def __init__(self):
        self.momentum_periods = [20, 50, 200]  # SMA periods for SPY
        self.volatility_window = 20

    def classify_regime(self, spy_data, vix_data, timestamp):
        """
        Classify current market regime

        Returns:
            regime: One of 5 regime classifications
            confidence: Confidence score (0-1)
        """
        # Calculate momentum indicators
        sma_20 = spy_data['close'].rolling(20).mean()
        sma_50 = spy_data['close'].rolling(50).mean()
        sma_200 = spy_data['close'].rolling(200).mean()

        # Price relative to moving averages
        price = spy_data['close'].iloc[-1]
        above_20 = price > sma_20.iloc[-1]
        above_50 = price > sma_50.iloc[-1]
        above_200 = price > sma_200.iloc[-1]

        # Momentum slope
        momentum_slope = (sma_20.iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20]

        # Volatility metrics
        current_vix = vix_data['close'].iloc[-1]
        vix_percentile = self._calculate_vix_percentile(vix_data, current_vix)

        # Regime classification logic
        if above_20 and above_50 and above_200:
            if momentum_slope > 0.02 and vix_percentile < 30:
                return 'STRONG_BULL', 0.85
            else:
                return 'WEAK_BULL', 0.75

        elif not above_20 and not above_50 and not above_200:
            if momentum_slope < -0.02 and vix_percentile > 70:
                return 'BEAR', 0.85
            else:
                return 'UNPREDICTABLE', 0.60

        else:
            if vix_percentile > 60:
                return 'UNPREDICTABLE', 0.65
            else:
                return 'SIDEWAYS', 0.70
```

### 2. Leveraged ETF Universe

```python
LEVERAGED_ETF_UNIVERSE = {
    # 3x Leveraged Long ETFs
    'TQQQ': {'leverage': 3, 'underlying': 'QQQ', 'direction': 'long'},
    'UPRO': {'leverage': 3, 'underlying': 'SPY', 'direction': 'long'},
    'UDOW': {'leverage': 3, 'underlying': 'DIA', 'direction': 'long'},
    'TNA': {'leverage': 3, 'underlying': 'IWM', 'direction': 'long'},
    'SOXL': {'leverage': 3, 'underlying': 'SOX', 'direction': 'long'},
    'FAS': {'leverage': 3, 'underlying': 'XLF', 'direction': 'long'},
    'LABU': {'leverage': 3, 'underlying': 'XBI', 'direction': 'long'},
    'TECL': {'leverage': 3, 'underlying': 'XLK', 'direction': 'long'},

    # 3x Leveraged Short ETFs
    'SQQQ': {'leverage': -3, 'underlying': 'QQQ', 'direction': 'short'},
    'SPXU': {'leverage': -3, 'underlying': 'SPY', 'direction': 'short'},
    'SDOW': {'leverage': -3, 'underlying': 'DIA', 'direction': 'short'},
    'TZA': {'leverage': -3, 'underlying': 'IWM', 'direction': 'short'},
    'SOXS': {'leverage': -3, 'underlying': 'SOX', 'direction': 'short'},
    'FAZ': {'leverage': -3, 'underlying': 'XLF', 'direction': 'short'},
    'LABD': {'leverage': -3, 'underlying': 'XBI', 'direction': 'short'},
    'TECS': {'leverage': -3, 'underlying': 'XLK', 'direction': 'short'},

    # 2x Leveraged ETFs
    'QLD': {'leverage': 2, 'underlying': 'QQQ', 'direction': 'long'},
    'SSO': {'leverage': 2, 'underlying': 'SPY', 'direction': 'long'},
    'QID': {'leverage': -2, 'underlying': 'QQQ', 'direction': 'short'},
    'SDS': {'leverage': -2, 'underlying': 'SPY', 'direction': 'short'},

    # Volatility ETFs
    'UVXY': {'leverage': 1.5, 'underlying': 'VIX', 'direction': 'long'},
    'SVXY': {'leverage': -0.5, 'underlying': 'VIX', 'direction': 'short'},
    'VIXY': {'leverage': 1, 'underlying': 'VIX', 'direction': 'long'},
}
```

### 3. Bayesian Probability Model

```python
class BayesianReversionModel:
    """
    Calculates probability of overnight reversion based on historical patterns
    """

    def __init__(self, lookback_years=10):
        self.lookback_years = lookback_years
        self.regime_probabilities = {}
        self.trained = False

    def train(self, historical_data):
        """
        Train on 10 years of historical data
        Calculate P(overnight_reversion | regime, intraday_move)
        """
        for symbol in LEVERAGED_ETF_UNIVERSE.keys():
            self.regime_probabilities[symbol] = {
                'STRONG_BULL': {},
                'WEAK_BULL': {},
                'SIDEWAYS': {},
                'UNPREDICTABLE': {},
                'BEAR': {}
            }

            # For each regime
            for regime in self.regime_probabilities[symbol].keys():
                # Calculate probabilities for different intraday move buckets
                for move_bucket in self._get_move_buckets():
                    prob = self._calculate_reversion_probability(
                        historical_data[symbol],
                        regime,
                        move_bucket
                    )
                    self.regime_probabilities[symbol][regime][move_bucket] = prob

        self.trained = True

    def _calculate_reversion_probability(self, data, regime, move_bucket):
        """
        Calculate P(profitable_overnight | regime, intraday_move_bucket)

        Returns:
            dict with probability, expected_return, sample_size
        """
        regime_data = data[data['regime'] == regime]
        bucket_data = regime_data[
            (regime_data['intraday_return'] >= move_bucket['min']) &
            (regime_data['intraday_return'] < move_bucket['max'])
        ]

        if len(bucket_data) < 20:  # Minimum sample size
            return None

        # Calculate win rate
        profitable = bucket_data['overnight_return'] > 0.001  # 0.1% threshold
        win_rate = profitable.sum() / len(profitable)

        # Calculate expected return
        expected_return = bucket_data['overnight_return'].mean()

        # Calculate confidence based on sample size
        sample_size = len(bucket_data)
        confidence = min(sample_size / 100, 1.0)  # Max confidence at 100 samples

        return {
            'probability': win_rate,
            'expected_return': expected_return,
            'sample_size': sample_size,
            'confidence': confidence,
            'sharpe': bucket_data['overnight_return'].mean() / bucket_data['overnight_return'].std()
        }

    def _get_move_buckets(self):
        """Define intraday move buckets for analysis"""
        return [
            {'min': -1.0, 'max': -0.05, 'label': 'large_down'},
            {'min': -0.05, 'max': -0.03, 'label': 'medium_down'},
            {'min': -0.03, 'max': -0.01, 'label': 'small_down'},
            {'min': -0.01, 'max': 0.01, 'label': 'flat'},
            {'min': 0.01, 'max': 0.03, 'label': 'small_up'},
            {'min': 0.03, 'max': 0.05, 'label': 'medium_up'},
            {'min': 0.05, 'max': 1.0, 'label': 'large_up'}
        ]
```

### 4. Signal Generator

```python
class OvernightReversionSignals:
    """
    Generates trading signals at 3:50 PM based on regime and probabilities
    """

    def __init__(self, regime_detector, bayesian_model, min_probability=0.55, min_expected_return=0.002):
        self.regime_detector = regime_detector
        self.bayesian_model = bayesian_model
        self.min_probability = min_probability
        self.min_expected_return = min_expected_return

    def generate_signals(self, market_data, timestamp):
        """
        Generate buy signals for overnight holding

        Returns:
            List of (symbol, confidence, expected_return) tuples
        """
        # Get current regime
        regime, regime_confidence = self.regime_detector.classify_regime(
            market_data['SPY'],
            market_data['VIX'],
            timestamp
        )

        signals = []

        # Check each leveraged ETF
        for symbol in LEVERAGED_ETF_UNIVERSE.keys():
            if symbol not in market_data:
                continue

            # Calculate intraday return
            data = market_data[symbol]
            intraday_return = (data['close'].iloc[-1] - data['open'].iloc[-1]) / data['open'].iloc[-1]

            # Get probability of overnight reversion
            move_bucket = self._get_move_bucket(intraday_return)
            prob_data = self.bayesian_model.regime_probabilities[symbol][regime].get(move_bucket)

            if prob_data is None:
                continue

            # Apply filters
            if (prob_data['probability'] > self.min_probability and
                prob_data['expected_return'] > self.min_expected_return and
                prob_data['sample_size'] >= 30):

                # Calculate signal strength
                signal_strength = (
                    prob_data['probability'] * 0.4 +
                    min(prob_data['expected_return'] / 0.01, 1.0) * 0.3 +
                    regime_confidence * 0.3
                )

                signals.append({
                    'symbol': symbol,
                    'regime': regime,
                    'intraday_return': intraday_return,
                    'probability': prob_data['probability'],
                    'expected_return': prob_data['expected_return'],
                    'signal_strength': signal_strength,
                    'sample_size': prob_data['sample_size']
                })

        # Sort by signal strength
        signals.sort(key=lambda x: x['signal_strength'], reverse=True)

        return signals[:5]  # Top 5 signals
```

### 5. Strategy Implementation

```python
class OvernightMeanReversionStrategy(BaseStrategy):
    """
    Main strategy implementation for Homeguard framework
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.regime_detector = MarketRegimeDetector()
        self.bayesian_model = BayesianReversionModel()
        self.signal_generator = OvernightReversionSignals(
            self.regime_detector,
            self.bayesian_model,
            min_probability=params.get('min_probability', 0.55),
            min_expected_return=params.get('min_expected_return', 0.002)
        )

    def train_model(self, historical_data):
        """Train Bayesian model on historical data"""
        logger.info("Training Bayesian reversion model on 10 years of data...")

        # Add regime labels to historical data
        for date in historical_data.index:
            regime, _ = self.regime_detector.classify_regime(
                historical_data.loc[:date],
                historical_data.loc[:date],
                date
            )
            historical_data.loc[date, 'regime'] = regime

        # Train model
        self.bayesian_model.train(historical_data)
        logger.success("Model training complete")

    def generate_signals(self, data):
        """Generate trading signals at 3:50 PM"""
        current_time = data.index[-1]

        # Only generate signals at 3:50 PM EST
        if current_time.hour != 15 or current_time.minute != 50:
            return pd.DataFrame()

        signals = self.signal_generator.generate_signals(data, current_time)

        # Convert to DataFrame for Homeguard
        if signals:
            df = pd.DataFrame(signals)
            df['action'] = 'BUY'
            df['position_size'] = 1.0 / len(signals)  # Equal weight
            return df
        else:
            return pd.DataFrame()

    def should_exit(self, position, current_data):
        """Exit all positions at market open"""
        current_time = current_data.index[-1]

        # Exit at 9:31 AM EST (1 minute after open)
        if current_time.hour == 9 and current_time.minute >= 31:
            return True

        return False
```

---

## Implementation Steps

### Phase 1: Data Collection (Week 1)

1. **Download Leveraged ETF Data**
   ```python
   symbols = list(LEVERAGED_ETF_UNIVERSE.keys())
   # Add SPY and VIX for regime detection
   symbols.extend(['SPY', 'VIX', 'VXX'])

   # Download 10 years of minute data
   start_date = '2015-01-01'
   end_date = '2025-01-01'
   ```

2. **Calculate Historical Features**
   - Intraday returns (open to 3:50 PM)
   - Overnight returns (3:50 PM to next open)
   - Regime classifications for each day
   - Volume patterns

### Phase 2: Model Training (Week 2)

1. **Train Regime Detector**
   - Validate regime classifications
   - Tune thresholds for each regime
   - Test stability across different periods

2. **Train Bayesian Model**
   - Calculate probabilities for each (symbol, regime, move_bucket) combination
   - Require minimum 30 samples for statistical significance
   - Calculate expected returns and Sharpe ratios

3. **Backtest Validation**
   - Walk-forward validation
   - Out-of-sample testing on 2024 data
   - Compare to reported results

### Phase 3: Signal Generation (Week 3)

1. **Implement Real-time Signal Generator**
   - Connect to live data feed
   - Generate signals at 3:50 PM EST daily
   - Rank by probability and expected return

2. **Risk Management**
   ```python
   RISK_PARAMETERS = {
       'max_positions': 5,
       'position_size': 0.2,  # 20% per position
       'max_correlation': 0.7,  # Between positions
       'stop_loss': None,  # No intraday stops
       'regime_limits': {
           'STRONG_BULL': 3,  # Max positions per regime
           'WEAK_BULL': 5,
           'SIDEWAYS': 4,
           'UNPREDICTABLE': 2,
           'BEAR': 5
       }
   }
   ```

### Phase 4: Paper Trading (Week 4)

1. **Setup Paper Trading**
   - Alpaca paper account
   - Automated execution at 3:50 PM and 9:31 AM
   - Track slippage and execution quality

2. **Performance Monitoring**
   - Track win rate by regime
   - Monitor expected vs actual returns
   - Analyze regime classification accuracy

---

## Advantages Over Pairs Trading

1. **Single-leg trades** - No complex hedge ratios or pair relationships
2. **High frequency** - Trade opportunities almost daily
3. **Regime-adaptive** - Works across different market conditions
4. **Lower capital requirements** - Can trade with smaller account
5. **Clear entry/exit** - Fixed times, no discretion needed
6. **Proven edge** - Overnight anomaly well-documented

---

## Risk Considerations

1. **Overnight gaps** - Risk of adverse news/events
2. **Leveraged decay** - Long-term holding would be problematic
3. **Regime misclassification** - Wrong regime = wrong probabilities
4. **Market structure changes** - Pattern may weaken over time
5. **Competition** - More traders exploiting same edge

---

## Expected Performance

Based on reported results and typical overnight reversion patterns:

- **Annual Return**: 40-60%
- **Sharpe Ratio**: 2.5-3.5
- **Win Rate**: 60-65%
- **Max Drawdown**: 10-15%
- **Correlation to Market**: 0.1-0.3

---

## Next Steps

1. **Immediate**: Start downloading leveraged ETF data
2. **Week 1**: Build regime detector and validate classifications
3. **Week 2**: Train Bayesian model on historical data
4. **Week 3**: Backtest and validate results
5. **Week 4**: Begin paper trading

This strategy is much more suitable for current market conditions than pairs trading!