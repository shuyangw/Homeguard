#!/usr/bin/env python3
"""
Test signal generation for today's market data.
Used to validate what signals would have been generated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, time
import pytz
import yaml
import yfinance as yf

from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.utils.logger import logger


def main():
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config/trading/omr_trading_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    symbols = config['strategy']['symbols']
    min_probability = config['strategy']['min_win_rate']
    min_expected_return = config['strategy']['min_expected_return']
    min_samples = config['strategy']['min_sample_size']

    logger.header(f"Testing signal generation for {datetime.now().strftime('%Y-%m-%d')}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Min win rate: {min_probability:.0%}")
    logger.info(f"Min expected return: {min_expected_return:.2%}")
    logger.info(f"Min samples: {min_samples}")
    logger.separator()

    # Initialize components
    bayesian_model = BayesianReversionModel()
    regime_detector = MarketRegimeDetector()

    logger.info(f"Bayesian model loaded: {len(bayesian_model.regime_probabilities)} symbols")

    # Fetch SPY and VIX for regime detection
    logger.info("Fetching market data...")
    spy = yf.download('SPY', period='60d', progress=False)
    vix = yf.download('^VIX', period='60d', progress=False)

    if spy.empty or vix.empty:
        logger.error("Could not fetch SPY/VIX data")
        return 1

    # Normalize columns
    spy.columns = [c[0].lower() if hasattr(c, '__iter__') and not isinstance(c, str) else c.lower() for c in spy.columns]
    vix.columns = [c[0].lower() if hasattr(c, '__iter__') and not isinstance(c, str) else c.lower() for c in vix.columns]

    # Detect regime
    regime = regime_detector.detect_regime(spy, vix)
    logger.info(f"Current market regime: {regime}")

    # Fetch intraday data for symbols and check signals
    logger.info("Fetching intraday data and checking signals...")
    signals = []
    et = pytz.timezone('America/New_York')

    for symbol in symbols:
        try:
            # Get today's intraday data
            df = yf.download(symbol, period='1d', interval='1m', progress=False)
            if df.empty:
                logger.debug(f"  {symbol}: No intraday data")
                continue

            # Normalize columns
            df.columns = [c[0].lower() if hasattr(c, '__iter__') and not isinstance(c, str) else c.lower() for c in df.columns]

            # Get open price (around 9:30 AM)
            df_et = df.copy()
            if df_et.index.tz is None:
                df_et.index = df_et.index.tz_localize('UTC').tz_convert(et)
            else:
                df_et.index = df_et.index.tz_convert(et)

            open_data = df_et.between_time(time(9, 30), time(9, 35))
            if open_data.empty:
                logger.debug(f"  {symbol}: No 9:30 AM data")
                continue
            open_price = float(open_data['open'].iloc[0])

            # Get 3:50 PM price
            close_data = df_et.between_time(time(15, 45), time(15, 55))
            if close_data.empty:
                logger.debug(f"  {symbol}: No 3:50 PM data")
                continue
            current_price = float(close_data['close'].iloc[-1])

            # Calculate intraday return
            intraday_return = (current_price - open_price) / open_price

            # Get Bayesian probability
            prob_data = bayesian_model.get_reversion_probability(symbol, regime, intraday_return)

            if prob_data is None:
                logger.info(f"  {symbol}: No prob data | move={intraday_return:+.2%}")
                continue

            prob = prob_data['probability']
            exp_ret = prob_data['expected_return']
            samples = prob_data['sample_size']

            # Check filters
            passed = True
            reasons = []
            if prob < min_probability:
                passed = False
                reasons.append(f"prob {prob:.1%} < {min_probability:.0%}")
            if exp_ret < min_expected_return:
                passed = False
                reasons.append(f"exp_ret {exp_ret:.2%} < {min_expected_return:.2%}")
            if samples < min_samples:
                passed = False
                reasons.append(f"samples {samples} < {min_samples}")
            if abs(intraday_return) < 0.01:
                passed = False
                reasons.append(f"move {abs(intraday_return):.2%} < 1%")

            if passed:
                signals.append({'symbol': symbol, 'prob': prob, 'exp_ret': exp_ret, 'move': intraday_return})
                logger.success(f"  {symbol}: PASS | move={intraday_return:+.2%} | prob={prob:.1%} | exp_ret={exp_ret:.2%} | n={samples}")
            else:
                reason_str = ", ".join(reasons)
                logger.info(f"  {symbol}: REJECT | move={intraday_return:+.2%} | {reason_str}")

        except Exception as e:
            logger.error(f"  {symbol}: ERROR - {e}")

    logger.separator()
    if signals:
        logger.success(f"RESULT: {len(signals)} signals would have been generated")
        for s in signals:
            side = 'BUY' if s['move'] < 0 else 'SELL'
            logger.info(f"  {s['symbol']}: {side} | prob={s['prob']:.1%} | exp_ret={s['exp_ret']:.2%}")
    else:
        logger.warning("RESULT: 0 signals would have been generated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
