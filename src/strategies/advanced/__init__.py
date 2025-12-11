"""
Advanced Production Trading Strategies.

This module contains production-deployed strategies:
- OMR (Overnight Mean Reversion): overnight_mean_reversion.py
- MP (Momentum Protection): momentum_protection_strategy.py
- RAMP (Regime-Aware Momentum Protection): ramp_strategy.py
- ORB (Opening Range Breakout): orb_strategy.py

Supporting modules:
- bayesian_reversion_model.py: ML model for OMR probability estimation
- overnight_signal_generator.py: Signal generation for OMR
- market_regime_detector.py: Regime classification for RAMP/ORB
- orb_indicators.py: Opening range and RVOL calculations for ORB

Research strategies have been moved to src/strategies/research/.
"""

from src.strategies.advanced.orb_strategy import ORBStrategy
from src.strategies.advanced.orb_indicators import ORBIndicators

__all__ = ['ORBStrategy', 'ORBIndicators']
