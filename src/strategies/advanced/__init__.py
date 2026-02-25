"""
Advanced Production Trading Strategies.

This module contains production-deployed strategies:
- OMR (Overnight Mean Reversion): overnight_mean_reversion.py
- MP (Momentum Protection): momentum_protection_strategy.py
- RAMP (Regime-Aware Momentum Protection): ramp_strategy.py
- ORB (Opening Range Breakout): orb_strategy.py
- DSTS (Dual Signal Trend Sentinel): dsts_strategy.py
- FRS (Fractal Regime Switching): frs_strategy.py

Supporting modules:
- bayesian_reversion_model.py: ML model for OMR probability estimation
- overnight_signal_generator.py: Signal generation for OMR
- market_regime_detector.py: Regime classification for RAMP/ORB
- orb_indicators.py: Opening range and RVOL calculations for ORB
- dsts_indicators.py: Z-score calculations for DSTS
- frs_indicators.py: Hurst exponent and regime routing for FRS

Research strategies have been moved to src/strategies/research/.
"""

from src.strategies.advanced.orb_strategy import ORBStrategy
from src.strategies.advanced.orb_indicators import ORBIndicators
from src.strategies.advanced.dsts_strategy import DSTSStrategy
from src.strategies.advanced.dsts_indicators import DSTSIndicators
from src.strategies.advanced.dsts_signals import DSTSSignals
from src.strategies.advanced.frs_strategy import FRSStrategy
from src.strategies.advanced.frs_indicators import FRSIndicators, FRSRegime, FRSSignal

__all__ = [
    'ORBStrategy',
    'ORBIndicators',
    'DSTSStrategy',
    'DSTSIndicators',
    'DSTSSignals',
    'FRSStrategy',
    'FRSIndicators',
    'FRSRegime',
    'FRSSignal',
]
