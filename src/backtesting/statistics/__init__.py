"""Statistical methodology per docs/methodology/backtesting.md Section 2.

Implements:
- sharpe        (Section 2.1)
- psr           (Section 2.2)  Probabilistic Sharpe Ratio
- expected_max_sharpe / dsr  (Section 2.3)  Deflated Sharpe Ratio
- pbo           (Section 2.4)  Probability of Backtest Overfitting via CSCV

These functions are the project's single source of truth for these metrics.
Callers must use them rather than reimplementing.
"""
from src.backtesting.statistics.sharpe import sharpe
from src.backtesting.statistics.psr import psr
from src.backtesting.statistics.dsr import dsr, expected_max_sharpe
from src.backtesting.statistics.pbo import pbo

__all__ = ["sharpe", "psr", "dsr", "expected_max_sharpe", "pbo"]
