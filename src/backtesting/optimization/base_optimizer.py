"""
Base optimizer class for all parameter optimization methods.

Provides common infrastructure:
- Result caching (Phase 3)
- Parallel execution (Phase 1)
- Progress tracking (Phase 2)
- CSV export (Phase 2)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backtesting.engine.backtest_engine import BacktestEngine


@dataclass
class _EngineConfig:
    """Configuration for backtest engine (pickleable for multiprocessing)."""
    initial_capital: float
    fees: float
    slippage: float
    freq: str
    market_hours_only: bool
    risk_config_dict: Optional[Dict[str, Any]]
    enable_regime_analysis: bool


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization methods.

    Subclasses implement different optimization algorithms:
    - GridSearchOptimizer: Exhaustive search (existing)
    - RandomSearchOptimizer: Random sampling (Phase 4a)
    - BayesianOptimizer: Gaussian process (Phase 4b)
    - GeneticOptimizer: Evolutionary algorithm (Phase 4c)

    All optimizers share:
    - Result caching (from Phase 3)
    - Parallel execution (from Phase 1)
    - Progress tracking (from Phase 2)
    - CSV export (from Phase 2)
    """

    def __init__(self, engine: 'BacktestEngine'):
        """
        Initialize optimizer.

        Args:
            engine: BacktestEngine instance to use for running backtests
        """
        self.engine = engine
        self._prepare_engine_config()

    @abstractmethod
    def optimize(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization to find best parameters.

        Each subclass implements this with their specific algorithm.

        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space definition (format varies by optimizer)
            symbols: Symbol or list of symbols
            start_date: Start date for backtest period
            end_date: End date for backtest period
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            **kwargs: Additional optimizer-specific parameters

        Returns:
            Dictionary with optimization results
        """
        pass

    def _prepare_engine_config(self) -> _EngineConfig:
        """Prepare pickleable engine configuration for multiprocessing."""
        # Convert risk config to dict for pickling
        risk_dict = None
        if self.engine.risk_config:
            risk_dict = {
                'position_sizing_method': self.engine.risk_config.position_sizing_method,
                'position_size_pct': self.engine.risk_config.position_size_pct,
                'use_stop_loss': self.engine.risk_config.use_stop_loss,
                'stop_loss_type': self.engine.risk_config.stop_loss_type,
                'stop_loss_pct': self.engine.risk_config.stop_loss_pct,
                'atr_multiplier': self.engine.risk_config.atr_multiplier,
                'atr_lookback': self.engine.risk_config.atr_lookback,
                'max_holding_bars': self.engine.risk_config.max_holding_bars,
                'take_profit_pct': self.engine.risk_config.take_profit_pct,
                'max_positions': self.engine.risk_config.max_positions,
                'max_single_position_pct': self.engine.risk_config.max_single_position_pct,
                'max_portfolio_heat': self.engine.risk_config.max_portfolio_heat,
                'risk_per_trade_pct': self.engine.risk_config.risk_per_trade_pct,
                'kelly_win_rate': self.engine.risk_config.kelly_win_rate,
            }

        self._engine_config = _EngineConfig(
            initial_capital=self.engine.initial_capital,
            fees=self.engine.fees,
            slippage=self.engine.slippage,
            freq=self.engine.freq,
            market_hours_only=self.engine.market_hours_only,
            risk_config_dict=risk_dict,
            enable_regime_analysis=self.engine.enable_regime_analysis
        )
        return self._engine_config

    def _is_better(self, value: float, best_value: float, metric: str) -> bool:
        """
        Determine if a value is better than the current best.

        Args:
            value: New value to compare
            best_value: Current best value
            metric: Metric being optimized

        Returns:
            True if value is better than best_value
        """
        if metric == 'max_drawdown':
            # For drawdown, smaller absolute value is better
            return value < best_value
        else:
            # For other metrics, larger is better
            return value > best_value

    def _extract_metric_value(self, stats: Dict[str, Any], metric: str) -> float:
        """
        Extract metric value from portfolio statistics.

        Args:
            stats: Portfolio statistics dictionary
            metric: Metric name ('sharpe_ratio', 'total_return', 'max_drawdown')

        Returns:
            Metric value as float

        Raises:
            ValueError: If unknown metric is specified
        """
        if metric == 'sharpe_ratio':
            return float(stats.get('Sharpe Ratio', float('-inf')))  # type: ignore[arg-type]
        elif metric == 'total_return':
            return float(stats.get('Total Return [%]', float('-inf')))  # type: ignore[arg-type]
        elif metric == 'max_drawdown':
            return float(stats.get('Max Drawdown [%]', float('inf')))  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown metric: {metric}")
