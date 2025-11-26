"""
Tests for config schema validation (Pydantic models).
"""

import pytest
from pydantic import ValidationError

from src.config.schema import (
    BacktestConfig,
    BacktestMode,
    PositionSizingMethod,
    StrategyConfig,
    SymbolsConfig,
    DatesConfig,
    BacktestSettings,
    RiskSettings,
    SweepSettings,
    OptimizationSettings,
    WalkForwardSettings,
    OutputSettings,
)


class TestBacktestMode:
    """Tests for BacktestMode enum."""

    def test_valid_modes(self):
        """Test all valid mode values."""
        assert BacktestMode.SINGLE.value == "single"
        assert BacktestMode.SWEEP.value == "sweep"
        assert BacktestMode.OPTIMIZE.value == "optimize"
        assert BacktestMode.WALK_FORWARD.value == "walk_forward"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert BacktestMode("single") == BacktestMode.SINGLE
        assert BacktestMode("sweep") == BacktestMode.SWEEP
        assert BacktestMode("optimize") == BacktestMode.OPTIMIZE
        assert BacktestMode("walk_forward") == BacktestMode.WALK_FORWARD


class TestPositionSizingMethod:
    """Tests for PositionSizingMethod enum."""

    def test_valid_methods(self):
        """Test all valid position sizing methods."""
        assert PositionSizingMethod.FIXED_PERCENT.value == "fixed_percent"
        assert PositionSizingMethod.KELLY.value == "kelly"
        assert PositionSizingMethod.VOLATILITY_TARGET.value == "volatility_target"
        assert PositionSizingMethod.EQUAL_WEIGHT.value == "equal_weight"


class TestStrategyConfig:
    """Tests for StrategyConfig model."""

    def test_valid_strategy_config(self):
        """Test valid strategy configuration."""
        config = StrategyConfig(
            name="MovingAverageCrossover",
            parameters={"fast_window": 10, "slow_window": 50}
        )
        assert config.name == "MovingAverageCrossover"
        assert config.parameters["fast_window"] == 10

    def test_strategy_config_default_params(self):
        """Test strategy config with default empty parameters."""
        config = StrategyConfig(name="MeanReversion")
        assert config.name == "MeanReversion"
        assert config.parameters == {}

    def test_strategy_config_requires_name(self):
        """Test that strategy name is required."""
        with pytest.raises(ValidationError):
            StrategyConfig(parameters={"foo": "bar"})


class TestSymbolsConfig:
    """Tests for SymbolsConfig model."""

    def test_symbols_list(self):
        """Test symbols from list."""
        config = SymbolsConfig(list=["AAPL", "MSFT", "GOOGL"])
        assert config.list == ["AAPL", "MSFT", "GOOGL"]
        assert config.universe is None
        assert config.file is None

    def test_symbols_universe(self):
        """Test symbols from universe reference."""
        config = SymbolsConfig(universe="technology.mag7")
        assert config.universe == "technology.mag7"
        assert config.list is None

    def test_symbols_file(self):
        """Test symbols from file path."""
        config = SymbolsConfig(file="data/symbols.txt")
        assert config.file == "data/symbols.txt"
        assert config.list is None

    def test_symbols_requires_one_source(self):
        """Test that at least one source is required."""
        with pytest.raises(ValidationError):
            SymbolsConfig()

    def test_symbols_only_one_source(self):
        """Test that only one source is allowed."""
        with pytest.raises(ValidationError):
            SymbolsConfig(list=["AAPL"], universe="technology.mag7")


class TestDatesConfig:
    """Tests for DatesConfig model."""

    def test_explicit_dates(self):
        """Test explicit start and end dates."""
        config = DatesConfig(start="2023-01-01", end="2024-01-01")
        assert config.start == "2023-01-01"
        assert config.end == "2024-01-01"
        assert config.preset is None

    def test_preset_dates(self):
        """Test preset date reference."""
        config = DatesConfig(preset="full_periods.five_years")
        assert config.preset == "full_periods.five_years"

    def test_dates_requires_both_or_preset(self):
        """Test that both start/end or preset is required."""
        with pytest.raises(ValidationError):
            DatesConfig(start="2023-01-01")  # Missing end

        with pytest.raises(ValidationError):
            DatesConfig(end="2024-01-01")  # Missing start


class TestBacktestSettings:
    """Tests for BacktestSettings model."""

    def test_default_settings(self):
        """Test default backtest settings."""
        settings = BacktestSettings()
        assert settings.initial_capital == 100000.0
        assert settings.fees == 0.001
        assert settings.slippage == 0.0005
        assert settings.benchmark == "SPY"
        assert settings.market_hours_only is True
        assert settings.allow_shorts is False

    def test_custom_settings(self):
        """Test custom backtest settings."""
        settings = BacktestSettings(
            initial_capital=50000.0,
            fees=0.002,
            allow_shorts=True
        )
        assert settings.initial_capital == 50000.0
        assert settings.fees == 0.002
        assert settings.allow_shorts is True

    def test_capital_must_be_positive(self):
        """Test that capital must be positive."""
        with pytest.raises(ValidationError):
            BacktestSettings(initial_capital=-1000)

    def test_fees_must_be_in_range(self):
        """Test fees must be between 0 and 0.1."""
        with pytest.raises(ValidationError):
            BacktestSettings(fees=0.5)  # 50% fees is invalid


class TestRiskSettings:
    """Tests for RiskSettings model."""

    def test_default_risk_settings(self):
        """Test default risk settings."""
        settings = RiskSettings()
        assert settings.enabled is True
        assert settings.position_sizing_method == PositionSizingMethod.FIXED_PERCENT
        assert settings.position_size_pct == 0.10
        assert settings.max_positions == 5
        assert settings.stop_loss_pct is None

    def test_custom_risk_settings(self):
        """Test custom risk settings."""
        settings = RiskSettings(
            position_size_pct=0.15,
            max_positions=10,
            stop_loss_pct=0.03
        )
        assert settings.position_size_pct == 0.15
        assert settings.max_positions == 10
        assert settings.stop_loss_pct == 0.03


class TestOptimizationSettings:
    """Tests for OptimizationSettings model."""

    def test_default_optimization_settings(self):
        """Test default optimization settings."""
        settings = OptimizationSettings()
        assert settings.metric == "sharpe_ratio"
        assert settings.param_grid == {}
        assert settings.n_jobs == 4

    def test_valid_metrics(self):
        """Test valid optimization metrics."""
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']:
            settings = OptimizationSettings(metric=metric)
            assert settings.metric == metric

    def test_invalid_metric(self):
        """Test that invalid metrics are rejected."""
        with pytest.raises(ValidationError):
            OptimizationSettings(metric="invalid_metric")


class TestBacktestConfig:
    """Tests for root BacktestConfig model."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = BacktestConfig(
            strategy=StrategyConfig(name="MovingAverageCrossover"),
            symbols=SymbolsConfig(list=["SPY"]),
            dates=DatesConfig(start="2023-01-01", end="2024-01-01")
        )
        assert config.mode == BacktestMode.SINGLE
        assert config.strategy.name == "MovingAverageCrossover"
        assert config.symbols.list == ["SPY"]

    def test_full_config(self):
        """Test full configuration with all options."""
        config = BacktestConfig(
            mode=BacktestMode.SWEEP,
            strategy=StrategyConfig(
                name="MeanReversion",
                parameters={"lookback": 20, "threshold": 2.0}
            ),
            symbols=SymbolsConfig(universe="technology.mag7"),
            dates=DatesConfig(preset="full_periods.five_years"),
            backtest=BacktestSettings(initial_capital=50000),
            risk=RiskSettings(position_size_pct=0.15),
            sweep=SweepSettings(parallel=True, max_workers=8),
            output=OutputSettings(quantstats=True, verbosity=2)
        )
        assert config.mode == BacktestMode.SWEEP
        assert config.backtest.initial_capital == 50000
        assert config.sweep.max_workers == 8

    def test_config_requires_strategy(self):
        """Test that strategy is required."""
        with pytest.raises(ValidationError):
            BacktestConfig(
                symbols=SymbolsConfig(list=["SPY"]),
                dates=DatesConfig(start="2023-01-01", end="2024-01-01")
            )

    def test_config_requires_symbols(self):
        """Test that symbols is required."""
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy=StrategyConfig(name="MeanReversion"),
                dates=DatesConfig(start="2023-01-01", end="2024-01-01")
            )

    def test_config_requires_dates(self):
        """Test that dates is required."""
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy=StrategyConfig(name="MeanReversion"),
                symbols=SymbolsConfig(list=["SPY"])
            )

    def test_config_allows_extra_fields(self):
        """Test that extra fields are allowed (forward compatibility)."""
        config = BacktestConfig(
            strategy=StrategyConfig(name="MeanReversion"),
            symbols=SymbolsConfig(list=["SPY"]),
            dates=DatesConfig(start="2023-01-01", end="2024-01-01"),
            future_field="some_value"  # Should not raise
        )
        assert config.strategy.name == "MeanReversion"
