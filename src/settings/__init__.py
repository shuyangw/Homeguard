"""
Configuration module for the Homeguard backtesting framework.

This module consolidates:
1. Application settings (from settings.ini) - OS detection, directory paths
2. Config-driven backtesting - YAML configuration with Pydantic validation

Usage - Application Settings:
    from src.settings import settings, OS_ENVIRONMENT
    from src.settings import get_backtest_results_dir, get_output_dir

Usage - Config-driven Backtesting:
    from src.settings import load_config, BacktestConfig

    # Load from YAML file
    config = load_config("config/backtesting/omr_backtest.yaml")

    # Load with CLI overrides
    config = load_config("config.yaml", overrides={
        "backtest.initial_capital": 50000,
        "dates.start": "2023-01-01"
    })

    # Access validated settings
    print(config.strategy.name)
    print(config.backtest.initial_capital)
"""

# ============================================================================
# Application Settings (from settings.ini)
# ============================================================================
from src.settings.settings import (
    # Settings objects
    settings,
    OS_ENVIRONMENT,
    PROJECT_ROOT,
    SETTINGS_FILE,
    # Directory getters
    get_os_environment,
    get_local_storage_dir,
    get_output_dir,
    get_log_output_dir,
    get_backtest_results_dir,
    get_backtest_tests_dir,
    get_live_trading_dir,
    get_models_dir,
    get_tearsheet_frequency,
)

# ============================================================================
# Config-driven Backtesting
# ============================================================================
from src.settings.schema import (
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

from src.settings.loader import (
    load_config,
    load_config_dict,
    validate_config,
    load_yaml,
    merge_dicts,
    apply_overrides,
    get_nested,
)

from src.settings.defaults import (
    DEFAULT_CONFIG,
    DATE_PRESETS,
    SYMBOL_UNIVERSES,
    get_date_preset,
    get_symbol_universe,
)

__all__ = [
    # =========== Application Settings ===========
    "settings",
    "OS_ENVIRONMENT",
    "PROJECT_ROOT",
    "SETTINGS_FILE",
    "get_os_environment",
    "get_local_storage_dir",
    "get_output_dir",
    "get_log_output_dir",
    "get_backtest_results_dir",
    "get_backtest_tests_dir",
    "get_live_trading_dir",
    "get_models_dir",
    "get_tearsheet_frequency",
    # =========== Backtest Config Classes ===========
    "BacktestConfig",
    "BacktestMode",
    "PositionSizingMethod",
    "StrategyConfig",
    "SymbolsConfig",
    "DatesConfig",
    "BacktestSettings",
    "RiskSettings",
    "SweepSettings",
    "OptimizationSettings",
    "WalkForwardSettings",
    "OutputSettings",
    # =========== Config Loader Functions ===========
    "load_config",
    "load_config_dict",
    "validate_config",
    "load_yaml",
    "merge_dicts",
    "apply_overrides",
    "get_nested",
    # =========== Config Defaults ===========
    "DEFAULT_CONFIG",
    "DATE_PRESETS",
    "SYMBOL_UNIVERSES",
    "get_date_preset",
    "get_symbol_universe",
]
