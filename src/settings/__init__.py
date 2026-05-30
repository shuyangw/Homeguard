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
    get_options_data_dir,
    get_output_dir,
    get_log_output_dir,
    get_backtest_results_dir,
    get_backtest_tests_dir,
    get_live_trading_dir,
    get_models_dir,
    get_discord_bot_log_dir,
    get_tearsheet_frequency,
)

# ============================================================================
# Canonical asset-class data path constants
# ============================================================================
from src.settings.data_paths import (
    # Constants
    EQUITIES_IEX_1MIN,
    EQUITIES_IEX_1MIN_BY_DATE,
    EQUITIES_SIP_RAW_1MIN,
    EQUITIES_SIP_SPLIT_1MIN,
    CRYPTO_ALPACA_1MIN,
    CRYPTO_ALPACA_1HOUR,
    CRYPTO_ALPACA_1DAY,
    CRYPTO_ALPACA_ARCHIVE_1MIN,
    FUTURES_DATABENTO_1MIN,
    FUTURES_DATABENTO_MBP1,
    FUTURES_DATABENTO_TRADES,
    FUTURES_DATABENTO_STAGING,
    FUTURES_DATABENTO_1MIN_OI_ROLL,
    FUTURES_DATABENTO_PER_CONTRACT_1MIN,
    FUTURES_DATABENTO_PER_CONTRACT_DAILY,
    FUTURES_DATABENTO_OPTIONS_1MIN,
    FUTURES_DATABENTO_STATISTICS,
    FUTURES_DATABENTO_STATUS,
    FUTURES_DEFINITIONS,
    FX_MASSIVE_1MIN,
    FX_MASSIVE_QUOTES_MINUTE_AGGREGATED,
    FX_MASSIVE_QUOTES_RAW,
    FX_POLYGON_1MIN_BACKFILL,
    NEWS_ALPACA,
    OPTIONS_THETADATA_LOGS,
    OPTIONS_THETADATA_CHAINS,
    OPTIONS_THETADATA_GEX_DAILY,
    OPTIONS_THETADATA_COMBINED,
    # Mapping + accessors
    LEGACY_TO_CANONICAL,
    manifest_filename,
    get_data_dir,
    get_equities_iex_1min_dir,
    get_equities_sip_raw_1min_dir,
    get_equities_sip_split_1min_dir,
    get_crypto_alpaca_1min_dir,
    get_futures_databento_1min_dir,
    get_futures_databento_mbp1_dir,
    get_futures_databento_trades_dir,
    get_fx_massive_1min_dir,
    get_news_alpaca_dir,
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
    "get_options_data_dir",
    "get_output_dir",
    "get_log_output_dir",
    "get_backtest_results_dir",
    "get_backtest_tests_dir",
    "get_live_trading_dir",
    "get_models_dir",
    "get_discord_bot_log_dir",
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
