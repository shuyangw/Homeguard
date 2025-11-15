"""
OMR Trading Configuration Loader.

Loads the authoritative OMR trading configuration from YAML file.
Ensures all production scripts use the validated 20-symbol universe.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Handle imports whether running as module or script
try:
    from src.utils.logger import logger
except ModuleNotFoundError:
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.logger import logger


class OMRConfig:
    """OMR trading configuration container."""

    def __init__(self, config_dict: Dict):
        """
        Initialize OMR config from dictionary.

        Args:
            config_dict: Parsed YAML config dictionary
        """
        if 'strategy' not in config_dict:
            raise ValueError("Config must contain 'strategy' section")

        strategy = config_dict['strategy']

        # Strategy metadata
        self.name = strategy.get('name', 'OMR')
        self.version = strategy.get('version', '1.0.0')
        self.description = strategy.get('description', 'Overnight Mean Reversion Strategy')

        # Timing
        self.entry_time = strategy.get('entry_time', '15:50:00')
        self.exit_time = strategy.get('exit_time', '09:31:00')
        self.entry_window_seconds = strategy.get('entry_window_seconds', 30)
        self.exit_window_seconds = strategy.get('exit_window_seconds', 30)

        # Position sizing
        self.position_size_pct = strategy.get('position_size_pct', 0.15)
        self.max_concurrent_positions = strategy.get('max_concurrent_positions', 3)
        self.max_total_exposure_pct = strategy.get('max_total_exposure_pct', 0.45)

        # Risk management
        self.stop_loss_pct = strategy.get('stop_loss_pct', -0.02)
        self.vix_threshold = strategy.get('vix_threshold', 35)

        # Signal quality filters
        self.min_win_rate = strategy.get('min_win_rate', 0.58)
        self.min_expected_return = strategy.get('min_expected_return', 0.002)
        self.min_sample_size = strategy.get('min_sample_size', 15)

        # Regime filtering
        self.skip_regimes = strategy.get('skip_regimes', ['BEAR'])

        # Symbol universe (CRITICAL: Authoritative source)
        self.symbols = strategy.get('symbols', [])
        if not self.symbols:
            raise ValueError("Config must contain non-empty 'symbols' list")

        # Historical data
        self.training_period_years = strategy.get('training_period_years', 2)
        self.retrain_frequency = strategy.get('retrain_frequency', 'daily')

        # Execution
        self.order_type = strategy.get('order_type', 'market')
        self.limit_order_offset_pct = strategy.get('limit_order_offset_pct', 0.001)

        # Position manager config
        pm_config = config_dict.get('position_manager', {})
        self.pm_max_position_size_pct = pm_config.get('max_position_size_pct', self.position_size_pct)
        self.pm_max_concurrent_positions = pm_config.get('max_concurrent_positions', self.max_concurrent_positions)
        self.pm_max_total_exposure_pct = pm_config.get('max_total_exposure_pct', self.max_total_exposure_pct)
        self.pm_stop_loss_pct = pm_config.get('stop_loss_pct', self.stop_loss_pct)
        self.pm_save_state = pm_config.get('save_state', True)
        self.pm_state_file = pm_config.get('state_file', 'data/trading/position_state.json')

        # Risk limits
        risk_config = config_dict.get('risk', {})
        self.risk_max_daily_loss_pct = risk_config.get('max_daily_loss_pct', 0.05)
        self.risk_max_weekly_loss_pct = risk_config.get('max_weekly_loss_pct', 0.10)
        self.risk_max_drawdown_pct = risk_config.get('max_drawdown_pct', 0.15)
        self.risk_max_orders_per_day = risk_config.get('max_orders_per_day', 50)
        self.risk_max_order_value = risk_config.get('max_order_value', 50000)
        self.risk_min_order_value = risk_config.get('min_order_value', 100)
        self.risk_min_buying_power = risk_config.get('min_buying_power', 10000)
        self.risk_min_account_value = risk_config.get('min_account_value', 50000)

    def __repr__(self):
        """String representation."""
        return (
            f"OMRConfig(name='{self.name}', version='{self.version}', "
            f"symbols={len(self.symbols)}, entry={self.entry_time}, exit={self.exit_time})"
        )

    def to_adapter_params(self) -> Dict:
        """
        Convert config to OMRLiveAdapter parameters.

        Returns:
            Dictionary of parameters for OMRLiveAdapter constructor
        """
        return {
            'symbols': self.symbols,
            'min_probability': self.min_win_rate,
            'min_expected_return': self.min_expected_return,
            'max_positions': self.max_concurrent_positions,
            'position_size': self.position_size_pct
        }


def load_omr_config(config_path: Optional[str] = None) -> OMRConfig:
    """
    Load OMR trading configuration from YAML file.

    This is the AUTHORITATIVE way to load OMR configuration.
    All production scripts should use this function.

    Args:
        config_path: Path to config file (default: config/trading/omr_trading_config.yaml)

    Returns:
        OMRConfig object with validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid

    Example:
        >>> config = load_omr_config()
        >>> print(f"Trading {len(config.symbols)} symbols")
        >>> adapter = OMRLiveAdapter(broker, **config.to_adapter_params())
    """
    # Default config path
    if config_path is None:
        # Try to find project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent  # src/trading/config/ -> root
        config_path = project_root / "config" / "trading" / "omr_trading_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"OMR config file not found: {config_path}")

    logger.info(f"Loading OMR config from: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = OMRConfig(config_dict)

    logger.info(f"Loaded OMR config: {config}")
    logger.info(f"  Symbols: {len(config.symbols)} ETFs")
    logger.info(f"  Entry: {config.entry_time} | Exit: {config.exit_time}")
    logger.info(f"  Position size: {config.position_size_pct:.1%}")
    logger.info(f"  Max positions: {config.max_concurrent_positions}")
    logger.info(f"  Min win rate: {config.min_win_rate:.1%}")

    return config


def get_production_symbols() -> List[str]:
    """
    Get the production symbol universe for OMR strategy.

    This is a convenience function that loads the config and returns
    just the symbol list.

    Returns:
        List of 20 validated symbols for OMR trading

    Example:
        >>> symbols = get_production_symbols()
        >>> print(len(symbols))  # 20
    """
    config = load_omr_config()
    return config.symbols


# Validation function
def validate_symbols(symbols: List[str]) -> bool:
    """
    Validate that the provided symbols match the production config.

    Args:
        symbols: List of symbols to validate

    Returns:
        True if symbols match production config, False otherwise

    Example:
        >>> from src.strategies.universe import ETFUniverse
        >>> # This should return False (wrong symbols)
        >>> validate_symbols(ETFUniverse.LEVERAGED_3X)
        False
        >>> # This should return True
        >>> validate_symbols(get_production_symbols())
        True
    """
    production_symbols = set(get_production_symbols())
    provided_symbols = set(symbols)

    if production_symbols == provided_symbols:
        logger.info("[OK] Symbols match production configuration")
        return True
    else:
        missing = production_symbols - provided_symbols
        extra = provided_symbols - production_symbols

        logger.warning("[!] Symbols do NOT match production configuration")
        if missing:
            logger.warning(f"  Missing from provided list: {sorted(missing)}")
        if extra:
            logger.warning(f"  Extra in provided list: {sorted(extra)}")

        logger.warning(f"  Expected: {len(production_symbols)} symbols")
        logger.warning(f"  Provided: {len(provided_symbols)} symbols")

        return False


if __name__ == "__main__":
    # Add project root to path for testing
    import sys
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Test the config loader
    print("Testing OMR config loader...")
    print()

    try:
        config = load_omr_config()
        print(f"Config loaded successfully: {config}")
        print()
        print(f"Symbols ({len(config.symbols)}):")
        for i, symbol in enumerate(config.symbols, 1):
            print(f"  {i:2d}. {symbol}")
        print()
        print("Adapter parameters:")
        import json
        print(json.dumps(config.to_adapter_params(), indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
