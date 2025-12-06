"""
Data Provider Factory - Creates providers from configuration.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
import yaml

from src.data.providers.base import DataProviderInterface
from src.data.providers.alpaca import AlpacaDataProvider
from src.data.providers.yfinance import YFinanceDataProvider
from src.data.providers.composite import CompositeDataProvider
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.brokers.broker_interface import BrokerInterface


def create_data_provider(
    broker: Optional["BrokerInterface"] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None
) -> DataProviderInterface:
    """
    Create a data provider based on configuration.

    Default behavior (no config):
        - If broker provided: CompositeDataProvider([Alpaca, yfinance])
        - If no broker: YFinanceDataProvider only

    Args:
        broker: Optional broker for Alpaca provider
        config: Optional configuration dict
        config_path: Optional path to YAML config file

    Returns:
        Configured DataProviderInterface

    Usage:
        # Default: Alpaca -> yfinance fallback
        provider = create_data_provider(broker=my_broker)

        # From YAML config
        provider = create_data_provider(
            broker=my_broker,
            config_path='config/data_providers.yaml'
        )
    """
    # Load config from file if path provided
    if config_path and config is None:
        config = _load_yaml_config(config_path)

    if config is None:
        config = {}

    # Get provider settings
    dp_config = config.get('data_providers', config)
    provider_names = dp_config.get('providers', ['alpaca', 'yfinance'])
    cache_config = dp_config.get('cache', {})
    cache_enabled = cache_config.get('enabled', True)
    cache_max_age = cache_config.get('max_age_hours', 24)

    # Build provider list
    providers: List[DataProviderInterface] = []

    for name in provider_names:
        name_lower = name.lower()

        if name_lower == 'alpaca':
            if broker is not None:
                providers.append(AlpacaDataProvider(broker))
            else:
                logger.warning("Alpaca provider requested but no broker provided")

        elif name_lower == 'yfinance':
            providers.append(YFinanceDataProvider())

        else:
            logger.warning(f"Unknown provider: {name}")

    if not providers:
        logger.warning("No providers configured, using yfinance")
        providers.append(YFinanceDataProvider())

    # Single provider - no composite needed
    if len(providers) == 1:
        return providers[0]

    return CompositeDataProvider(
        providers=providers,
        cache_enabled=cache_enabled,
        cache_max_age_hours=cache_max_age
    )


def _load_yaml_config(path: str) -> Dict:
    """Load YAML configuration file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}
