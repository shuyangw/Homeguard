"""
Broker Factory - Factory Pattern for Creating Broker Instances

Creates broker instances from configuration, enabling easy switching
between brokers (Alpaca, Interactive Brokers, TD Ameritrade, etc.).

Usage:
    >>> config = {'api_key': 'KEY', 'secret_key': 'SECRET', 'paper': True}
    >>> broker = BrokerFactory.create_broker('alpaca', config)
    >>> bot = PaperTradingBot(broker=broker, config=trading_config)

Design Principle:
- Factory Pattern: Centralized broker creation
- Configuration-Driven: Change brokers via config, not code
"""

from typing import Dict, List
import os
import yaml

from .broker_interface import BrokerInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrokerFactory:
    """
    Factory for creating broker instances.

    Makes it easy to switch between brokers via configuration.
    """

    @staticmethod
    def create_broker(broker_type: str, config: Dict) -> BrokerInterface:
        """
        Create broker instance from configuration.

        Args:
            broker_type: Broker type ('alpaca', 'ib', 'tdameritrade', etc.)
            config: Broker configuration dict

        Returns:
            BrokerInterface implementation

        Raises:
            ValueError: If broker type not supported
            ImportError: If broker implementation not available

        Example:
            >>> config = {
            ...     'api_key': 'YOUR_KEY',
            ...     'secret_key': 'YOUR_SECRET',
            ...     'paper': True
            ... }
            >>> broker = BrokerFactory.create_broker('alpaca', config)
        """
        broker_type = broker_type.lower().strip()

        if broker_type == 'alpaca':
            from .alpaca_broker import AlpacaBroker
            logger.info("Creating AlpacaBroker instance")
            return AlpacaBroker(
                api_key=config['api_key'],
                secret_key=config['secret_key'],
                paper=config.get('paper', True)
            )

        elif broker_type in ['ib', 'interactive_brokers', 'interactivebrokers']:
            # Future implementation
            logger.error("Interactive Brokers not implemented yet")
            raise NotImplementedError(
                "Interactive Brokers support not implemented yet. "
                "To add IB support, implement IBBroker class in ib_broker.py"
            )

        elif broker_type in ['tdameritrade', 'tda', 'td_ameritrade']:
            # Future implementation
            logger.error("TD Ameritrade not implemented yet")
            raise NotImplementedError(
                "TD Ameritrade support not implemented yet. "
                "To add TDA support, implement TDAmeritradeBroker class in tdameritrade_broker.py"
            )

        else:
            logger.error(f"Unsupported broker type: {broker_type}")
            raise ValueError(
                f"Unsupported broker type: {broker_type}. "
                f"Supported types: 'alpaca', 'ib', 'tdameritrade'"
            )

    @staticmethod
    def create_from_yaml(config_path: str) -> BrokerInterface:
        """
        Create broker from YAML config file.

        Args:
            config_path: Path to broker config YAML

        Returns:
            BrokerInterface implementation

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid

        Example YAML:
            broker:
              type: alpaca
              api_key: ${ALPACA_PAPER_KEY_ID}
              secret_key: ${ALPACA_PAPER_SECRET_KEY}
              paper: true
        """
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading broker config from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'broker' not in config:
            raise ValueError("Config must contain 'broker' section")

        broker_config = config['broker']

        if 'type' not in broker_config:
            raise ValueError("Broker config must contain 'type' field")

        broker_type = broker_config['type']

        # Resolve environment variables
        for key, value in broker_config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.warning(f"Environment variable not set: {env_var}")
                    raise ValueError(f"Environment variable not set: {env_var}")
                broker_config[key] = env_value
                logger.info(f"Resolved {key} from environment variable: {env_var}")

        return BrokerFactory.create_broker(broker_type, broker_config)

    @staticmethod
    def create_from_env() -> BrokerInterface:
        """
        Create broker from environment variables.

        Reads broker type and configuration from environment.
        Useful for containerized deployments.

        Environment Variables:
            BROKER_TYPE: Broker type ('alpaca', 'ib', etc.)
            ALPACA_PAPER_KEY_ID: Alpaca API key
            ALPACA_PAPER_SECRET_KEY: Alpaca secret key
            (Plus broker-specific variables)

        Returns:
            BrokerInterface implementation

        Raises:
            ValueError: If required environment variables not set
        """
        broker_type = os.getenv('BROKER_TYPE', 'alpaca').lower()
        logger.info(f"Creating broker from environment (type: {broker_type})")

        if broker_type == 'alpaca':
            api_key = os.getenv('ALPACA_PAPER_KEY_ID')
            secret_key = os.getenv('ALPACA_PAPER_SECRET_KEY')

            if not api_key or not secret_key:
                raise ValueError(
                    "Alpaca credentials not found in environment. "
                    "Set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY"
                )

            config = {
                'api_key': api_key,
                'secret_key': secret_key,
                'paper': True
            }

            return BrokerFactory.create_broker('alpaca', config)

        else:
            raise NotImplementedError(
                f"Environment-based config for {broker_type} not implemented yet"
            )

    @staticmethod
    def list_supported_brokers() -> List[str]:
        """
        List all supported broker types.

        Returns:
            List of supported broker type strings
        """
        return [
            'alpaca',  # Alpaca Markets (implemented)
            'ib',  # Interactive Brokers (planned)
            'tdameritrade',  # TD Ameritrade (planned)
        ]
