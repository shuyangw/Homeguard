"""
Broker Routing - Config-driven strategy-to-broker assignment.

Reads broker_routing.yaml and creates shared broker instances.
Strategies get their assigned broker; unlisted strategies get the default.
"""

from typing import Dict, Optional

import yaml

from src.trading.brokers.broker_factory import BrokerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrokerRoutingMap:
    """Maps strategy names to broker instances."""

    def __init__(self, strategy_map: Dict, default_broker):
        self._map = strategy_map
        self._default = default_broker

    def __getitem__(self, strategy_name: str):
        return self._map.get(strategy_name, self._default)

    def __contains__(self, strategy_name: str) -> bool:
        return strategy_name in self._map

    def get(self, strategy_name: str, fallback=None):
        return self._map.get(strategy_name, fallback or self._default)

    def get_default(self):
        return self._default


def load_broker_routing(
    config_path: str = "config/trading/broker_routing.yaml",
) -> BrokerRoutingMap:
    """
    Load broker routing config and create broker instances.

    Brokers are shared: two strategies assigned to 'alpaca' get the same
    instance.

    Args:
        config_path: Path to broker_routing.yaml

    Returns:
        BrokerRoutingMap mapping strategy names to broker instances
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    brokers_config = config.get("brokers", {})
    strategies_config = config.get("strategies", {})
    default_broker_name = config.get("default_broker", "alpaca")

    # Create broker instances (shared)
    broker_instances: Dict[str, object] = {}
    for broker_name, broker_cfg in brokers_config.items():
        try:
            cfg = dict(broker_cfg)
            broker_type = cfg.pop("type", broker_name)
            broker_instances[broker_name] = BrokerFactory.create_broker(
                broker_type, cfg
            )
            logger.info(f"[Routing] Created broker: {broker_name}")
        except Exception as e:
            logger.error(
                f"[Routing] Failed to create broker '{broker_name}': {e}"
            )

    # Map strategies to broker instances
    strategy_map = {}
    for strategy_name, strategy_cfg in strategies_config.items():
        broker_name = strategy_cfg.get("broker", default_broker_name)
        if broker_name in broker_instances:
            strategy_map[strategy_name] = broker_instances[broker_name]
        else:
            logger.warning(
                f"[Routing] Strategy '{strategy_name}' references unknown "
                f"broker '{broker_name}', using default"
            )

    default = broker_instances.get(default_broker_name)

    logger.info(
        f"[Routing] Loaded {len(strategy_map)} strategy assignments, "
        f"default broker: {default_broker_name}"
    )

    return BrokerRoutingMap(strategy_map, default)
