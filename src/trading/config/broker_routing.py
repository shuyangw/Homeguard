"""
Broker Routing - Config-driven strategy-to-broker assignment.

Reads broker_routing.yaml and creates shared broker instances.
Strategies get their assigned broker; unlisted strategies get the default.
"""

from typing import Any, Dict, Tuple

import yaml

from src.trading.brokers.broker_factory import BrokerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BrokerRoutingMap:
    """Maps strategy names to (broker_name, broker_instance) pairs."""

    def __init__(
        self,
        strategy_map: Dict[str, Tuple[str, Any]],
        default: Tuple[str, Any],
    ):
        self._map = strategy_map
        self._default = default

    def __getitem__(self, strategy_name: str):
        return self._map.get(strategy_name, self._default)[1]

    def __contains__(self, strategy_name: str) -> bool:
        return strategy_name in self._map

    def get(self, strategy_name: str, fallback=None):
        if strategy_name in self._map:
            return self._map[strategy_name][1]
        return fallback if fallback is not None else self._default[1]

    def get_broker_name(self, strategy_name: str) -> str:
        return self._map.get(strategy_name, self._default)[0]

    def get_default(self):
        return self._default[1]


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
        BrokerRoutingMap mapping strategy names to (broker_name, instance) pairs
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

    # Map strategies to (broker_name, broker_instance) tuples
    strategy_map: Dict[str, Tuple[str, object]] = {}
    for strategy_name, strategy_cfg in strategies_config.items():
        broker_name = strategy_cfg.get("broker", default_broker_name)
        if broker_name in broker_instances:
            strategy_map[strategy_name] = (
                broker_name,
                broker_instances[broker_name],
            )
        else:
            logger.warning(
                f"[Routing] Strategy '{strategy_name}' references unknown "
                f"broker '{broker_name}', using default"
            )

    default_instance = broker_instances.get(default_broker_name)
    default_tuple = (default_broker_name, default_instance)

    logger.info(
        f"[Routing] Loaded {len(strategy_map)} strategy assignments, "
        f"default broker: {default_broker_name}"
    )

    return BrokerRoutingMap(strategy_map, default_tuple)
