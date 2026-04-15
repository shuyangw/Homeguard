#!/usr/bin/env python3
"""
Pre-flight check for broker switching.

Usage:
    python scripts/trading/check_broker_switch.py --strategy omr --to ibkr
    python scripts/trading/check_broker_switch.py --strategy ramp --to alpaca
    python scripts/trading/check_broker_switch.py --all

Exit code 0 = safe to switch. Exit code 1 = not safe.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.config.broker_routing import load_broker_routing
from src.trading.state.strategy_state_manager import StrategyStateManager
from src.utils.logger import logger


def print_report(strategy: str, current_name: str, new_name: str, result: dict) -> None:
    logger.info(f"[+] Checking broker switch: {strategy} ({current_name} -> {new_name})")
    logger.info("")

    logger.info("    State file positions:")
    if result['state_positions']:
        for symbol, info in result['state_positions'].items():
            logger.info(
                f"      {symbol}: {info['qty']} shares (broker: {info['broker']})"
            )
    else:
        logger.info("      (none)")
    logger.info("")

    logger.info(f"    {current_name} positions:")
    if result['positions_on_current']:
        for symbol, qty in result['positions_on_current'].items():
            logger.info(f"      {symbol}: {qty} shares")
    else:
        logger.info("      (none)")
    logger.info("")

    logger.info(f"    {new_name} positions:")
    if result['positions_on_new']:
        for symbol, qty in result['positions_on_new'].items():
            logger.info(f"      {symbol}: {qty} shares")
    else:
        logger.info("      (none)")
    logger.info("")

    if result['safe']:
        logger.success("[+] SAFE TO SWITCH")
        logger.info("")
        logger.info("    Proceed:")
        logger.info(f"      1. Edit config/trading/broker_routing.yaml: {strategy}.broker -> {new_name}")
        logger.info(f"      2. Restart: sudo systemctl restart homeguard-{strategy}")
    else:
        logger.error("[-] NOT SAFE TO SWITCH")
        logger.info("")
        logger.info("    Blocking reasons:")
        for reason in result['blocking_reasons']:
            logger.error(f"      - {reason}")
        logger.info("")
        logger.info("    Action required:")
        logger.info(f"      {result['action_required']}")


def check_one(strategy: str, target_name: str) -> int:
    routing = load_broker_routing()
    current_name = routing.get_broker_name(strategy)

    if current_name == target_name:
        logger.warning(
            f"[!] {strategy} is already on {target_name}. Nothing to switch."
        )
        return 0

    current_broker = routing[strategy]

    # Build the target broker from the routing config.
    # The target broker must be defined in broker_routing.yaml's brokers section.
    import yaml
    from src.trading.brokers.broker_factory import BrokerFactory

    with open("config/trading/broker_routing.yaml") as f:
        cfg = yaml.safe_load(f)
    target_cfg = cfg.get("brokers", {}).get(target_name)
    if target_cfg is None:
        logger.error(
            f"[-] Target broker '{target_name}' not defined in broker_routing.yaml. "
            f"Uncomment the {target_name} block in the 'brokers:' section first."
        )
        return 1

    target_cfg = dict(target_cfg)
    target_type = target_cfg.pop("type", target_name)
    try:
        target_broker = BrokerFactory.create_broker(target_type, target_cfg)
    except Exception as e:
        logger.error(f"[-] Failed to create target broker '{target_name}': {e}")
        return 1

    mgr = StrategyStateManager()
    result = mgr.check_broker_switch_safety(
        strategy, current_name, target_name, current_broker, target_broker,
    )
    print_report(strategy, current_name, target_name, result)
    return 0 if result['safe'] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight broker switch safety check.")
    parser.add_argument('--strategy', help="Strategy name (omr, ramp, mp)")
    parser.add_argument('--to', dest='target', help="Target broker name (alpaca, ibkr)")
    parser.add_argument(
        '--all', action='store_true',
        help="Check all strategies against their currently-assigned broker (sanity check).",
    )
    args = parser.parse_args()

    if args.all:
        routing = load_broker_routing()
        mgr = StrategyStateManager()
        worst = 0
        for strategy in ['omr', 'ramp', 'mp']:
            current_name = routing.get_broker_name(strategy)
            current_broker = routing[strategy]
            result = mgr.check_broker_switch_safety(
                strategy, current_name, current_name, current_broker, current_broker,
            )
            print_report(strategy, current_name, current_name, result)
            if not result['safe']:
                worst = 1
        return worst

    if not args.strategy or not args.target:
        parser.error("Either --all, or both --strategy and --to must be given.")

    return check_one(args.strategy, args.target)


if __name__ == "__main__":
    sys.exit(main())
