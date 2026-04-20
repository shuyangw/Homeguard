"""
CSCM (Cross-Sectional Crypto Momentum) Live Trading Runner.

Runs the CSCM strategy continuously with Coinbase (primary) or Alpaca (secondary).
Features:
- 24/7 crypto trading (no market hours)
- Weekly rebalancing (Sunday 0:00 UTC)
- BTC regime filter
- Automatic broker failover

Usage:
    python scripts/trading/run_cscm_live.py
    python scripts/trading/run_cscm_live.py --paper
    python scripts/trading/run_cscm_live.py --broker alpaca
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import argparse
import threading
import time as time_module
from dotenv import load_dotenv

# Set process title for easy identification in ps/htop
try:
    import setproctitle
    setproctitle.setproctitle("homeguard-cscm")
except ImportError:
    pass

from src.trading.adapters.cscm_live_adapter import CSCMLiveAdapter
from src.utils.logger import logger


STRATEGY_NAME = 'cscm'
METRICS_EMIT_INTERVAL_SECONDS = 30


def _compute_today_realized_pnl(strategy: str) -> float:
    """Sum today's realized PnL for `strategy` by scanning the trade-log JSONL."""
    try:
        from src.utils.trading_logger import get_trade_log_writer
        writer = get_trade_log_writer()
        log_file = writer._get_log_file()
        if not log_file.exists():
            return 0.0
        total = 0.0
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if (row.get('strategy') == strategy
                        and row.get('trade_type') == 'exit'
                        and row.get('pnl_dollars') is not None):
                    total += float(row['pnl_dollars'])
        return total
    except Exception as e:
        logger.error(f"[CSCM-metrics] realized PnL aggregation failed: {e}")
        return 0.0


def _emit_metrics_tick(adapter, metrics_registry, state) -> None:
    """Emit one metrics snapshot for CSCM (portfolio, positions, strategy aggregates)."""
    from src.monitoring.hooks import (
        update_portfolio_metrics,
        update_market_status,
        update_process_metrics,
        update_position_metrics,
        update_strategy_metrics,
    )

    broker = adapter.broker
    broker_name = getattr(broker, '_active_broker_name', 'crypto')

    # ---- Portfolio ----
    equity = 0.0
    try:
        account = broker.get_account()
        if account:
            update_portfolio_metrics(metrics_registry, account, broker_name)
            equity = float(account.get('portfolio_value', 0) or 0)
    except Exception as e:
        logger.error(f"[CSCM-metrics] broker.get_account failed: {e}")

    # ---- Drawdown + day PnL (in-process tracking, since Coinbase has no last_equity) ----
    try:
        if equity > state['peak_equity']:
            state['peak_equity'] = equity
        drawdown_pct = 0.0 if state['peak_equity'] <= 0 else (
            (equity - state['peak_equity']) / state['peak_equity'] * 100.0
        )
        metrics_registry.update_drawdown(drawdown_pct)

        if state['session_open_equity'] is None and equity > 0:
            state['session_open_equity'] = equity
        day_pnl = equity - (state['session_open_equity'] or equity)
        metrics_registry.update_day_pnl(day_pnl)
    except Exception as e:
        logger.error(f"[CSCM-metrics] derived portfolio metrics failed: {e}")

    # ---- Per-position + strategy aggregates ----
    try:
        owned = adapter.state_manager.get_positions(STRATEGY_NAME) or {}

        positions = []
        unrealized = 0.0
        capital = 0.0
        for symbol, info in owned.items():
            qty = float(info.get('qty', 0) or 0)
            entry_price = float(info.get('entry_price', 0) or 0)
            current_price = 0.0
            try:
                quote = broker.get_crypto_quote(symbol)
                current_price = float(quote.get('last', 0) or 0)
            except Exception as quote_err:
                logger.error(f"[CSCM-metrics] quote failed for {symbol}: {quote_err}")

            pos_unrealized = (current_price - entry_price) * qty if (current_price and entry_price) else 0.0
            market_value = current_price * qty

            positions.append({
                'symbol': symbol,
                'quantity': qty,
                'unrealized_pnl': pos_unrealized,
            })
            unrealized += pos_unrealized
            capital += abs(market_value)

        update_position_metrics(metrics_registry, positions)

        realized = _compute_today_realized_pnl(STRATEGY_NAME)
        update_strategy_metrics(
            metrics_registry,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            positions=len(positions),
            capital_allocated=capital,
        )
    except Exception as e:
        logger.error(f"[CSCM-metrics] position/strategy metrics failed: {e}")

    # ---- Market status (crypto = always open) + process RSS ----
    try:
        update_market_status(metrics_registry, True)
        update_process_metrics(metrics_registry)
    except Exception as e:
        logger.error(f"[CSCM-metrics] market/process metrics failed: {e}")


def _start_metrics_sidecar(adapter, metrics_registry) -> threading.Thread:
    """Start daemon thread that emits CSCM metrics periodically."""
    state = {'peak_equity': 0.0, 'session_open_equity': None}

    def _loop():
        while True:
            try:
                _emit_metrics_tick(adapter, metrics_registry, state)
            except Exception as e:
                logger.error(f"[CSCM-metrics] tick failed: {e}")
            time_module.sleep(METRICS_EMIT_INTERVAL_SECONDS)

    thread = threading.Thread(target=_loop, name='cscm-metrics-sidecar', daemon=True)
    thread.start()
    logger.info(f"[CSCM-metrics] sidecar started (interval={METRICS_EMIT_INTERVAL_SECONDS}s)")
    return thread


def load_config(config_path: str = None) -> dict:
    """Load CSCM configuration from YAML file."""
    import yaml

    if config_path is None:
        config_path = project_root / 'config' / 'trading' / 'cscm_live.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run CSCM live crypto trading')

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config/trading/cscm_live.yaml)'
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Use paper trading (Alpaca paper mode)'
    )

    parser.add_argument(
        '--broker',
        type=str,
        default='auto',
        choices=['auto', 'coinbase', 'alpaca'],
        help='Broker to use (default: auto with failover)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=1,
        help='Hours between rebalance checks (default: 1)'
    )

    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (for testing)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current strategy status and exit'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=None,
        help='Cap strategy allocation at this USD amount (default: use full account balance)'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    config = load_config(args.config)

    logger.info("=" * 80)
    logger.info("CSCM LIVE CRYPTO TRADING")
    logger.info("=" * 80)
    logger.info(f"Paper mode: {args.paper}")
    logger.info(f"Broker: {args.broker}")
    logger.info(f"Check interval: {args.check_interval} hours")
    logger.info("=" * 80)

    try:
        # Extract parameters from config
        universe = config.get('universe', CSCMLiveAdapter.DEFAULT_UNIVERSE)
        top_n = config.get('top_n', CSCMLiveAdapter.DEFAULT_TOP_N)
        momentum_period = config.get('momentum_period', CSCMLiveAdapter.DEFAULT_MOMENTUM_PERIOD)
        btc_sma_period = config.get('btc_sma_period', CSCMLiveAdapter.DEFAULT_BTC_SMA_PERIOD)
        rebalance_day = config.get('rebalance_day', 'sunday')
        go_to_cash_in_bear = config.get('go_to_cash_in_bear', True)

        # Setup broker based on selection
        broker = None
        if args.broker != 'auto':
            if args.broker == 'coinbase':
                from src.trading.brokers.coinbase_broker import CoinbaseBroker
                broker_instance = CoinbaseBroker()
            else:  # alpaca
                from src.trading.brokers.alpaca_crypto_broker import AlpacaCryptoBroker
                broker_instance = AlpacaCryptoBroker(paper=args.paper)

            # Wrap in router with single broker
            from src.trading.brokers.crypto_router import CryptoBrokerRouter
            broker = CryptoBrokerRouter(
                primary=broker_instance if args.broker == 'coinbase' else None,
                secondary=broker_instance if args.broker == 'alpaca' else None,
                auto_failover=False
            )

        # Create signal logger for hypothetical performance tracking
        from src.trading.logging import CSCMSignalLogger

        signal_log_dir = Path(project_root) / "data" / "trading" / "logs"
        signal_logger = CSCMSignalLogger(log_dir=signal_log_dir)

        # Create adapter
        adapter = CSCMLiveAdapter(
            universe=universe,
            top_n=top_n,
            momentum_period=momentum_period,
            btc_sma_period=btc_sma_period,
            rebalance_day=rebalance_day,
            go_to_cash_in_bear=go_to_cash_in_bear,
            broker=broker,
            paper=args.paper,
            signal_logger=signal_logger,
            max_capital_usd=args.initial_capital,
        )

        logger.info(f"Universe: {len(universe)} symbols")
        logger.info(f"Top N: {top_n}")
        logger.info(f"Momentum Period: {momentum_period} days")
        logger.info(f"BTC SMA Period: {btc_sma_period} days")
        logger.info(f"Rebalance Day: {rebalance_day}")
        logger.info(f"Go to Cash in Bear: {go_to_cash_in_bear}")
        logger.info("")

        # Initialize metrics registry if enabled (mirrors run_live_paper_trading.py)
        if os.getenv('ENABLE_METRICS', 'false').lower() in ('true', '1', 'yes'):
            from src.monitoring import MetricsRegistry, start_metrics_server
            from src.monitoring.snapshot import SnapshotWriter
            from src.settings import get_local_storage_dir

            metrics_port = int(os.getenv('METRICS_PORT', '8084'))
            metrics_registry = MetricsRegistry(strategy=STRATEGY_NAME)
            start_metrics_server(metrics_registry, port=metrics_port)

            snapshot_dir = os.path.join(get_local_storage_dir(), 'metrics_snapshots')
            SnapshotWriter(metrics_registry, snapshot_dir=snapshot_dir).start_background()

            _start_metrics_sidecar(adapter, metrics_registry)

            logger.info(f"Metrics enabled on port {metrics_port}")

        if args.status:
            # Show status and exit
            status = adapter.get_status()
            logger.info("Current Strategy Status:")
            logger.info("-" * 40)
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
            return 0

        if args.once:
            # Run once and exit
            logger.info("Running single iteration...")
            adapter.run_once()
            logger.success("Done")
        else:
            # Run continuous loop
            logger.info("Starting continuous trading loop...")
            adapter.run(check_interval_hours=args.check_interval)

        return 0

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
