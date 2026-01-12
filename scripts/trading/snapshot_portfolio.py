#!/usr/bin/env python3
"""
Hourly Portfolio Snapshot Script.

Captures portfolio state, positions, value, and P&L at the current time.
Designed to run via cron at XX:00 system time.

Output: Appends to ~/.homeguard/demo/snapshots/portfolio_history.csv

Usage:
    python scripts/trading/snapshot_portfolio.py

Cron setup (run hourly at :00):
    0 * * * * cd ~/Homeguard && venv/bin/python scripts/trading/snapshot_portfolio.py
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("scripts", 1)[0].rstrip("/\\"))

from src.data.providers.binance import BinanceDataProvider
from src.utils.timezone import tz

# Configuration
SNAPSHOT_DIR = Path.home() / '.homeguard' / 'demo' / 'snapshots'
SNAPSHOT_FILE = SNAPSHOT_DIR / 'portfolio_history.csv'
PORTFOLIO_STATE_FILE = Path.home() / '.homeguard' / 'demo' / 'portfolio_state.json'
INITIAL_CAPITAL = 100000.0


def load_portfolio_state() -> dict:
    """Load portfolio state from JSON file."""
    if not PORTFOLIO_STATE_FILE.exists():
        return None

    with open(PORTFOLIO_STATE_FILE, 'r') as f:
        return json.load(f)


def get_current_prices(symbols: list) -> dict:
    """Fetch current prices from Binance.US REST API."""
    provider = BinanceDataProvider()
    return provider.get_current_prices(symbols)


def calculate_portfolio_value(state: dict, prices: dict) -> dict:
    """Calculate portfolio metrics."""
    cash = state.get('cash', 0)
    positions = state.get('positions', [])
    realized_pnl = state.get('realized_pnl', 0)

    # Calculate position values
    total_invested = 0
    position_details = []

    for pos in positions:
        symbol = pos['symbol']
        qty = float(pos['quantity'])
        entry = float(pos['avg_entry_price'])
        cost = float(pos['cost_basis'])
        current = prices.get(symbol, 0)

        if current > 0:
            value = qty * current
            pnl = value - cost
            pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
            total_invested += value

            position_details.append({
                'symbol': symbol,
                'quantity': qty,
                'entry_price': entry,
                'current_price': current,
                'value': value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })

    total_value = cash + total_invested
    unrealized_pnl = total_invested - sum(float(p['cost_basis']) for p in positions)
    total_pnl = total_value - INITIAL_CAPITAL
    total_pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100

    return {
        'cash': cash,
        'invested': total_invested,
        'total_value': total_value,
        'unrealized_pnl': unrealized_pnl,
        'realized_pnl': realized_pnl,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'num_positions': len(positions),
        'positions': position_details
    }


def write_snapshot(metrics: dict) -> None:
    """Append snapshot to CSV file."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if file exists (need to write header)
    write_header = not SNAPSHOT_FILE.exists()

    timestamp = tz.now().strftime('%Y-%m-%d %H:%M:%S %Z')

    # Build position string (compact format)
    pos_str = '; '.join(
        f"{p['symbol']}:{p['quantity']:.4f}@{p['current_price']:.2f}"
        for p in metrics['positions']
    )

    row = {
        'timestamp': timestamp,
        'cash': f"{metrics['cash']:.2f}",
        'invested': f"{metrics['invested']:.2f}",
        'total_value': f"{metrics['total_value']:.2f}",
        'unrealized_pnl': f"{metrics['unrealized_pnl']:.2f}",
        'realized_pnl': f"{metrics['realized_pnl']:.2f}",
        'total_pnl': f"{metrics['total_pnl']:.2f}",
        'total_pnl_pct': f"{metrics['total_pnl_pct']:.2f}",
        'num_positions': metrics['num_positions'],
        'positions': pos_str
    }

    fieldnames = ['timestamp', 'cash', 'invested', 'total_value',
                  'unrealized_pnl', 'realized_pnl', 'total_pnl',
                  'total_pnl_pct', 'num_positions', 'positions']

    with open(SNAPSHOT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    """Main entry point."""
    # Load portfolio state
    state = load_portfolio_state()
    if state is None:
        print(f"[Snapshot] No portfolio state found at {PORTFOLIO_STATE_FILE}")
        return

    # Get symbols from positions
    symbols = [p['symbol'] for p in state.get('positions', [])]
    if not symbols:
        print("[Snapshot] No positions to track")
        # Still record snapshot with 0 invested
        metrics = {
            'cash': state.get('cash', 0),
            'invested': 0,
            'total_value': state.get('cash', 0),
            'unrealized_pnl': 0,
            'realized_pnl': state.get('realized_pnl', 0),
            'total_pnl': state.get('cash', 0) - INITIAL_CAPITAL,
            'total_pnl_pct': ((state.get('cash', 0) / INITIAL_CAPITAL) - 1) * 100,
            'num_positions': 0,
            'positions': []
        }
    else:
        # Get current prices
        prices = get_current_prices(symbols)
        if not prices:
            print("[Snapshot] Failed to fetch prices from Binance")
            return

        # Calculate metrics
        metrics = calculate_portfolio_value(state, prices)

    # Write snapshot
    write_snapshot(metrics)

    print(f"[Snapshot] {tz.now().strftime('%Y-%m-%d %H:%M:%S %Z')}: "
          f"${metrics['total_value']:,.2f} ({metrics['total_pnl_pct']:+.2f}%)")


if __name__ == '__main__':
    main()
