#!/bin/bash
# CSCM Demo Positions Detail
#
# Shows detailed position breakdown with current values.
# Uses DemoBroker with Binance streaming (self-contained paper trading).
#
# Usage:
#   ./cscm_demo_positions.sh
#   cscm-demo-positions  (alias)

set -e

echo "=========================================="
echo "CSCM Demo Position Details"
echo "=========================================="
echo ""

cd ~/Homeguard

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from src.trading.demo import DemoBroker

    broker = DemoBroker()
    positions = broker.get_crypto_positions()
    balance = broker.get_crypto_balance()

    if not positions:
        print('No positions currently held.')
        print()
        print(f'Cash Balance: \${balance[\"available\"]:,.2f}')
        sys.exit(0)

    # Print header
    print(f'{\"Symbol\":<12} {\"Quantity\":>12} {\"Entry\":>12} {\"Current\":>12} {\"P&L\":>12} {\"P&L%\":>8}')
    print('-' * 72)

    # Print positions
    total_value = balance['available']
    total_pnl = 0

    for pos in positions:
        symbol = pos['symbol']
        qty = float(pos['quantity'])
        entry = pos['avg_entry_price']
        current = pos['current_price']
        value = pos['market_value']
        pnl = pos['unrealized_pnl']
        pnl_pct = pos['unrealized_pnl_pct']

        total_value += value
        total_pnl += pnl

        # Color P&L
        pnl_color = '\033[32m' if pnl >= 0 else '\033[31m'
        reset = '\033[0m'

        print(f'{symbol:<12} {qty:>12.6f} \${entry:>11,.2f} \${current:>11,.2f} {pnl_color}\${pnl:>11,.2f}{reset} {pnl_color}{pnl_pct:>7.1f}%{reset}')

    # Print summary
    print('-' * 72)
    print(f'{\"Cash\":<12} {\"\":>12} {\"\":>12} {\"\":>12} \${balance[\"available\"]:>11,.2f}')

    pnl_color = '\033[32m' if total_pnl >= 0 else '\033[31m'
    reset = '\033[0m'
    print(f'{\"Total\":<12} {\"\":>12} {\"\":>12} {\"\":>12} \${total_value:>11,.2f} {pnl_color}(\${total_pnl:>+,.2f}){reset}')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
