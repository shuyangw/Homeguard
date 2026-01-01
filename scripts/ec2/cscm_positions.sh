#!/bin/bash
# CSCM Positions Detail
#
# Shows detailed position breakdown with current values.
#
# Usage:
#   ./cscm_positions.sh
#   cscm-positions  (alias)

set -e

echo "=========================================="
echo "CSCM Position Details"
echo "=========================================="
echo ""

cd ~/Homeguard

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter

    adapter = CSCMPaperAdapter()

    # Get positions and prices
    positions = adapter._get_current_positions()
    prices = adapter._get_current_prices()
    cash = adapter._get_account_balance()

    if not positions:
        print('No positions currently held.')
        print()
        print(f'Cash Balance: \${cash:,.2f}')
        sys.exit(0)

    # Print header
    print(f'{\"Symbol\":<12} {\"Quantity\":>12} {\"Price\":>12} {\"Value\":>14}')
    print('-' * 52)

    # Print positions
    total = cash
    for symbol, qty in positions.items():
        price = prices.get(symbol, 0)
        value = float(qty) * price
        total += value
        print(f'{symbol:<12} {float(qty):>12.6f} \${price:>11,.2f} \${value:>13,.2f}')

    # Print summary
    print('-' * 52)
    print(f'{\"Cash\":<12} {\"\":>12} {\"\":>12} \${cash:>13,.2f}')
    print(f'{\"Total\":<12} {\"\":>12} {\"\":>12} \${total:>13,.2f}')
    print()

    # Show regime
    try:
        status = adapter.get_status()
        print(f'Regime: {status[\"regime\"].upper()}')
        if status.get('btc_price'):
            print(f'BTC: \${status[\"btc_price\"]:,.2f} (SMA: \${status[\"btc_sma\"]:,.2f})')
    except:
        pass

except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
