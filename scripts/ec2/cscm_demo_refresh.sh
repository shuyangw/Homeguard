#!/bin/bash
# CSCM Demo Rebalance
#
# Force a rebalance of the CSCM demo portfolio.
# Calculates momentum rankings and rebalances positions.
#
# WARNING: This triggers real trades in the demo account!
#
# Usage:
#   ./cscm_demo_refresh.sh
#   cscm-demo-refresh  (alias)

set -e

echo "=========================================="
echo "CSCM Demo Force Rebalance"
echo "=========================================="
echo ""

cd ~/Homeguard

# Confirm action
read -p "Force rebalance CSCM demo positions? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting rebalance..."
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from src.trading.adapters.cscm_demo_adapter import CSCMDemoAdapter

    # Demo universe
    DEMO_UNIVERSE = [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD',
        'DOGE/USD', 'DOT/USD', 'LTC/USD', 'BCH/USD', 'UNI/USD',
        'AAVE/USD', 'SUSHI/USD', 'XRP/USD', 'CRV/USD', 'GRT/USD',
        'MKR/USD',
    ]

    adapter = CSCMDemoAdapter(universe=DEMO_UNIVERSE, top_n=7)

    # Start streaming for fresh data
    print('[*] Starting data stream...')
    adapter.start_streaming()

    # Wait for data
    import time
    print('[*] Waiting for market data (30s)...')
    time.sleep(30)

    # Get pre-rebalance state
    broker = adapter.broker
    pre_positions = len(broker.get_crypto_positions())
    pre_value = broker.get_portfolio_value()

    print()
    print(f'Pre-rebalance:')
    print(f'  Positions: {pre_positions}')
    print(f'  Value:     \${pre_value:,.2f}')
    print()

    # Execute rebalance
    print('[*] Executing rebalance...')
    results = adapter.rebalance()

    # Get post-rebalance state
    post_positions = len(broker.get_crypto_positions())
    post_value = broker.get_portfolio_value()

    print()
    print(f'Post-rebalance:')
    print(f'  Positions: {post_positions}')
    print(f'  Value:     \${post_value:,.2f}')
    print()

    # Show orders executed
    if results:
        print(f'Orders executed: {len(results)}')
        for order in results:
            side = order.get('side', 'unknown')
            qty = order.get('filled_qty', order.get('quantity', 0))
            symbol = order.get('symbol', '?')
            price = order.get('avg_fill_price', 0)
            print(f'  {side.upper():<4} {float(qty):.6f} {symbol} @ \${price:,.2f}')
    else:
        print('No trades executed (portfolio already balanced or in cash regime)')

    # Stop streaming
    adapter.stop_streaming()

    print()
    print('[+] Rebalance complete')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
