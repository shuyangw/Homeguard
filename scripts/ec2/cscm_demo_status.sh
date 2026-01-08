#!/bin/bash
# CSCM Demo Trading Status
#
# Shows current portfolio value, positions, and regime status.
# Uses DemoBroker with Binance streaming (self-contained paper trading).
#
# Usage:
#   ./cscm_demo_status.sh
#   cscm-demo-status  (alias)

set -e

echo "=========================================="
echo "CSCM Demo Trading Status"
echo "=========================================="
echo ""

# Check service status
if systemctl is-active --quiet homeguard-cscm-demo 2>/dev/null; then
    echo -e "Service:        \033[1;32mRUNNING\033[0m"
else
    echo -e "Service:        \033[1;31mSTOPPED\033[0m"
fi

echo ""

# Get portfolio status via Python
cd ~/Homeguard

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from src.trading.demo import DemoBroker

    broker = DemoBroker()
    summary = broker.get_account_summary()

    print('--- Portfolio ---')
    print(f'Total Value:    \${summary[\"total_value\"]:,.2f}')
    print(f'Cash:           \${summary[\"cash\"]:,.2f}')
    print(f'Realized P&L:   \${summary[\"realized_pnl\"]:,.2f}')
    print(f'Unrealized P&L: \${summary[\"unrealized_pnl\"]:,.2f}')
    print()
    print('--- Status ---')
    print(f'Positions:      {len(summary[\"positions\"])}')
    print(f'Streaming:      {\"Yes\" if summary.get(\"streaming\") else \"No\"}')
    print(f'Bars Buffered:  {summary.get(\"bars_buffered\", 0)}')
    print(f'Orders Placed:  {summary.get(\"orders_placed\", 0)}')

    # Execution config
    exec_config = summary.get('execution_config', {})
    if exec_config:
        print()
        print('--- Execution ---')
        print(f'Slippage:       {exec_config.get(\"slippage_bps\", 0):.1f} bps')
        print(f'Fees:           {exec_config.get(\"fee_bps\", 0):.1f} bps')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Recent Logs (last 10 lines)"
echo "=========================================="
sudo journalctl -u homeguard-cscm-demo -n 10 --no-pager --output=short 2>/dev/null || echo "No logs available"
