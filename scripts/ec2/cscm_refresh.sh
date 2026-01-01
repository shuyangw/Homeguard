#!/bin/bash
# CSCM Portfolio Refresh
#
# Force refresh portfolio value (ignores equity hours).
# Use this for on-demand portfolio updates.
#
# Usage:
#   ./cscm_refresh.sh
#   cscm-refresh  (alias)

set -e

echo "Refreshing CSCM portfolio value..."
echo ""

cd ~/Homeguard

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
    from src.trading.portfolio.tracker import PortfolioTracker

    adapter = CSCMPaperAdapter()
    tracker = PortfolioTracker(adapter)

    # Force refresh
    value = tracker.update_value(force=True)

    print(f'Portfolio value updated: \${value:,.2f}')

    # Show positions
    status = tracker.get_status()
    if status['positions_count'] > 0:
        print()
        print('Current Positions:')
        for symbol, qty in status['positions'].items():
            print(f'  {symbol}: {qty:.6f}')

except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

echo ""
echo "[+] Refresh complete"
