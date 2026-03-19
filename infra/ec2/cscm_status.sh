#!/bin/bash
# CSCM Paper Trading Status
#
# Shows current portfolio value, positions, and regime status.
#
# Usage:
#   ./cscm_status.sh
#   cscm-status  (alias)

set -e

echo "=========================================="
echo "CSCM Paper Trading Status"
echo "=========================================="
echo ""

# Check service status
if systemctl is-active --quiet homeguard-cscm-paper 2>/dev/null; then
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
    from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
    from src.trading.portfolio.tracker import PortfolioTracker

    adapter = CSCMPaperAdapter()
    tracker = PortfolioTracker(adapter)

    # Get status
    status = tracker.get_status()

    print('--- Portfolio ---')
    print(f'Value:          \${status[\"current_value\"]:,.2f}')
    print(f'Cash:           \${status[\"cash\"]:,.2f}')
    print(f'Positions:      {status[\"positions_count\"]}')
    print()
    print('--- Strategy ---')
    print(f'Regime:         {status[\"regime\"].upper()}')
    print(f'Last Update:    {status[\"last_update\"] or \"Never\"}')
    print(f'Equity Hours:   {\"Yes\" if status[\"is_equity_hours\"] else \"No\"}')

except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Recent Logs (last 10 lines)"
echo "=========================================="
sudo journalctl -u homeguard-cscm-paper -n 10 --no-pager --output=short 2>/dev/null || echo "No logs available"
