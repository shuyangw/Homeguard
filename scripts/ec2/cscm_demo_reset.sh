#!/bin/bash
# CSCM Demo Portfolio Reset
#
# Resets the CSCM demo portfolio to a blank state with optimal configuration.
# Clears all positions, trade history, and persisted state.
#
# Optimal Config:
#   - Initial Cash: $100,000
#   - Top N: 5 positions
#   - Allocation: 18% per position
#   - Trailing Stop: 8%
#   - Profit Target: 20%
#
# Usage:
#   ./cscm_demo_reset.sh           # Reset and restart service
#   ./cscm_demo_reset.sh --no-start  # Reset only, don't restart
#   cscm-demo-reset                # (alias)

set -e

NO_START=false
if [[ "$1" == "--no-start" ]]; then
    NO_START=true
fi

echo "=========================================="
echo "CSCM Demo Portfolio Reset"
echo "=========================================="
echo ""

# Stop service if running
if systemctl is-active --quiet homeguard-cscm-demo 2>/dev/null; then
    echo "[*] Stopping CSCM demo service..."
    sudo systemctl stop homeguard-cscm-demo
    sleep 2
fi

# Clear persisted state files
STATE_DIR="$HOME/.homeguard/demo"
echo "[*] Clearing persisted state..."

if [[ -f "$STATE_DIR/portfolio_state.json" ]]; then
    rm -f "$STATE_DIR/portfolio_state.json"
    echo "    [-] Removed portfolio_state.json"
fi

if [[ -f "$STATE_DIR/cscm_adapter_state.json" ]]; then
    rm -f "$STATE_DIR/cscm_adapter_state.json"
    echo "    [-] Removed cscm_adapter_state.json"
fi

# Clear trade logs (optional - keep for audit trail)
# Uncomment to also clear trade history:
# if [[ -d "$STATE_DIR/trades" ]]; then
#     rm -rf "$STATE_DIR/trades"
#     echo "    [-] Removed trade logs"
# fi

echo ""

# Reset portfolio with optimal config
cd ~/Homeguard

echo "[*] Initializing fresh portfolio with optimal config..."
python3 -c "
import sys
sys.path.insert(0, '.')

from src.trading.demo import DemoBroker

# Optimal CSCM configuration
INITIAL_CASH = 100000.0
SLIPPAGE_BPS = 5.0
FEE_BPS = 10.0

# Create fresh broker with optimal settings
broker = DemoBroker(
    initial_cash=INITIAL_CASH,
    slippage_bps=SLIPPAGE_BPS,
    fee_bps=FEE_BPS,
)

# Force save clean state
broker._save_state()

print()
print('Portfolio initialized:')
print(f'  Cash:      \${INITIAL_CASH:,.2f}')
print(f'  Positions: 0')
print(f'  Slippage:  {SLIPPAGE_BPS} bps')
print(f'  Fees:      {FEE_BPS} bps')
print()
print('Optimal strategy config (applied at runtime):')
print('  Top N:         5 positions')
print('  Allocation:    18% per position')
print('  Trailing Stop: 8%')
print('  Profit Target: 20%')
print('  Rebalance:     Sunday 00:00 UTC')
"

echo ""

# Restart service if requested
if [[ "$NO_START" == false ]]; then
    if systemctl is-enabled --quiet homeguard-cscm-demo 2>/dev/null; then
        echo "[*] Restarting CSCM demo service..."
        sudo systemctl start homeguard-cscm-demo
        sleep 3

        if systemctl is-active --quiet homeguard-cscm-demo; then
            echo -e "[+] Service: \033[1;32mRUNNING\033[0m"
        else
            echo -e "[-] Service: \033[1;31mFAILED TO START\033[0m"
            echo "    Check logs: sudo journalctl -u homeguard-cscm-demo -n 20"
        fi
    else
        echo "[!] Service not enabled. Enable with:"
        echo "    sudo systemctl enable homeguard-cscm-demo"
        echo "    sudo systemctl start homeguard-cscm-demo"
    fi
else
    echo "[*] Service not restarted (--no-start flag)"
fi

echo ""
echo "=========================================="
echo "Reset complete"
echo "=========================================="
