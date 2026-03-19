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
#   ./cscm_demo_reset.sh              # Reset with confirmation prompt
#   ./cscm_demo_reset.sh --force      # Reset without confirmation
#   ./cscm_demo_reset.sh --no-start   # Reset only, don't restart service
#   cscm-reset                        # (alias)

set -e

NO_START=false
FORCE=false

for arg in "$@"; do
    case $arg in
        --no-start)
            NO_START=true
            ;;
        --force|-f)
            FORCE=true
            ;;
    esac
done

echo "=========================================="
echo "CSCM Demo Portfolio Reset"
echo "=========================================="
echo ""
echo -e "\033[1;31m[!] WARNING: This will permanently delete:\033[0m"
echo "    - All current positions"
echo "    - All unrealized P&L"
echo "    - Realized P&L history"
echo "    - Entry price tracking"
echo "    - Adapter state (regime, trailing stops)"
echo ""
echo "    Portfolio will be reset to \$100,000 cash."
echo ""

# Confirm unless --force flag
if [[ "$FORCE" == false ]]; then
    read -p "Are you sure you want to reset? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Reset cancelled."
        exit 0
    fi
    echo ""
fi

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
