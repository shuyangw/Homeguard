#!/bin/bash
#
# Toggle Strategy - Enable/Disable trading strategies
#
# Usage:
#   ./toggle_strategy.sh status                    # Show current status
#   ./toggle_strategy.sh omr enable                # Enable OMR strategy
#   ./toggle_strategy.sh mp disable                # Disable MP (keep positions)
#   ./toggle_strategy.sh mp disable --close-positions  # Disable and close
#   ./toggle_strategy.sh mp close-orphaned         # Close orphaned positions
#
# This script modifies config/trading/strategy_toggle.yaml
# The trading process will pick up changes on next cycle
#

set -e

# Configuration
REPO_DIR="${REPO_DIR:-$HOME/Homeguard}"
TOGGLE_FILE="$REPO_DIR/config/trading/strategy_toggle.yaml"
STATE_FILE="$REPO_DIR/data/trading/strategy_positions.json"

# Auto-detect Python: prefer venv, then system python
if [ -f "$REPO_DIR/venv/bin/python" ]; then
    PYTHON_CMD="${PYTHON_CMD:-$REPO_DIR/venv/bin/python}"
elif [ -f "$HOME/Homeguard/venv/bin/python" ]; then
    PYTHON_CMD="${PYTHON_CMD:-$HOME/Homeguard/venv/bin/python}"
else
    PYTHON_CMD="${PYTHON_CMD:-python3}"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if running from repo or need to find it
if [ ! -f "$TOGGLE_FILE" ]; then
    # Try current directory
    if [ -f "config/trading/strategy_toggle.yaml" ]; then
        REPO_DIR="."
        TOGGLE_FILE="config/trading/strategy_toggle.yaml"
        STATE_FILE="data/trading/strategy_positions.json"
    else
        echo -e "${RED}Error: Cannot find strategy_toggle.yaml${NC}"
        echo "Run from Homeguard directory or set REPO_DIR"
        exit 1
    fi
fi

show_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  status                     Show strategy status"
    echo "  <strategy> enable          Enable a strategy"
    echo "  <strategy> disable         Disable a strategy (keep positions)"
    echo "  <strategy> disable --close-positions"
    echo "                             Disable and close all positions"
    echo "  <strategy> close-orphaned  Close orphaned positions"
    echo ""
    echo "Strategies: omr, mp"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 omr enable"
    echo "  $0 mp disable --close-positions"
}

show_status() {
    echo -e "${BLUE}Strategy Toggle Status${NC}"
    echo "========================================"
    echo ""

    # Use Python to read and display status
    cd "$REPO_DIR"
    $PYTHON_CMD << 'EOF'
import sys
sys.path.insert(0, '.')

from src.trading.state import StrategyStateManager

try:
    manager = StrategyStateManager()
    manager.print_status()
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
}

toggle_strategy() {
    local strategy="$1"
    local action="$2"
    local close_positions="$3"

    # Validate strategy name
    if [[ "$strategy" != "omr" && "$strategy" != "mp" ]]; then
        echo -e "${RED}Error: Unknown strategy '$strategy'${NC}"
        echo "Valid strategies: omr, mp"
        exit 1
    fi

    cd "$REPO_DIR"

    case "$action" in
        enable)
            echo -e "${BLUE}Enabling $strategy strategy...${NC}"
            $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '.')
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()
manager.set_enabled('$strategy', True, modified_by='toggle_strategy.sh')
manager.set_shutdown_requested('$strategy', False)
print("Done. Strategy will be active on next trading cycle.")
print("Note: Process restart may be required for full effect.")
EOF
            echo -e "${GREEN}$strategy strategy enabled${NC}"
            ;;

        disable)
            if [[ "$close_positions" == "--close-positions" ]]; then
                echo -e "${YELLOW}Disabling $strategy and closing positions...${NC}"
                $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '.')
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()

# Set shutdown requested first
manager.set_shutdown_requested('$strategy', True)
print("Shutdown flag set. Waiting for current execution to complete...")

# Get positions to close
positions = manager.get_positions('$strategy')
if positions:
    print(f"Positions to close: {list(positions.keys())}")
    print("")
    print("WARNING: Position closing requires broker connection.")
    print("Run the following to close positions:")
    print(f"  python scripts/trading/close_strategy_positions.py --strategy $strategy")
    print("")
else:
    print("No positions to close.")

# Disable the strategy
manager.set_enabled('$strategy', False, modified_by='toggle_strategy.sh')
manager.set_shutdown_requested('$strategy', False)
print(f"Strategy '$strategy' disabled.")
EOF
            else
                echo -e "${BLUE}Disabling $strategy (keeping positions open)...${NC}"
                $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '.')
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()
positions = manager.get_positions('$strategy')

if positions:
    print(f"Warning: {len(positions)} positions will become orphaned:")
    for symbol, data in positions.items():
        print(f"  {symbol}: {data['qty']} shares")
    print("")

manager.set_enabled('$strategy', False, modified_by='toggle_strategy.sh')
print(f"Strategy '$strategy' disabled.")
print("Note: Positions remain open. Use --close-positions to close them.")
EOF
            fi
            echo -e "${GREEN}$strategy strategy disabled${NC}"
            ;;

        close-orphaned)
            echo -e "${YELLOW}Closing orphaned positions for $strategy...${NC}"
            $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '.')
from src.trading.state import StrategyStateManager

manager = StrategyStateManager()

# Check if strategy is disabled
if manager.is_enabled('$strategy'):
    print("Error: Strategy is enabled. Disable it first or use normal position management.")
    sys.exit(1)

positions = manager.get_positions('$strategy')
if not positions:
    print("No orphaned positions found.")
    sys.exit(0)

print(f"Orphaned positions to close: {list(positions.keys())}")
print("")
print("WARNING: Position closing requires broker connection.")
print("Run the following to close positions:")
print(f"  python scripts/trading/close_strategy_positions.py --strategy $strategy")
EOF
            ;;

        *)
            echo -e "${RED}Error: Unknown action '$action'${NC}"
            echo "Valid actions: enable, disable, close-orphaned"
            exit 1
            ;;
    esac
}

# Main
case "$1" in
    status)
        show_status
        ;;
    omr|mp)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Missing action${NC}"
            show_usage
            exit 1
        fi
        toggle_strategy "$1" "$2" "$3"
        ;;
    -h|--help|help)
        show_usage
        ;;
    "")
        show_status
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        show_usage
        exit 1
        ;;
esac
