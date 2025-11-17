#!/bin/bash
#
# Start or Restart Trading Bot on EC2
#
# This script should be run on the EC2 instance in the home directory.
# It will start the trading bot if not running, or restart it if already running.
#
# Prerequisites:
#   - .env file in Homeguard repository with Alpaca credentials
#   - Homeguard repository cloned in ~/Homeguard
#
# Usage:
#   ~/start_or_restart_bot.sh
#

set -e

# Configuration
REPO_DIR="$HOME/Homeguard"
ENV_FILE="$REPO_DIR/.env"
LOG_DIR="$HOME/logs/live_trading/paper"
TRADING_SCRIPT="$REPO_DIR/scripts/trading/run_live_paper_trading.py"
PYTHON_CMD="$REPO_DIR/venv/bin/python"
PROCESS_NAME="run_live_paper_trading.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Trading Bot Start/Restart Script"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Error: .env file not found at $ENV_FILE${NC}"
    echo "Please create .env file with your Alpaca credentials:"
    echo ""
    echo "ALPACA_PAPER_KEY_ID=your_api_key"
    echo "ALPACA_PAPER_SECRET_KEY=your_secret_key"
    echo ""
    echo "Or check the Terraform deployment created it correctly."
    exit 1
fi

# Check if repository exists
if [ ! -d "$REPO_DIR" ]; then
    echo -e "${RED}❌ Error: Homeguard repository not found at $REPO_DIR${NC}"
    echo "Please clone the repository first:"
    echo "  git clone https://github.com/shuyangw/Homeguard.git ~/Homeguard"
    exit 1
fi

# Check if trading script exists
if [ ! -f "$TRADING_SCRIPT" ]; then
    echo -e "${RED}❌ Error: Trading script not found at $TRADING_SCRIPT${NC}"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Configuration:${NC}"
echo "  Repository: $REPO_DIR"
echo "  Environment: $ENV_FILE"
echo "  Log directory: $LOG_DIR"
echo "  Trading script: $TRADING_SCRIPT"
echo ""

# Check if bot is already running
BOT_PID=$(pgrep -f "$PROCESS_NAME" || true)

if [ -n "$BOT_PID" ]; then
    echo -e "${YELLOW}⚠️  Trading bot is currently running (PID: $BOT_PID)${NC}"
    echo "Stopping existing bot..."

    # Kill the process gracefully
    kill "$BOT_PID" 2>/dev/null || true

    # Wait for process to stop (max 10 seconds)
    for i in {1..10}; do
        if ! pgrep -f "$PROCESS_NAME" > /dev/null; then
            echo -e "${GREEN}✅ Bot stopped successfully${NC}"
            break
        fi
        sleep 1
    done

    # Force kill if still running
    if pgrep -f "$PROCESS_NAME" > /dev/null; then
        echo -e "${YELLOW}⚠️  Process still running, forcing stop...${NC}"
        pkill -9 -f "$PROCESS_NAME" || true
        sleep 2
    fi

    echo ""
else
    echo -e "${BLUE}ℹ️  Trading bot is not currently running${NC}"
    echo ""
fi

# Source environment variables
echo "Loading environment variables from .env..."
set -a  # Automatically export all variables
source "$ENV_FILE"
set +a
echo -e "${GREEN}✅ Environment loaded${NC}"
echo ""

# Start the trading bot
echo "Starting trading bot..."
echo ""

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TIMESTAMP}_bot_startup.log"

# Start the bot in background with nohup
nohup $PYTHON_CMD "$TRADING_SCRIPT" > "$LOG_FILE" 2>&1 &
NEW_PID=$!

# Wait a moment and check if process started successfully
sleep 2

if ps -p $NEW_PID > /dev/null; then
    echo -e "${GREEN}=========================================="
    echo "✅ Trading bot started successfully!"
    echo -e "==========================================${NC}"
    echo ""
    echo "  PID: $NEW_PID"
    echo "  Log file: $LOG_FILE"
    echo ""
    echo "To view logs in real-time:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop the bot:"
    echo "  kill $NEW_PID"
    echo ""
    echo "To check if bot is running:"
    echo "  pgrep -f '$PROCESS_NAME'"
    echo ""
else
    echo -e "${RED}=========================================="
    echo "❌ Failed to start trading bot"
    echo -e "==========================================${NC}"
    echo ""
    echo "Check the log file for errors:"
    echo "  cat $LOG_FILE"
    echo ""
    exit 1
fi

# Reset terminal colors
printf "${NC}"
