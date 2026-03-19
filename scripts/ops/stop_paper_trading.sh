#!/bin/bash
# Stop Background Paper Trading Process (Linux/Mac)
#
# Usage:
#   ./stop_paper_trading.sh

echo "================================================================================"
echo "           STOPPING HOMEGUARD PAPER TRADING"
echo "================================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

PIDFILE="logs/paper_trading.pid"

# Check if PID file exists
if [ ! -f "$PIDFILE" ]; then
    echo "[WARNING] PID file not found: $PIDFILE"
    echo "[INFO] Paper trading may not be running in background"
    echo ""
    echo "Checking for any running paper trading processes..."

    # Try to find by process name
    PIDS=$(ps aux | grep "run_live_paper_trading.py" | grep -v grep | awk '{print $2}')

    if [ -z "$PIDS" ]; then
        echo "[INFO] No paper trading processes found"
        exit 0
    else
        echo "[FOUND] Paper trading processes:"
        ps aux | grep "run_live_paper_trading.py" | grep -v grep
        echo ""
        read -p "Kill these processes? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill $PIDS
            echo "[SUCCESS] Processes killed"
        else
            echo "[CANCELLED] No processes killed"
        fi
        exit 0
    fi
fi

# Read PID from file
PID=$(cat "$PIDFILE")

# Check if process is running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "[INFO] Found paper trading process (PID: $PID)"

    # Get process info
    echo ""
    echo "Process Info:"
    ps -p "$PID" -o pid,comm,etime,cmd
    echo ""

    # Kill process
    echo "[INFO] Sending SIGTERM to process..."
    kill "$PID"

    # Wait for process to stop (max 10 seconds)
    TIMEOUT=10
    COUNTER=0
    while ps -p "$PID" > /dev/null 2>&1 && [ $COUNTER -lt $TIMEOUT ]; do
        sleep 1
        COUNTER=$((COUNTER + 1))
        echo -n "."
    done
    echo ""

    # Check if process stopped
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[WARNING] Process did not stop gracefully"
        echo "[INFO] Sending SIGKILL..."
        kill -9 "$PID"
        sleep 1
    fi

    # Verify stopped
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[ERROR] Failed to stop process"
        exit 1
    else
        echo ""
        echo "================================================================================"
        echo "[SUCCESS] Paper trading stopped"
        echo "================================================================================"
        rm "$PIDFILE"
    fi
else
    echo "[INFO] Process (PID: $PID) is not running"
    echo "[INFO] Removing stale PID file"
    rm "$PIDFILE"
fi

echo ""
echo "Latest log files:"
ls -lht logs/paper_trading_*.log 2>/dev/null | head -3
echo ""
