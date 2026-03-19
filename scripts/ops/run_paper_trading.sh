#!/bin/bash
# Live Paper Trading Launcher (Linux/Mac)
#
# Usage:
#   ./run_paper_trading.sh                          (Run MA Crossover continuous)
#   ./run_paper_trading.sh --once                   (Run once and exit)
#   ./run_paper_trading.sh --strategy omr           (Run OMR strategy)
#   ./run_paper_trading.sh --strategy triple-ma     (Run Triple MA strategy)
#   ./run_paper_trading.sh --universe leveraged     (Trade leveraged ETFs)
#   ./run_paper_trading.sh --no-intraday-prefetch  (Disable 3:45PM data pre-fetch)
#   ./run_paper_trading.sh --background             (Run in background with nohup)
#   ./run_paper_trading.sh --help                   (Show all options)

echo "================================================================================"
echo "                     HOMEGUARD LIVE PAPER TRADING"
echo "================================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found!"
    echo ""
    echo "Please create a .env file with your Alpaca credentials:"
    echo "  ALPACA_PAPER_KEY_ID=your_key_id"
    echo "  ALPACA_PAPER_SECRET_KEY=your_secret_key"
    echo "  ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets/v2"
    echo ""
    exit 1
fi

# Try to find Python from conda environment or system
PYTHON_CMD=""

# Option 1: Try conda environment (adjust path as needed)
if [ -f "$HOME/anaconda3/envs/fintech/bin/python" ]; then
    PYTHON_CMD="$HOME/anaconda3/envs/fintech/bin/python"
elif [ -f "$HOME/miniconda3/envs/fintech/bin/python" ]; then
    PYTHON_CMD="$HOME/miniconda3/envs/fintech/bin/python"
# Option 2: Try conda activate
elif command -v conda &> /dev/null; then
    # Source conda for this shell
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    # Try to activate fintech environment
    if conda activate fintech 2>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "[WARNING] Could not activate fintech conda environment"
    fi
fi

# Option 3: Fall back to system Python
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "[ERROR] Python not found!"
        echo "Please install Python or set up the fintech conda environment."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Parse arguments for --background flag
BACKGROUND_MODE=false
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --background)
            BACKGROUND_MODE=true
            shift
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if background mode is enabled
if [ "$BACKGROUND_MODE" = true ]; then
    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Generate log filename with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="logs/paper_trading_${TIMESTAMP}.log"
    PIDFILE="logs/paper_trading.pid"

    # Check if already running
    if [ -f "$PIDFILE" ]; then
        OLD_PID=$(cat "$PIDFILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "[WARNING] Paper trading is already running (PID: $OLD_PID)"
            echo "To stop it: kill $OLD_PID"
            echo "Or run: ./stop_paper_trading.sh"
            exit 1
        else
            echo "[INFO] Removing stale PID file"
            rm "$PIDFILE"
        fi
    fi

    echo "[INFO] Starting paper trading in background mode..."
    echo "[INFO] Log file: $LOGFILE"
    echo "[INFO] PID file: $PIDFILE"
    echo ""

    # Run in background using nohup
    nohup $PYTHON_CMD "scripts/trading/run_live_paper_trading.py" "${OTHER_ARGS[@]}" >> "$LOGFILE" 2>&1 &
    BG_PID=$!

    # Save PID to file
    echo $BG_PID > "$PIDFILE"

    # Wait a moment to check if process started successfully
    sleep 2
    if ps -p $BG_PID > /dev/null 2>&1; then
        echo "================================================================================"
        echo "[SUCCESS] Paper trading started in background!"
        echo "================================================================================"
        echo ""
        echo "Process Details:"
        echo "  - PID: $BG_PID"
        echo "  - Logs: $LOGFILE"
        echo "  - View logs: tail -f $LOGFILE"
        echo ""
        echo "To Stop:"
        echo "  kill $BG_PID"
        echo "  OR"
        echo "  ./stop_paper_trading.sh"
        echo ""
        echo "To Monitor:"
        echo "  tail -f $LOGFILE"
        echo ""
        echo "================================================================================"
    else
        echo "[ERROR] Failed to start paper trading process"
        rm "$PIDFILE"
        exit 1
    fi
else
    # Run in foreground mode
    $PYTHON_CMD "scripts/trading/run_live_paper_trading.py" "${OTHER_ARGS[@]}"

    # Check exit code
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "================================================================================"
        echo "[ERROR] Script exited with error code $EXIT_CODE"
        echo "================================================================================"
        exit $EXIT_CODE
    fi

    echo ""
    echo "================================================================================"
    echo "Script completed successfully"
    echo "================================================================================"
fi
