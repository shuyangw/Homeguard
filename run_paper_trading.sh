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

# Run the script with all passed arguments
$PYTHON_CMD "scripts/trading/run_live_paper_trading.py" "$@"

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
