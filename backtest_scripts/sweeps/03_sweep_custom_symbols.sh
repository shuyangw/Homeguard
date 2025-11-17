#!/bin/bash
# Sweep Backtest Using Custom Symbols File
# Tests strategy across symbols loaded from a custom file

# Environment variable defaults
BACKTEST_START=${BACKTEST_START:-2023-01-01}
BACKTEST_END=${BACKTEST_END:-2024-01-01}
BACKTEST_CAPITAL=${BACKTEST_CAPITAL:-100000}
BACKTEST_FEES=${BACKTEST_FEES:-0}
BACKTEST_VERBOSITY=${BACKTEST_VERBOSITY:-1}
BACKTEST_PARALLEL=${BACKTEST_PARALLEL:---parallel}
BACKTEST_MAX_WORKERS=${BACKTEST_MAX_WORKERS:-4}

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS (optional overrides)
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            BACKTEST_PARALLEL="--parallel"
            shift
            ;;
        --no-parallel)
            BACKTEST_PARALLEL=""
            shift
            ;;
        --max-workers)
            BACKTEST_MAX_WORKERS="$2"
            shift 2
            ;;
        --fees)
            BACKTEST_FEES="$2"
            shift 2
            ;;
        --capital)
            BACKTEST_CAPITAL="$2"
            shift 2
            ;;
        --start)
            BACKTEST_START="$2"
            shift 2
            ;;
        --end)
            BACKTEST_END="$2"
            shift 2
            ;;
        --verbosity)
            BACKTEST_VERBOSITY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create example watchlist if it doesn't exist
WATCHLIST="$SCRIPT_DIR/../my_watchlist.txt"
if [ ! -f "$WATCHLIST" ]; then
    echo "Creating example watchlist..."
    cat > "$WATCHLIST" << 'EOF'
AAPL
MSFT
GOOGL
AMZN
TSLA
NVDA
META
EOF
fi

echo ""
echo "========================================"
echo "Sweep Backtest: Custom Symbols File"
echo "Strategy: MeanReversion"
echo "Period: $BACKTEST_START to $BACKTEST_END"
echo "Source: my_watchlist.txt"
echo "========================================"
echo ""

python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy MeanReversion \
  --symbols-file "$WATCHLIST" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$BACKTEST_START" \
  --end "$BACKTEST_END" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sweep_custom_watchlist" \
  --sort-by "Sharpe Ratio" \
  --verbosity "$BACKTEST_VERBOSITY"

echo ""
echo "Sweep complete! Results saved to logs directory."
echo ""
echo "Tip: Edit my_watchlist.txt to customize the symbols list."
