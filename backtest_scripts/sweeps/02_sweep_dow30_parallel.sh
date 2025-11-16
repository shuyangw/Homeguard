#!/bin/bash
# Sweep Backtest Across DOW 30 (Parallel Execution)
# Tests breakout strategy on DOW 30 stocks using parallel processing

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

echo ""
echo "========================================"
echo "Sweep Backtest: DOW 30 (Parallel)"
echo "Strategy: BreakoutStrategy"
echo "Period: $BACKTEST_START to $BACKTEST_END"
echo "Universe: DOW30 (30 symbols)"
echo "Mode: Parallel (4 workers)"
echo "========================================"
echo ""

python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy BreakoutStrategy \
  --universe DOW30 \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --parallel \
  --max-workers 4 \
  --start "$BACKTEST_START" \
  --end "$BACKTEST_END" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sweep_dow30_breakout" \
  --sort-by "Total Return [%]" \
  --top-n 10 \
  --verbosity "$BACKTEST_VERBOSITY"

echo ""
echo "Sweep complete! Check logs directory for CSV and HTML reports."
