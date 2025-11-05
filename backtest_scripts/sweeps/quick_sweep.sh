#!/bin/bash
# Quick Sweep - Fast testing with FAANG stocks
# Uses FAANG (5 stocks) for quick validation and testing

if [ -z "$1" ]; then
    echo "Error: Strategy name required"
    echo ""
    echo "Usage: ./quick_sweep.sh STRATEGY [PARAMS]"
    echo ""
    echo "Examples:"
    echo "  ./quick_sweep.sh MovingAverageCrossover"
    echo "  ./quick_sweep.sh BreakoutStrategy"
    echo "  ./quick_sweep.sh MeanReversion \"window=30,num_std=2.5\""
    echo ""
    exit 1
fi

STRATEGY=$1
PARAMS=$2

# Environment variable defaults
BACKTEST_CAPITAL=${BACKTEST_CAPITAL:-100000}
BACKTEST_FEES=${BACKTEST_FEES:-0}
BACKTEST_PARALLEL=${BACKTEST_PARALLEL:---parallel}
BACKTEST_MAX_WORKERS=${BACKTEST_MAX_WORKERS:-4}
BACKTEST_VERBOSITY=${BACKTEST_VERBOSITY:-1}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "========================================"
echo "Quick Sweep (FAANG)"
echo "========================================"
echo "Strategy: $STRATEGY"
[ -n "$PARAMS" ] && echo "Parameters: $PARAMS"
echo "Universe: FAANG (5 stocks)"
echo "Period: Last year (2023-2024)"
if [ "$BACKTEST_PARALLEL" = "--parallel" ]; then
    echo "Mode: Parallel ($BACKTEST_MAX_WORKERS workers)"
else
    echo "Mode: Sequential"
fi
echo "========================================"
echo ""

if [ -z "$PARAMS" ]; then
    python "$SCRIPT_DIR/../../src/backtest_runner.py" \
      --strategy "$STRATEGY" \
      --universe FAANG \
      --sweep \
      $BACKTEST_PARALLEL \
      --max-workers "$BACKTEST_MAX_WORKERS" \
      --start 2023-01-01 \
      --end 2024-01-01 \
      --capital "$BACKTEST_CAPITAL" \
      --fees "$BACKTEST_FEES" \
      --run-name "quick_sweep_${STRATEGY}" \
      --verbosity "$BACKTEST_VERBOSITY"
else
    python "$SCRIPT_DIR/../../src/backtest_runner.py" \
      --strategy "$STRATEGY" \
      --universe FAANG \
      --sweep \
      $BACKTEST_PARALLEL \
      --max-workers "$BACKTEST_MAX_WORKERS" \
      --start 2023-01-01 \
      --end 2024-01-01 \
      --capital "$BACKTEST_CAPITAL" \
      --fees "$BACKTEST_FEES" \
      --params "$PARAMS" \
      --run-name "quick_sweep_${STRATEGY}" \
      --verbosity "$BACKTEST_VERBOSITY"
fi

echo ""
echo "Quick sweep complete! Check logs for results."
