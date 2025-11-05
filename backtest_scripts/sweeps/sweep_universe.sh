#!/bin/bash
# Generic Universe Sweep Script
# Usage: ./sweep_universe.sh UNIVERSE STRATEGY [START_DATE] [END_DATE]
# Example: ./sweep_universe.sh FAANG MovingAverageCrossover 2023-01-01 2024-01-01

if [ -z "$1" ]; then
    echo "Error: Universe name required"
    echo ""
    echo "Usage: ./sweep_universe.sh UNIVERSE STRATEGY [START_DATE] [END_DATE]"
    echo ""
    echo "Examples:"
    echo "  ./sweep_universe.sh FAANG MovingAverageCrossover"
    echo "  ./sweep_universe.sh DOW30 BreakoutStrategy 2023-01-01 2024-01-01"
    echo "  ./sweep_universe.sh TECH_GIANTS MeanReversion"
    echo ""
    echo "Run './list_universes.sh' to see available universes"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Strategy name required"
    echo ""
    echo "Available strategies:"
    echo "  - MovingAverageCrossover"
    echo "  - TripleMovingAverage"
    echo "  - MeanReversion"
    echo "  - RSIMeanReversion"
    echo "  - MomentumStrategy"
    echo "  - BreakoutStrategy"
    echo "  - VolatilityTargetedMomentum"
    echo "  - OvernightMeanReversion"
    exit 1
fi

UNIVERSE=$1
STRATEGY=$2

# Use provided dates or defaults
START_DATE=${3:-${BACKTEST_START:-2023-01-01}}
END_DATE=${4:-${BACKTEST_END:-2024-01-01}}

# Use environment variables or defaults
CAPITAL=${BACKTEST_CAPITAL:-100000}
FEES=${BACKTEST_FEES:-0}
VERBOSITY=${BACKTEST_VERBOSITY:-1}
PARALLEL=${BACKTEST_PARALLEL:---parallel}
MAX_WORKERS=${BACKTEST_MAX_WORKERS:-4}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "========================================"
echo "Universe Sweep"
echo "========================================"
echo "Universe: $UNIVERSE"
echo "Strategy: $STRATEGY"
echo "Period: $START_DATE to $END_DATE"
echo "Capital: \$$CAPITAL"
echo "Fees: $FEES"
echo "========================================"
echo ""

python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe "$UNIVERSE" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$CAPITAL" \
  --fees "$FEES" \
  --run-name "sweep_${UNIVERSE}_${STRATEGY}" \
  --sort-by "Sharpe Ratio" \
  --parallel \
  --verbosity "$VERBOSITY"

echo ""
echo "Sweep complete! Check logs directory for CSV and HTML reports."
