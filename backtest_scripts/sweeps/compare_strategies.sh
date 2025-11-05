#!/bin/bash
# Compare Multiple Strategies on Same Universe
# Runs several strategies on one universe and saves results for comparison

if [ -z "$1" ]; then
    echo "Error: Universe name required"
    echo ""
    echo "Usage: ./compare_strategies.sh UNIVERSE [START_DATE] [END_DATE]"
    echo ""
    echo "Example: ./compare_strategies.sh FAANG 2023-01-01 2024-01-01"
    exit 1
fi

UNIVERSE=$1

# Use provided dates or defaults
if [ -n "$2" ]; then
    START_DATE=$2
else
    START_DATE=2023-01-01
fi

if [ -n "$3" ]; then
    END_DATE=$3
else
    END_DATE=2024-01-01
fi

# Environment variable defaults
BACKTEST_CAPITAL=${BACKTEST_CAPITAL:-100000}
BACKTEST_PARALLEL=${BACKTEST_PARALLEL:---parallel}
BACKTEST_MAX_WORKERS=${BACKTEST_MAX_WORKERS:-4}
BACKTEST_FEES=${BACKTEST_FEES:-0}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "========================================"
echo "Strategy Comparison on $UNIVERSE"
echo "========================================"
echo "Period: $START_DATE to $END_DATE"
echo ""
echo "Testing the following strategies:"
echo "  1. MovingAverageCrossover (MA Cross)"
echo "  2. BreakoutStrategy (Breakout)"
echo "  3. MeanReversion (Bollinger Bands)"
echo "  4. MomentumStrategy (MACD)"
echo "  5. RSIMeanReversion (RSI)"
echo "========================================"
echo ""

# Strategy 1: Moving Average Crossover
echo ""
echo "[1/5] Testing MovingAverageCrossover..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy MovingAverageCrossover \
  --universe "$UNIVERSE" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "compare_${UNIVERSE}_MA_Crossover" \
  --parallel \
  --verbosity 0

# Strategy 2: Breakout
echo ""
echo "[2/5] Testing BreakoutStrategy..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy BreakoutStrategy \
  --universe "$UNIVERSE" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "compare_${UNIVERSE}_Breakout" \
  --parallel \
  --verbosity 0

# Strategy 3: Mean Reversion
echo ""
echo "[3/5] Testing MeanReversion..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy MeanReversion \
  --universe "$UNIVERSE" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "compare_${UNIVERSE}_MeanReversion" \
  --parallel \
  --verbosity 0

# Strategy 4: Momentum (MACD)
echo ""
echo "[4/5] Testing MomentumStrategy..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy MomentumStrategy \
  --universe "$UNIVERSE" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "compare_${UNIVERSE}_MACD" \
  --parallel \
  --verbosity 0

# Strategy 5: RSI Mean Reversion
echo ""
echo "[5/5] Testing RSIMeanReversion..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy RSIMeanReversion \
  --universe "$UNIVERSE" \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "compare_${UNIVERSE}_RSI" \
  --parallel \
  --verbosity 0

echo ""
echo "========================================"
echo "All strategy comparisons complete!"
echo "========================================"
echo ""
echo "Results saved to logs directory with prefix: compare_${UNIVERSE}_"
echo ""
echo "Compare the HTML reports to see which strategy performs best on $UNIVERSE"
echo ""
