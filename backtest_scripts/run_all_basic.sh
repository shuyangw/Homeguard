#!/usr/bin/env bash
# Run All Basic Backtests
# Executes all 5 basic backtest scenarios sequentially
#
# Usage: ./run_all_basic.sh [OPTIONS]
#
# Arguments (all optional):
#   --fees VALUE        Transaction fees as decimal (e.g., 0.001 = 0.1%)
#   --capital VALUE     Initial capital amount (e.g., 100000)
#   --start DATE        Start date in YYYY-MM-DD format
#   --end DATE          End date in YYYY-MM-DD format
#   --quantstats        Enable QuantStats reporting (tearsheets)
#   --verbosity LEVEL   Logging verbosity: 0=minimal, 1=normal, 2=detailed, 3=verbose
#   --parallel          Enable parallel sweep execution (default)
#   --no-parallel       Disable parallel execution (run sequentially)
#   --max-workers N     Number of parallel workers (default: 4)
#
# Examples:
#   ./run_all_basic.sh --fees 0.002 --capital 50000
#   ./run_all_basic.sh --quantstats --verbosity 2
#   ./run_all_basic.sh --no-parallel
#   ./run_all_basic.sh --max-workers 8

# ============================================================================
# DEFAULT PARAMETERS - Can be overridden by command-line arguments
# ============================================================================

# Initial capital for all backtests (default: 100000)
export BACKTEST_CAPITAL=100000

# Transaction fees as decimal (default: 0 = no fees)
export BACKTEST_FEES=0

# Start date for all backtests (YYYY-MM-DD format)
export BACKTEST_START="2023-01-01"

# End date for all backtests (YYYY-MM-DD format)
export BACKTEST_END="2024-01-01"

# QuantStats reporting enabled by default
export BACKTEST_QUANTSTATS="--quantstats"

# Visualization enabled by default
export BACKTEST_VISUALIZE="--visualize"

# Logging verbosity (0-3, default: 1)
export BACKTEST_VERBOSITY=1

# Parallel execution (default: enabled with --parallel flag)
export BACKTEST_PARALLEL="--parallel"

# Maximum parallel workers (default: 4)
export BACKTEST_MAX_WORKERS=4

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --quantstats)
            BACKTEST_QUANTSTATS="--quantstats"
            shift
            ;;
        --verbosity)
            BACKTEST_VERBOSITY="$2"
            shift 2
            ;;
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
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Define strategies to run
declare -a STRATEGIES=(
    "MovingAverageCrossover:AAPL:MA Crossover"
    "RSIMeanReversion:AAPL:RSI Mean Reversion"
    "MeanReversion:MSFT:Bollinger Bands"
    "MomentumStrategy:GOOGL:MACD Momentum"
    "BreakoutStrategy:AMZN:Breakout Strategy"
)

echo "================================================================================"
echo "                        RUN ALL BASIC BACKTESTS"
echo "================================================================================"
echo
echo "Configuration:"
echo "  Initial Capital: \$$BACKTEST_CAPITAL"
echo "  Transaction Fees: $BACKTEST_FEES"
echo "  Period: $BACKTEST_START to $BACKTEST_END"
if [[ "$BACKTEST_QUANTSTATS" == "--quantstats" ]]; then
    echo "  QuantStats Reports: ENABLED"
else
    echo "  QuantStats Reports: DISABLED"
fi
echo
echo "This will run 5 basic backtests sequentially:"
echo "  1. Simple MA Crossover (AAPL)"
echo "  2. RSI Mean Reversion (AAPL)"
echo "  3. Bollinger Bands (MSFT)"
echo "  4. MACD Momentum (GOOGL)"
echo "  5. Breakout Strategy (AMZN)"
echo
echo "Estimated time: 5-10 minutes"
echo

COUNTER=1
TOTAL=5

for strategy_info in "${STRATEGIES[@]}"; do
    IFS=':' read -r strategy symbol name <<< "$strategy_info"

    echo "[$COUNTER/$TOTAL] Running $name..."
    echo "========================================"
    echo "Strategy: $name"
    echo "Symbol: $symbol"
    echo "Period: $BACKTEST_START to $BACKTEST_END"
    echo "========================================"
    echo

    python src/backtest_runner.py \
        --strategy "$strategy" \
        --symbols "$symbol" \
        --start "$BACKTEST_START" \
        --end "$BACKTEST_END" \
        --capital "$BACKTEST_CAPITAL" \
        --fees "$BACKTEST_FEES" \
        $BACKTEST_VISUALIZE \
        $BACKTEST_QUANTSTATS \
        --verbosity "$BACKTEST_VERBOSITY"

    echo
    echo "Backtest complete!"
    echo

    ((COUNTER++))
done

echo "================================================================================"
echo "                        ALL BASIC BACKTESTS COMPLETE!"
echo "================================================================================"
echo
echo "All 5 basic backtests have been completed."
if [[ "$BACKTEST_QUANTSTATS" == "--quantstats" ]]; then
    echo
    echo "QuantStats tearsheets have been generated for each strategy."
    echo "Check the configured log_output_dir in settings.ini for reports."
    echo "Open tearsheet.html files in your browser to view detailed performance."
fi
echo
echo "Review the results above to compare strategy performance."
echo
