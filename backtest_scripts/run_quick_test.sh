#!/usr/bin/env bash
# Quick Test - Single Backtest with QuantStats Report
# Runs one fast backtest to verify system is working
#
# Usage: ./run_quick_test.sh [--symbol AAPL] [--fees 0.002] [--capital 50000]
#
# Arguments (all optional):
#   --symbol VALUE      Stock symbol to test (e.g., AAPL, MSFT)
#   --fees VALUE        Transaction fees as decimal (e.g., 0.001 = 0.1%)
#   --capital VALUE     Initial capital amount (e.g., 100000)
#   --start DATE        Start date in YYYY-MM-DD format
#   --end DATE          End date in YYYY-MM-DD format
#   --no-report         Skip QuantStats report generation (console output only)
#
# Examples:
#   ./run_quick_test.sh
#   ./run_quick_test.sh --symbol MSFT --fees 0.002
#   ./run_quick_test.sh --symbol AAPL --no-report

# ============================================================================
# DEFAULT PARAMETERS - Can be overridden by command-line arguments
# ============================================================================

# Symbol to backtest (default: AAPL)
QUICK_SYMBOL="AAPL"

# Initial capital (default: 100000)
QUICK_CAPITAL=100000

# Transaction fees as decimal (default: 0 = no fees)
QUICK_FEES=0

# Start date (YYYY-MM-DD format)
QUICK_START="2024-01-01"

# End date (YYYY-MM-DD format)
QUICK_END="2024-06-30"

# QuantStats report enabled by default
QUICK_QUANTSTATS="--quantstats"

# Visualization enabled by default
QUICK_VISUALIZE="--visualize"

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --symbol)
            QUICK_SYMBOL="$2"
            shift 2
            ;;
        --fees)
            QUICK_FEES="$2"
            shift 2
            ;;
        --capital)
            QUICK_CAPITAL="$2"
            shift 2
            ;;
        --start)
            QUICK_START="$2"
            shift 2
            ;;
        --end)
            QUICK_END="$2"
            shift 2
            ;;
        --no-report)
            QUICK_QUANTSTATS=""
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

# Change to repo root (parent directory of backtest_scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "================================================================================"
echo "                   QUICK TEST WITH QUANTSTATS REPORTING"
echo "================================================================================"
echo
echo "This will run a backtest with comprehensive QuantStats tearsheet:"
echo "  Strategy: Moving Average Crossover"
echo "  Symbol: $QUICK_SYMBOL"
echo "  Period: $QUICK_START to $QUICK_END"
echo "  Initial Capital: \$$QUICK_CAPITAL"
echo "  Transaction Fees: $QUICK_FEES"
echo
echo "Report includes:"
echo "  - Executive summary with performance rating"
echo "  - 50+ performance metrics (Sharpe, Sortino, Calmar, etc.)"
echo "  - Benchmark comparison (vs SPY)"
echo "  - Risk analysis (VaR, CVaR, drawdown)"
echo "  - Monthly/yearly returns heatmaps"
echo "  - Interactive HTML tearsheet"
echo
echo "Estimated time: 1-2 minutes"
echo

python src/backtest_runner.py \
  --strategy MovingAverageCrossover \
  --symbols "$QUICK_SYMBOL" \
  --start "$QUICK_START" \
  --end "$QUICK_END" \
  --capital "$QUICK_CAPITAL" \
  --fees "$QUICK_FEES" \
  $QUICK_VISUALIZE \
  $QUICK_QUANTSTATS

echo
echo "================================================================================"
echo "                           QUICK TEST COMPLETE!"
echo "================================================================================"
echo

if [[ "$QUICK_QUANTSTATS" == "--quantstats" ]]; then
    echo "NEXT STEPS:"
    echo "  1. Check the console output above for the report location"
    echo "  2. Open the tearsheet.html file in your web browser"
    echo "  3. Review the executive summary and performance metrics"
    echo
    echo "The report includes:"
    echo "  - tearsheet.html (main report)"
    echo "  - quantstats_metrics.txt (text summary)"
    echo "  - daily_returns.csv (returns data)"
    echo "  - equity_curve.csv (portfolio value)"
else
    echo "Backtest completed! No report generated (--no-report flag used)."
fi
echo
