#!/bin/bash
# Run All Backtests (Basic + Intermediate + Advanced + Optimization)
# Executes all available backtest scenarios sequentially
#
# Usage: ./RUN_ALL_STRATEGIES.sh [OPTIONS]
#
# Arguments (all optional):
#   --fees VALUE        Transaction fees as decimal (e.g., 0.001 = 0.1%)
#   --capital VALUE     Initial capital amount (e.g., 100000)
#   --start DATE        Start date in YYYY-MM-DD format
#   --end DATE          End date in YYYY-MM-DD format
#   --quantstats        Enable QuantStats reporting with tearsheet
#   --visualize         Enable static chart visualization
#   --verbosity LEVEL   Logging verbosity: 0=minimal, 1=normal, 2=detailed, 3=verbose
#   --basic-only        Run only basic strategies (5 tests)
#   --inter-only        Run only intermediate strategies (6 tests)
#   --adv-only          Run only advanced strategies (6 tests)
#   --opt-only          Run only optimization strategies (5 tests)
#
# Examples:
#   ./RUN_ALL_STRATEGIES.sh --fees 0.002 --capital 50000 --quantstats
#   ./RUN_ALL_STRATEGIES.sh --basic-only --quantstats --verbosity 2
#   ./RUN_ALL_STRATEGIES.sh --start 2023-01-01 --end 2024-12-31 --quantstats

# ============================================================================
# DEFAULT PARAMETERS - Can be overridden by command-line arguments
# ============================================================================

# Initial capital for all backtests (default: 100000)
BACKTEST_CAPITAL=100000

# Transaction fees as decimal (default: 0 = no fees)
BACKTEST_FEES=0

# Start date for all backtests (YYYY-MM-DD format)
BACKTEST_START="2023-01-01"

# End date for all backtests (YYYY-MM-DD format)
BACKTEST_END="2024-01-01"

# QuantStats reporting enabled
BACKTEST_QUANTSTATS="--quantstats"

# Visualization enabled by default
BACKTEST_VISUALIZE="--visualize"

# Logging verbosity (0-3, default: 1)
BACKTEST_VERBOSITY=1

# Test categories to run (default: all)
RUN_BASIC=1
RUN_INTERMEDIATE=1
RUN_ADVANCED=1
RUN_OPTIMIZATION=1

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --visualize)
            BACKTEST_VISUALIZE="--visualize"
            shift
            ;;
        --verbosity)
            BACKTEST_VERBOSITY="$2"
            shift 2
            ;;
        --basic-only)
            RUN_BASIC=1
            RUN_INTERMEDIATE=0
            RUN_ADVANCED=0
            RUN_OPTIMIZATION=0
            shift
            ;;
        --inter-only)
            RUN_BASIC=0
            RUN_INTERMEDIATE=1
            RUN_ADVANCED=0
            RUN_OPTIMIZATION=0
            shift
            ;;
        --adv-only)
            RUN_BASIC=0
            RUN_INTERMEDIATE=0
            RUN_ADVANCED=1
            RUN_OPTIMIZATION=0
            shift
            ;;
        --opt-only)
            RUN_BASIC=0
            RUN_INTERMEDIATE=0
            RUN_ADVANCED=0
            RUN_OPTIMIZATION=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export parameters for child scripts
export BACKTEST_CAPITAL
export BACKTEST_FEES
export BACKTEST_START
export BACKTEST_END
export BACKTEST_QUANTSTATS
export BACKTEST_VISUALIZE
export BACKTEST_VERBOSITY

# Count total tests to run
TOTAL_TESTS=0
[[ $RUN_BASIC -eq 1 ]] && ((TOTAL_TESTS+=5))
[[ $RUN_INTERMEDIATE -eq 1 ]] && ((TOTAL_TESTS+=6))
[[ $RUN_ADVANCED -eq 1 ]] && ((TOTAL_TESTS+=6))
[[ $RUN_OPTIMIZATION -eq 1 ]] && ((TOTAL_TESTS+=5))

echo "================================================================================"
echo "                        RUN ALL BACKTEST STRATEGIES"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Initial Capital: \$$BACKTEST_CAPITAL"
echo "  Transaction Fees: $BACKTEST_FEES"
echo "  Period: $BACKTEST_START to $BACKTEST_END"
echo "  QuantStats: $BACKTEST_QUANTSTATS"
echo "  Verbosity: $BACKTEST_VERBOSITY"
echo ""
echo "Test Categories:"
[[ $RUN_BASIC -eq 1 ]] && echo "  [X] Basic Strategies (5 tests)"
[[ $RUN_INTERMEDIATE -eq 1 ]] && echo "  [X] Intermediate Strategies (6 tests)"
[[ $RUN_ADVANCED -eq 1 ]] && echo "  [X] Advanced Strategies (6 tests)"
[[ $RUN_OPTIMIZATION -eq 1 ]] && echo "  [X] Optimization Strategies (5 tests)"
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo "Estimated Time: 15-30 minutes (depending on categories selected)"
echo ""
echo "Starting automated run..."
echo ""

TEST_COUNTER=0

# ============================================================================
# BASIC STRATEGIES
# ============================================================================

if [[ $RUN_BASIC -eq 1 ]]; then
    echo ""
    echo "================================================================================"
    echo "                        BASIC STRATEGIES (5 tests)"
    echo "================================================================================"
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running MA Crossover (AAPL)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running RSI Mean Reversion (AAPL)..."
    python ../src/backtest_runner.py --strategy RSIMeanReversion --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Bollinger Bands (MSFT)..."
    python ../src/backtest_runner.py --strategy MeanReversion --symbols MSFT \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "window=20,num_std=2.0,exit_threshold=0.5" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running MACD Momentum (GOOGL)..."
    python ../src/backtest_runner.py --strategy MomentumStrategy --symbols GOOGL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Breakout Strategy (AMZN)..."
    python ../src/backtest_runner.py --strategy BreakoutStrategy --symbols AMZN \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""
fi

# ============================================================================
# INTERMEDIATE STRATEGIES
# ============================================================================

if [[ $RUN_INTERMEDIATE -eq 1 ]]; then
    echo ""
    echo "================================================================================"
    echo "                     INTERMEDIATE STRATEGIES (6 tests)"
    echo "================================================================================"
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Multi-Symbol MA Crossover (AAPL,MSFT,GOOGL)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL,MSFT,GOOGL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Triple MA Crossover (TSLA)..."
    python ../src/backtest_runner.py --strategy TripleMovingAverage --symbols TSLA \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Aggressive RSI Mean Reversion (NVDA)..."
    python ../src/backtest_runner.py --strategy RSIMeanReversion --symbols NVDA \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "period=10,oversold=25,overbought=75" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Conservative Bollinger Bands (META)..."
    python ../src/backtest_runner.py --strategy MeanReversion --symbols META \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "window=30,num_std=2.5,exit_threshold=0.5" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Fast MACD Momentum (NFLX)..."
    python ../src/backtest_runner.py --strategy MomentumStrategy --symbols NFLX \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "fast_period=8,slow_period=17,signal_period=9" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Long-Period Breakout (AMD)..."
    python ../src/backtest_runner.py --strategy BreakoutStrategy --symbols AMD \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "window=30" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""
fi

# ============================================================================
# ADVANCED STRATEGIES
# ============================================================================

if [[ $RUN_ADVANCED -eq 1 ]]; then
    echo ""
    echo "================================================================================"
    echo "                       ADVANCED STRATEGIES (6 tests)"
    echo "================================================================================"
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Portfolio Strategy (5 tech stocks)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover \
        --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Custom Fast-Slow MA (AAPL)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "fast_window=10,slow_window=30,ma_type=ema" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Triple MA with EMA (MSFT)..."
    python ../src/backtest_runner.py --strategy TripleMovingAverage --symbols MSFT \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "fast_window=5,medium_window=20,slow_window=50,ma_type=ema" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Custom RSI (oversold=25, overbought=75)..."
    python ../src/backtest_runner.py --strategy RSIMeanReversion --symbols NVDA \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "period=14,oversold=25,overbought=75" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Wide Bollinger Bands (std=3.0)..."
    python ../src/backtest_runner.py --strategy MeanReversion --symbols GOOGL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "window=20,num_std=3.0,exit_threshold=0.5" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running High-Frequency Breakout (window=10)..."
    python ../src/backtest_runner.py --strategy BreakoutStrategy --symbols TSLA \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --params "window=10" \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""
fi

# ============================================================================
# OPTIMIZATION STRATEGIES
# ============================================================================

if [[ $RUN_OPTIMIZATION -eq 1 ]]; then
    echo ""
    echo "================================================================================"
    echo "                     OPTIMIZATION STRATEGIES (5 tests)"
    echo "================================================================================"
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Higher Fees Test (0.002 = 0.2%)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees 0.002 --capital $BACKTEST_CAPITAL \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Lower Capital Test (\$50,000)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital 50000 \
        --start $BACKTEST_START --end $BACKTEST_END \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Extended Period Test (2020-2024)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start 2020-01-01 --end 2024-01-01 \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Bull Market Test (2020-2021)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start 2020-01-01 --end 2021-12-31 \
        --verbosity $BACKTEST_VERBOSITY
    echo ""

    ((TEST_COUNTER++))
    echo "[$TEST_COUNTER/$TOTAL_TESTS] Running Bear Market Test (2022)..."
    python ../src/backtest_runner.py --strategy MovingAverageCrossover --symbols AAPL \
        $BACKTEST_QUANTSTATS $BACKTEST_VISUALIZE \
        --fees $BACKTEST_FEES --capital $BACKTEST_CAPITAL \
        --start 2022-01-01 --end 2022-12-31 \
        --verbosity $BACKTEST_VERBOSITY
    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "                        ALL BACKTESTS COMPLETED!"
echo "================================================================================"
echo ""
echo "Total Tests Run: $TOTAL_TESTS"
echo ""
echo "Reports saved to: logs/"
echo ""
echo "View tearsheets: Open logs/*/tearsheet.html in your browser"
echo ""
