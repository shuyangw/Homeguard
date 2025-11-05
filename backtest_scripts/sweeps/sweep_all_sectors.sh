#!/bin/bash
# Sweep All Sector Universes with One Strategy
# Tests a strategy across all sector universes: SEMICONDUCTORS, ENERGY, FINANCE, HEALTHCARE, CONSUMER

if [ -z "$1" ]; then
    echo "Error: Strategy name required"
    echo ""
    echo "Usage: ./sweep_all_sectors.sh STRATEGY [START_DATE] [END_DATE]"
    echo ""
    echo "Example: ./sweep_all_sectors.sh MovingAverageCrossover 2023-01-01 2024-01-01"
    exit 1
fi

STRATEGY=$1

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
BACKTEST_FEES=${BACKTEST_FEES:-0}
BACKTEST_PARALLEL=${BACKTEST_PARALLEL:---parallel}
BACKTEST_MAX_WORKERS=${BACKTEST_MAX_WORKERS:-4}
BACKTEST_VERBOSITY=${BACKTEST_VERBOSITY:-0}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "========================================"
echo "Sector-Wide Sweep"
echo "========================================"
echo "Strategy: $STRATEGY"
echo "Period: $START_DATE to $END_DATE"
if [ "$BACKTEST_PARALLEL" = "--parallel" ]; then
    echo "Mode: Parallel ($BACKTEST_MAX_WORKERS workers)"
else
    echo "Mode: Sequential"
fi
echo ""
echo "Testing across 6 sector universes:"
echo "  1. SEMICONDUCTORS (10 stocks)"
echo "  2. ENERGY (10 stocks)"
echo "  3. FINANCE (10 stocks)"
echo "  4. HEALTHCARE (10 stocks)"
echo "  5. CONSUMER (10 stocks)"
echo "  6. TECH_GIANTS (10 stocks)"
echo "========================================"
echo ""

# Sector 1: Semiconductors
echo ""
echo "[1/6] Sweeping SEMICONDUCTORS sector..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe SEMICONDUCTORS \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sector_SEMICONDUCTORS_${STRATEGY}" \
  --verbosity "$BACKTEST_VERBOSITY"

# Sector 2: Energy
echo ""
echo "[2/6] Sweeping ENERGY sector..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe ENERGY \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sector_ENERGY_${STRATEGY}" \
  --verbosity "$BACKTEST_VERBOSITY"

# Sector 3: Finance
echo ""
echo "[3/6] Sweeping FINANCE sector..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe FINANCE \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sector_FINANCE_${STRATEGY}" \
  --verbosity "$BACKTEST_VERBOSITY"

# Sector 4: Healthcare
echo ""
echo "[4/6] Sweeping HEALTHCARE sector..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe HEALTHCARE \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sector_HEALTHCARE_${STRATEGY}" \
  --verbosity "$BACKTEST_VERBOSITY"

# Sector 5: Consumer
echo ""
echo "[5/6] Sweeping CONSUMER sector..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe CONSUMER \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sector_CONSUMER_${STRATEGY}" \
  --verbosity "$BACKTEST_VERBOSITY"

# Sector 6: Tech Giants
echo ""
echo "[6/6] Sweeping TECH_GIANTS sector..."
python "$SCRIPT_DIR/../../src/backtest_runner.py" \
  --strategy "$STRATEGY" \
  --universe TECH_GIANTS \
  --sweep \
  $BACKTEST_PARALLEL \
  --max-workers "$BACKTEST_MAX_WORKERS" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --capital "$BACKTEST_CAPITAL" \
  --fees "$BACKTEST_FEES" \
  --run-name "sector_TECH_GIANTS_${STRATEGY}" \
  --verbosity "$BACKTEST_VERBOSITY"

echo ""
echo "========================================"
echo "All sector sweeps complete!"
echo "========================================"
echo ""
echo "Results saved to logs with prefix: sector_*_${STRATEGY}"
echo ""
echo "Compare HTML reports to see which sector works best with $STRATEGY"
echo ""
