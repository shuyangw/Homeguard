#!/bin/bash
# Clear all Python cache files from the project

echo "Clearing Python cache files..."

# Navigate to project root
cd "$(dirname "$0")"

# Count before
CACHE_DIRS=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
PYC_FILES=$(find . -name "*.pyc" 2>/dev/null | wc -l)

echo "Found $CACHE_DIRS __pycache__ directories"
echo "Found $PYC_FILES .pyc files"

# Remove cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "✓ Cache cleared!"
echo ""
echo "Next steps:"
echo "1. Restart your GUI application completely"
echo "2. Run a portfolio backtest"
echo "3. Check for new JSON and HTML files in the output directory"
