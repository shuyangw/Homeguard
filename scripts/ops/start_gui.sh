#!/bin/bash
################################################################################
# Homeguard GUI Launcher (macOS/Linux)
################################################################################
# This script launches the Homeguard backtesting GUI application.
#
# Requirements:
#   - Anaconda/Miniconda installed
#   - 'fintech' conda environment configured
#
# Usage:
#   ./start_gui.sh
#
# Make executable:
#   chmod +x start_gui.sh
################################################################################

echo ""
echo "============================================================================"
echo " Homeguard Backtesting Framework - GUI Launcher"
echo "============================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo "[INFO] $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda not found in PATH"
    echo ""
    echo "Please ensure Anaconda or Miniconda is installed and added to PATH."
    echo ""
    echo "Common conda locations:"
    echo "  - ~/anaconda3/bin/conda"
    echo "  - ~/miniconda3/bin/conda"
    echo "  - /opt/anaconda3/bin/conda"
    echo ""
    echo "To add conda to PATH, add this to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"\$HOME/anaconda3/bin:\$PATH\""
    echo ""
    exit 1
fi

print_info "[1/3] Activating fintech conda environment..."

# Initialize conda for bash shell
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the fintech environment
conda activate fintech 2>/dev/null

if [ $? -ne 0 ]; then
    print_error "Failed to activate 'fintech' environment"
    echo ""
    echo "Please create the environment first:"
    echo "  conda create -n fintech python=3.13"
    echo "  conda activate fintech"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

print_success "Environment activated"

# Change to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "[2/3] Launching Homeguard GUI..."
echo "Current directory: $PWD"
echo ""

# Add src to Python path and launch GUI
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m gui

# Capture exit code
GUI_EXIT_CODE=$?

echo ""
print_info "[3/3] GUI closed with exit code: $GUI_EXIT_CODE"

# If there was an error, show message
if [ $GUI_EXIT_CODE -ne 0 ]; then
    print_error "GUI exited with an error"
    echo ""
    exit $GUI_EXIT_CODE
fi

print_success "GUI closed successfully"
exit 0
