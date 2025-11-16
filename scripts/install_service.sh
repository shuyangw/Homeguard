#!/bin/bash
# Install Homeguard Trading Bot as systemd service
# Run this script after you've set up your .env and confirmed the bot works

set -e  # Exit on error

echo "========================================"
echo "Homeguard Trading Bot - Service Installer"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run this script as root (don't use sudo)"
    echo "The script will ask for sudo password when needed"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "ERROR: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please create it first:"
    echo "  python3.11 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "WARNING: .env file not found"
    echo "Make sure to create it with your Alpaca API keys before starting the service"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create logs directory
mkdir -p ~/logs
echo "✓ Created log directory: ~/logs"

# Copy service file to systemd
echo ""
echo "Installing systemd service..."
sudo cp "$SCRIPT_DIR/systemd/homeguard-trading.service" /etc/systemd/system/
echo "✓ Service file copied to /etc/systemd/system/"

# Reload systemd
sudo systemctl daemon-reload
echo "✓ Systemd reloaded"

# Enable service (auto-start on boot)
sudo systemctl enable homeguard-trading
echo "✓ Service enabled (will auto-start on boot)"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Service Commands:"
echo "  Start:   sudo systemctl start homeguard-trading"
echo "  Stop:    sudo systemctl stop homeguard-trading"
echo "  Status:  sudo systemctl status homeguard-trading"
echo "  Logs:    sudo journalctl -u homeguard-trading -f"
echo ""
echo "File Logs:"
echo "  Main:    tail -f ~/logs/trading_\$(date +%Y%m%d).log"
echo "  Trades:  tail -f ~/logs/executions_\$(date +%Y%m%d).log"
echo ""
echo "The service is NOT started yet. Start it manually:"
echo "  sudo systemctl start homeguard-trading"
echo ""
