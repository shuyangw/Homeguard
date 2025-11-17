#!/bin/bash
# Complete EC2 Instance Setup for Homeguard Trading Bot
# This script finishes the installation after user-data failed

set -e  # Exit on error

echo "===================================="
echo "Homeguard Trading Bot - Setup Script"
echo "===================================="

# Update requirements.txt with compatible numpy version
echo "[1/7] Fixing numpy dependency conflict..."
cd ~/Homeguard
sed -i 's/numpy==2.3.2/numpy==2.1.3/' requirements.txt
echo "✓ requirements.txt updated"

# Create virtual environment if it doesn't exist
echo "[2/7] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "[3/7] Installing Python dependencies (this may take 5-10 minutes)..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create logs directory
echo "[4/7] Creating logs directory..."
mkdir -p ~/logs
echo "✓ Logs directory created"

# Create systemd service
echo "[5/7] Creating systemd service..."
sudo tee /etc/systemd/system/homeguard-trading.service > /dev/null <<'EOF'
[Unit]
Description=Homeguard Trading Bot - Live Paper Trading
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/Homeguard
Environment="PATH=/home/ec2-user/Homeguard/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ec2-user/Homeguard/venv/bin/python scripts/trading/run_live_paper_trading.py --strategy omr
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=homeguard-trading

# Resource limits
MemoryMax=1G
CPUQuota=150%

[Install]
WantedBy=multi-user.target
EOF

sudo chmod 644 /etc/systemd/system/homeguard-trading.service
echo "✓ Systemd service created"

# Enable and start service
echo "[6/7] Starting trading bot service..."
sudo systemctl daemon-reload
sudo systemctl enable homeguard-trading
sudo systemctl start homeguard-trading
echo "✓ Service started"

# Setup .bashrc with bot aliases and banner
echo "[7/7] Setting up .bashrc with bot commands..."
cd ~/Homeguard
./scripts/ec2/instance_setup_bashrc.sh
echo "✓ Shell configured"

echo ""
echo "===================================="
echo "Setup Complete!"
echo "===================================="
echo ""
echo "Service Status:"
sudo systemctl status homeguard-trading --no-pager
echo ""
echo "Helpful aliases (use after reconnecting):"
echo "  bot-update   → Pull code + restart"
echo "  bot-status   → Check bot status"
echo "  bot-logs     → View live logs (with colors!)"
echo ""
echo "To activate aliases now:"
echo "  source ~/.bashrc"
echo ""
