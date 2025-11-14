#!/bin/bash
# User Data Script - Automated Installation for Homeguard Trading Bot
# This script runs on first boot of the EC2 instance

set -e  # Exit on error
set -x  # Print commands (for debugging in /var/log/cloud-init-output.log)

# Log file
LOG_FILE="/var/log/homeguard-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Homeguard Trading Bot - Installation Started"
echo "Time: $(date)"
echo "=========================================="

# ===== SYSTEM UPDATES =====
echo "[1/8] Updating system packages..."
yum update -y

# ===== INSTALL DEPENDENCIES =====
echo "[2/8] Installing dependencies..."
yum install -y \
    python3.11 \
    python3.11-pip \
    git \
    htop \
    tmux

# Verify Python installation
python3.11 --version

# ===== CREATE DIRECTORIES =====
echo "[3/8] Creating directories..."
sudo -u ec2-user mkdir -p /home/ec2-user/logs
sudo -u ec2-user mkdir -p /home/ec2-user/stock_data

# ===== CLONE REPOSITORY =====
echo "[4/8] Cloning Homeguard repository..."
cd /home/ec2-user

if [ ! -d "Homeguard" ]; then
    sudo -u ec2-user git clone ${git_repo_url} Homeguard
    cd Homeguard
    sudo -u ec2-user git checkout ${git_branch}
else
    echo "Repository already exists, pulling latest..."
    cd Homeguard
    sudo -u ec2-user git pull origin ${git_branch}
fi

# ===== CREATE VIRTUAL ENVIRONMENT =====
echo "[5/8] Creating Python virtual environment..."
sudo -u ec2-user python3.11 -m venv /home/ec2-user/Homeguard/venv

# ===== INSTALL PYTHON DEPENDENCIES =====
echo "[6/8] Installing Python packages..."
sudo -u ec2-user /home/ec2-user/Homeguard/venv/bin/pip install --upgrade pip
sudo -u ec2-user /home/ec2-user/Homeguard/venv/bin/pip install -r /home/ec2-user/Homeguard/requirements.txt

# ===== CREATE .ENV FILE =====
echo "[7/8] Creating .env file with Alpaca credentials..."
cat > /home/ec2-user/Homeguard/.env <<EOF
ALPACA_PAPER_KEY_ID=${alpaca_key_id}
ALPACA_PAPER_SECRET_KEY=${alpaca_secret}
EOF

# Set proper permissions
chown ec2-user:ec2-user /home/ec2-user/Homeguard/.env
chmod 600 /home/ec2-user/Homeguard/.env

# ===== INSTALL SYSTEMD SERVICE =====
echo "[8/8] Installing systemd service..."

# Copy service file
cp /home/ec2-user/Homeguard/scripts/systemd/homeguard-trading.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service (auto-start on boot)
systemctl enable homeguard-trading

# NOTE: Don't start the service automatically yet
# User should verify configuration first and start manually

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Service installed but NOT started."
echo "SSH to the instance and verify configuration before starting:"
echo ""
echo "  1. Check .env file has correct Alpaca keys"
echo "  2. Verify config files exist:"
echo "     - config/trading/broker_alpaca.yaml"
echo "     - config/trading/omr_trading_config.yaml"
echo ""
echo "  3. Start the service:"
echo "     sudo systemctl start homeguard-trading"
echo ""
echo "  4. Check status:"
echo "     sudo systemctl status homeguard-trading"
echo ""
echo "  5. View logs:"
echo "     tail -f ~/logs/trading_\$(date +%%Y%%m%%d).log"
echo ""
echo "=========================================="
echo "Installation completed at: $(date)"
echo "=========================================="
