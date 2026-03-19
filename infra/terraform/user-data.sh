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
echo "[1/10] Updating system packages..."
yum update -y

# ===== INSTALL DEPENDENCIES =====
echo "[2/10] Installing dependencies..."
yum install -y \
    python3.11 \
    python3.11-pip \
    git \
    htop \
    tmux

# Verify Python installation
python3.11 --version

# ===== CREATE DIRECTORIES =====
echo "[3/10] Creating directories..."
sudo -u ec2-user mkdir -p /home/ec2-user/logs
sudo -u ec2-user mkdir -p /home/ec2-user/stock_data

# ===== CLONE REPOSITORY =====
echo "[4/10] Cloning Homeguard repository..."
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
echo "[5/10] Creating Python virtual environment..."
sudo -u ec2-user python3.11 -m venv /home/ec2-user/Homeguard/venv

# ===== INSTALL PYTHON DEPENDENCIES =====
echo "[6/10] Installing Python packages..."
sudo -u ec2-user /home/ec2-user/Homeguard/venv/bin/pip install --upgrade pip
sudo -u ec2-user /home/ec2-user/Homeguard/venv/bin/pip install -r /home/ec2-user/Homeguard/requirements.txt

# Install Discord bot dependencies (optional addon)
sudo -u ec2-user /home/ec2-user/Homeguard/venv/bin/pip install discord.py anthropic

# ===== CREATE .ENV FILE =====
echo "[7/10] Creating .env file with credentials..."
cat > /home/ec2-user/Homeguard/.env <<EOF
# Trading API Credentials
ALPACA_PAPER_KEY_ID=${alpaca_key_id}
ALPACA_PAPER_SECRET_KEY=${alpaca_secret}

# Discord Bot (Optional - leave empty if not using)
DISCORD_TOKEN=${discord_token}
ANTHROPIC_API_KEY=${anthropic_api_key}
ALLOWED_CHANNELS=${discord_allowed_channels}
EOF

# Set proper permissions
chown ec2-user:ec2-user /home/ec2-user/Homeguard/.env
chmod 600 /home/ec2-user/Homeguard/.env

# ===== CREATE DISCORD BOT LOG DIRECTORY =====
echo "[8/10] Creating Discord bot log directory..."
sudo -u ec2-user mkdir -p /home/ec2-user/logs/discord_bot

# ===== INSTALL SYSTEMD SERVICES =====
echo "[9/10] Installing systemd services..."

# Copy trading service file
cp /home/ec2-user/Homeguard/scripts/systemd/homeguard-trading.service /etc/systemd/system/

# Copy Discord bot service file (optional addon)
cp /home/ec2-user/Homeguard/scripts/systemd/homeguard-discord.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable trading service (auto-start on boot)
systemctl enable homeguard-trading

# Enable Discord bot service (auto-start on boot) - only starts if tokens are configured
systemctl enable homeguard-discord

# NOTE: Don't start the services automatically yet
# User should verify configuration first and start manually

echo "[10/10] Verifying installation..."
/home/ec2-user/Homeguard/venv/bin/python --version
ls -la /home/ec2-user/Homeguard/.env
systemctl is-enabled homeguard-trading
systemctl is-enabled homeguard-discord

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Services installed but NOT started."
echo "SSH to the instance and verify configuration before starting:"
echo ""
echo "TRADING BOT:"
echo "  1. Check .env file has correct Alpaca keys"
echo "  2. Verify config files exist:"
echo "     - config/trading/broker_alpaca.yaml"
echo "     - config/trading/omr_trading_config.yaml"
echo "  3. Start the service:"
echo "     sudo systemctl start homeguard-trading"
echo "  4. Check status:"
echo "     sudo systemctl status homeguard-trading"
echo "  5. View logs:"
echo "     tail -f ~/logs/trading_\$(date +%%Y%%m%%d).log"
echo ""
echo "DISCORD BOT (Optional):"
echo "  1. Add Discord tokens to .env if not already configured:"
echo "     DISCORD_TOKEN=your_bot_token"
echo "     ANTHROPIC_API_KEY=your_api_key"
echo "     ALLOWED_CHANNELS=channel_id"
echo "  2. Start the service:"
echo "     sudo systemctl start homeguard-discord"
echo "  3. Check status:"
echo "     sudo systemctl status homeguard-discord"
echo "  4. View logs:"
echo "     tail -f ~/logs/discord_bot/discord_bot_\$(date +%%Y%%m%%d).log"
echo ""
echo "=========================================="
echo "Installation completed at: $(date)"
echo "=========================================="
