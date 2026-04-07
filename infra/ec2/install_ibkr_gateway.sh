#!/bin/bash
# Install IB Gateway stack on EC2 (ARM64).
# Idempotent: safe to run multiple times.
# Usage: bash infra/ec2/install_ibkr_gateway.sh

set -e

echo "[1/5] Installing Xvfb..."
sudo yum install -y xorg-x11-server-Xvfb

echo "[2/5] Installing Bellsoft Liberica JDK 17 (aarch64 Full)..."
cd /tmp
if ! java -version 2>&1 | grep -q "17"; then
    wget -q https://download.bell-sw.com/java/17.0.14+10/bellsoft-jdk17.0.14+10-linux-aarch64-full.rpm
    sudo rpm -ivh bellsoft-jdk17.0.14+10-linux-aarch64-full.rpm || true
fi

echo "[3/5] Installing IB Gateway (stable)..."
cd /tmp
if [ ! -d /home/ec2-user/ibgateway ]; then
    wget -q https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
    chmod +x ibgateway-stable-standalone-linux-x64.sh
    sudo -u ec2-user bash ibgateway-stable-standalone-linux-x64.sh -q -dir /home/ec2-user/ibgateway
fi

echo "[4/5] Installing IBC..."
IBC_VERSION="3.19.0"
cd /tmp
if [ ! -d /opt/ibc ]; then
    wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
    sudo mkdir -p /opt/ibc
    sudo unzip -o "IBCLinux-${IBC_VERSION}.zip" -d /opt/ibc
    sudo chmod +x /opt/ibc/scripts/*.sh
fi

echo "[5/5] Installing systemd service..."
sudo cp /home/ec2-user/Homeguard/infra/ec2/services/homeguard-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable homeguard-gateway

echo ""
echo "[+] IB Gateway stack installed."
echo "    Configure credentials in .env:"
echo "      IBKR_USERNAME, IBKR_PASSWORD, IBKR_TRADING_MODE"
echo "    Then: sudo systemctl start homeguard-gateway"
echo "    Verify: journalctl -u homeguard-gateway -f"
