#!/bin/bash
# Install IB Gateway stack on EC2 (ARM64).
# Idempotent: safe to run multiple times.
# Usage: bash infra/ec2/install_ibkr_gateway.sh

set -e

REPO_DIR="/home/ec2-user/Homeguard"
IBC_VERSION="3.23.0"
JDK_VERSION="17.0.14+10"

echo "[1/6] Installing system dependencies..."
sudo yum install -y xorg-x11-server-Xvfb gettext libXi libXtst alsa-lib

echo "[2/6] Installing Bellsoft Liberica JDK 17 (aarch64 Full)..."
if ! java -version 2>&1 | grep -q "17"; then
    cd /tmp
    wget -q "https://download.bell-sw.com/java/${JDK_VERSION}/bellsoft-jdk${JDK_VERSION}-linux-aarch64-full.rpm"
    sudo rpm -ivh "bellsoft-jdk${JDK_VERSION}-linux-aarch64-full.rpm" || true
fi

echo "[3/6] Installing IB Gateway (stable)..."
if [ ! -d /home/ec2-user/ibgateway ]; then
    # Installer bundles x64 JRE; on ARM64 we patch it to use the system JDK
    mkdir -p /home/ec2-user/tmp
    cd /home/ec2-user/tmp
    wget -q https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
    chmod +x ibgateway-stable-standalone-linux-x64.sh
    cp ibgateway-stable-standalone-linux-x64.sh ibgateway-patched.sh
    JAVA_HOME_PATH="/usr/lib/jvm/bellsoft-java17-full.aarch64"
    sed -i "s|app_java_home=\`pwd\`\$|app_java_home=${JAVA_HOME_PATH}|" ibgateway-patched.sh
    sed -i 's|app_java_home=$app_java_home/jre|# patched: using system JDK|' ibgateway-patched.sh
    export TMPDIR=/home/ec2-user/tmp
    sudo -u ec2-user bash ibgateway-patched.sh -q -dir /home/ec2-user/ibgateway
fi

echo "[4/6] Installing IBC ${IBC_VERSION}..."
if [ ! -d /opt/ibc ] || ! grep -q "${IBC_VERSION}" /opt/ibc/version 2>/dev/null; then
    cd /tmp
    wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
    sudo mkdir -p /opt/ibc
    sudo unzip -o "IBCLinux-${IBC_VERSION}.zip" -d /opt/ibc
    sudo chmod +x /opt/ibc/scripts/*.sh
    echo "${IBC_VERSION}" | sudo tee /opt/ibc/version > /dev/null
fi

echo "[5/6] Installing systemd services..."
sudo cp "${REPO_DIR}/infra/ec2/services/homeguard-xvfb.service" /etc/systemd/system/
sudo cp "${REPO_DIR}/infra/ec2/services/homeguard-gateway.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable homeguard-xvfb homeguard-gateway

echo "[6/6] Adding 2FA reminder to crontab..."
# IB forces weekly restart on Sunday. Remind to approve 2FA.
(crontab -l 2>/dev/null | grep -v "ibkr-2fa-reminder"; echo "0 8 * * 0 echo 'IB Gateway needs 2FA approval after weekly restart' | logger -t ibkr-2fa") | crontab -

echo ""
echo "[+] IB Gateway stack installed (IBC ${IBC_VERSION})."
echo ""
echo "    Configure credentials in .env:"
echo "      IBKR_USERNAME, IBKR_PASSWORD, IBKR_TRADING_MODE, IBKR_GATEWAY_PORT"
echo ""
echo "    Start:"
echo "      sudo systemctl start homeguard-xvfb"
echo "      sudo systemctl start homeguard-gateway"
echo ""
echo "    Verify:"
echo "      journalctl -u homeguard-gateway -f"
echo ""
echo "    First start requires 2FA approval on IB Key mobile app."
