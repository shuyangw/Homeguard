#!/bin/bash
# Install and configure the CloudWatch Agent on Amazon Linux 2023 (ARM64).
#
# Prerequisites:
#   - IAM instance profile with CloudWatchAgentServerPolicy attached
#   - Run as root or with sudo
#
# Usage:
#   sudo bash infra/ec2/install_cloudwatch_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_SRC="${SCRIPT_DIR}/cloudwatch-agent-config.json"
CONFIG_DST="/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json"

echo "[1/4] Installing CloudWatch Agent..."
if ! command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
    sudo yum install -y amazon-cloudwatch-agent
else
    echo "  Already installed"
fi

echo "[2/4] Copying agent config..."
sudo cp "$CONFIG_SRC" "$CONFIG_DST"

echo "[3/4] Starting CloudWatch Agent..."
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c "file:${CONFIG_DST}"

echo "[4/4] Verifying agent status..."
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a status

echo ""
echo "[+] CloudWatch Agent installed and running."
echo "    Metrics will appear in CloudWatch console under namespace 'CWAgent'"
echo "    within 2-3 minutes."
