#!/bin/bash
#
# View Homeguard Trading Bot Live Logs (Plain Text - No Colors)
# Shell Script for Linux/macOS
#
# This script views the trading bot logs with ANSI color codes stripped
# for better compatibility with terminals that don't support colors well.
#
# Usage:
#   ./scripts/ec2/local_view_logs_plain.sh
#

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

echo "========================================"
echo "Viewing Live Trading Bot Logs (Plain)"
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

# Strip ANSI color codes using sed on remote server
ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP" "sudo journalctl -u homeguard-trading -f --output=cat | sed 's/\x1b\[[0-9;]*m//g'"
