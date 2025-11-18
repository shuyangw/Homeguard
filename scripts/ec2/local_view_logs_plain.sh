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

echo "========================================"
echo "Viewing Live Trading Bot Logs (Plain)"
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

# Strip ANSI color codes using sed on remote server
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -f --output=cat | sed 's/\x1b\[[0-9;]*m//g'"
