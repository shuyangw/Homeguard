#!/bin/bash
# View Homeguard Trading Bot Live Logs
# Linux/Mac Shell Script

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

echo "========================================"
echo "Viewing Live Trading Bot Logs"
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP" "sudo journalctl -u homeguard-trading -f"
