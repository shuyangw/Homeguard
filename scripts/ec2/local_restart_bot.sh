#!/bin/bash
# Restart Homeguard Trading Bot
# Linux/Mac Shell Script

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

echo "========================================"
echo "Restarting Homeguard Trading Bot"
echo "========================================"
echo ""

ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP" "sudo systemctl restart homeguard-trading && echo 'Bot restarted successfully' && sleep 3 && sudo systemctl status homeguard-trading --no-pager"
