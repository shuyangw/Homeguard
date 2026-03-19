#!/bin/bash
# View Homeguard Discord Bot Logs
# Linux/Mac Shell Script

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

echo "========================================"
echo "Homeguard Discord Bot Logs (Live Stream)"
echo "========================================"
echo "Press Ctrl+C to stop streaming"
echo

ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP" "sudo journalctl -u homeguard-discord -f"
