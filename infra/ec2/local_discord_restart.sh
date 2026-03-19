#!/bin/bash
# Restart Homeguard Discord Bot
# Linux/Mac Shell Script

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

echo "========================================"
echo "Restarting Homeguard Discord Bot"
echo "========================================"
echo

ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP" "sudo systemctl restart homeguard-discord"

echo
echo "Waiting 5 seconds for service to start..."
sleep 5

echo
echo "========================================"
echo "Current Status:"
echo "========================================"
echo

ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP" "sudo systemctl status homeguard-discord --no-pager"
