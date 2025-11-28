#!/bin/bash
# Restart Homeguard Discord Bot
# Linux/Mac Shell Script

echo "========================================"
echo "Restarting Homeguard Discord Bot"
echo "========================================"
echo

ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo systemctl restart homeguard-discord"

echo
echo "Waiting 5 seconds for service to start..."
sleep 5

echo
echo "========================================"
echo "Current Status:"
echo "========================================"
echo

ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo systemctl status homeguard-discord --no-pager"
