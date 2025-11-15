#!/bin/bash
# Restart Homeguard Trading Bot
# Linux/Mac Shell Script

echo "========================================"
echo "Restarting Homeguard Trading Bot"
echo "========================================"
echo ""

ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo systemctl restart homeguard-trading && echo 'Bot restarted successfully' && sleep 3 && sudo systemctl status homeguard-trading --no-pager"
