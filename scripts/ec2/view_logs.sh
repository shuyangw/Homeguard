#!/bin/bash
# View Homeguard Trading Bot Live Logs
# Linux/Mac Shell Script

echo "========================================"
echo "Viewing Live Trading Bot Logs"
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -f"
