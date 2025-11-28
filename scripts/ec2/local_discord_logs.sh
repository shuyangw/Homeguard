#!/bin/bash
# View Homeguard Discord Bot Logs
# Linux/Mac Shell Script

echo "========================================"
echo "Homeguard Discord Bot Logs (Live Stream)"
echo "========================================"
echo "Press Ctrl+C to stop streaming"
echo

ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo journalctl -u homeguard-discord -f"
