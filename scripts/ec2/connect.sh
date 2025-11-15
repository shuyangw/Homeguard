#!/bin/bash
# Quick SSH to Homeguard Trading Bot EC2 Instance
# Linux/Mac Shell Script

echo "========================================"
echo "Connecting to Homeguard Trading Bot"
echo "Instance: 100.30.95.146"
echo "========================================"
echo ""

ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146
