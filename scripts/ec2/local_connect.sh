#!/bin/bash
# Quick SSH to Homeguard Trading Bot EC2 Instance
# Linux/Mac Shell Script

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

echo "========================================"
echo "Connecting to Homeguard Trading Bot"
echo "Instance: $EC2_IP"
echo "========================================"
echo ""

ssh -i "$EC2_SSH_KEY_PATH" "$EC2_USER@$EC2_IP"
