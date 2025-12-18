#!/bin/bash
# Deploy streaming infrastructure to EC2 and enable it
#
# This script:
# 1. Pulls latest code from GitHub
# 2. Enables streaming in .env
# 3. Restarts both OMR and RAMP services
# 4. Verifies streaming is active

set -e  # Exit on error

echo "=========================================="
echo "DEPLOYING STREAMING TO EC2"
echo "=========================================="
echo ""

# Navigate to project directory
cd /home/ec2-user/Homeguard

echo "Step 1: Pulling latest code from GitHub..."
git fetch origin
git pull origin main
echo "✓ Code updated"
echo ""

echo "Step 2: Checking current .env configuration..."
if grep -q "USE_STREAMING" .env; then
    current_value=$(grep "USE_STREAMING" .env | cut -d '=' -f2 | tr -d '"')
    echo "  Current USE_STREAMING=$current_value"
else
    echo "  USE_STREAMING not found in .env"
fi
echo ""

echo "Step 3: Enabling streaming in .env..."
# Check if USE_STREAMING exists
if grep -q "^USE_STREAMING=" .env; then
    # Update existing value
    sed -i 's/^USE_STREAMING=.*/USE_STREAMING="true"/' .env
    echo "  Updated existing USE_STREAMING to true"
else
    # Add new entry
    echo "" >> .env
    echo "# Streaming configuration" >> .env
    echo "USE_STREAMING=\"true\"" >> .env
    echo "STREAMING_FEED=\"iex\"" >> .env
    echo "  Added USE_STREAMING=true to .env"
fi

# Verify
new_value=$(grep "USE_STREAMING" .env | cut -d '=' -f2 | tr -d '"')
echo "  New USE_STREAMING=$new_value"
echo "✓ Streaming enabled"
echo ""

echo "Step 4: Checking systemd services status..."
systemctl status homeguard-omr --no-pager | head -3
systemctl status homeguard-ramp --no-pager | head -3
echo ""

echo "Step 5: Restarting services..."
echo "  Restarting homeguard-omr..."
sudo systemctl restart homeguard-omr
sleep 2

echo "  Restarting homeguard-ramp..."
sudo systemctl restart homeguard-ramp
sleep 2
echo "✓ Services restarted"
echo ""

echo "Step 6: Verifying services are running..."
if systemctl is-active --quiet homeguard-omr; then
    echo "  ✓ homeguard-omr is active"
else
    echo "  ✗ homeguard-omr failed to start"
    exit 1
fi

if systemctl is-active --quiet homeguard-ramp; then
    echo "  ✓ homeguard-ramp is active"
else
    echo "  ✗ homeguard-ramp failed to start"
    exit 1
fi
echo ""

echo "Step 7: Checking logs for streaming confirmation..."
echo ""
echo "OMR Logs (last 20 lines):"
echo "---"
journalctl -u homeguard-omr -n 20 --no-pager | grep -E "STREAMING|streaming|WebSocket|LiveDataProvider" || echo "  (No streaming messages yet - may take a moment to initialize)"
echo ""

echo "RAMP Logs (last 20 lines):"
echo "---"
journalctl -u homeguard-ramp -n 20 --no-pager | grep -E "STREAMING|streaming|WebSocket|LiveDataProvider" || echo "  (No streaming messages yet - may take a moment to initialize)"
echo ""

echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Streaming is now ENABLED on EC2"
echo ""
echo "To verify streaming is working:"
echo "  journalctl -u homeguard-omr -f | grep -i streaming"
echo "  journalctl -u homeguard-ramp -f | grep -i streaming"
echo ""
echo "To check full logs:"
echo "  journalctl -u homeguard-omr -n 100"
echo "  journalctl -u homeguard-ramp -n 100"
echo ""
echo "To disable streaming:"
echo "  sed -i 's/USE_STREAMING=\"true\"/USE_STREAMING=\"false\"/' /home/ec2-user/Homeguard/.env"
echo "  sudo systemctl restart homeguard-omr homeguard-ramp"
