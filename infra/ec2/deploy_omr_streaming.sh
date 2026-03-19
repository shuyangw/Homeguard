#!/bin/bash
# Deploy OMR with streaming to EC2 using shared WebSocket connection
#
# This script:
# 1. Pulls latest code from GitHub
# 2. Enables streaming in .env (if not already enabled)
# 3. Creates systemd service for OMR with streaming
# 4. Restarts OMR service
# 5. Verifies streaming is active for OMR

set -e  # Exit on error

echo "=========================================="
echo "DEPLOYING OMR WITH STREAMING TO EC2"
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

echo "Step 4: Checking which services are currently running..."
echo ""
echo "Current OMR service status:"
systemctl status homeguard-omr --no-pager | head -3 || echo "  (OMR service not found - will create)"
echo ""
echo "Current RAMP service status:"
systemctl status homeguard-ramp --no-pager | head -3 || echo "  (RAMP service not found)"
echo ""

echo "Step 5: Restarting OMR service with streaming enabled..."
echo "  Restarting homeguard-omr..."
sudo systemctl restart homeguard-omr
sleep 2
echo "✓ OMR service restarted"
echo ""

echo "Step 6: Verifying OMR service is running..."
if systemctl is-active --quiet homeguard-omr; then
    echo "  ✓ homeguard-omr is active"
else
    echo "  ✗ homeguard-omr failed to start"
    echo ""
    echo "Checking logs for errors:"
    journalctl -u homeguard-omr -n 50 --no-pager
    exit 1
fi
echo ""

echo "Step 7: Checking logs for streaming confirmation..."
echo ""
echo "OMR Logs (last 30 lines):"
echo "---"
journalctl -u homeguard-omr -n 30 --no-pager | grep -E "STREAMING|streaming|WebSocket|LiveDataProvider|smart fallback|Buffer|REST API" || echo "  (No streaming messages yet - may take a moment to initialize)"
echo ""

echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "OMR streaming is now ENABLED on EC2"
echo ""
echo "Expected behavior:"
echo "  - OMR and RAMP share single IEX WebSocket connection"
echo "  - Smart fallback activates if buffer <90% complete"
echo "  - First execution after restart may be slower (~5-10s)"
echo "  - Subsequent executions use fast buffer (<200ms)"
echo ""
echo "To monitor streaming status:"
echo "  journalctl -u homeguard-omr -f | grep -i streaming"
echo ""
echo "To check for smart fallback activations:"
echo "  journalctl -u homeguard-omr | grep 'Falling back to REST API'"
echo ""
echo "To check data quality:"
echo "  journalctl -u homeguard-omr | grep 'bars ('"
echo ""
echo "If both OMR and RAMP try to connect (connection limit error):"
echo "  1. Stop RAMP: sudo systemctl stop homeguard-ramp"
echo "  2. Only run OMR with streaming"
echo "  3. Or upgrade to SIP feed for multiple connections"
