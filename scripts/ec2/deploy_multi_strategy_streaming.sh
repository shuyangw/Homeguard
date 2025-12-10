#!/bin/bash
# Deploy multi-strategy runner with shared WebSocket connection
#
# This script:
# 1. Pulls latest code from GitHub
# 2. Stops existing separate OMR and RAMP services
# 3. Installs new multi-strategy service
# 4. Enables streaming in .env
# 5. Starts multi-strategy service with shared WebSocket
# 6. Verifies both strategies are running correctly

set -e  # Exit on error

echo "=================================================================="
echo "DEPLOYING MULTI-STRATEGY WITH SHARED STREAMING"
echo "=================================================================="
echo ""
echo "This will:"
echo "  - Stop separate homeguard-omr and homeguard-ramp services"
echo "  - Install homeguard-multi service (OMR + RAMP in one process)"
echo "  - Share single IEX WebSocket connection"
echo "  - Enable smart fallback for incomplete buffer"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Deployment cancelled"
    exit 1
fi

# Navigate to project directory
cd /home/ec2-user/Homeguard

echo ""
echo "Step 1: Pulling latest code from GitHub..."
git fetch origin
git pull origin main
echo "✓ Code updated"
echo ""

echo "Step 2: Enabling streaming in .env..."
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
echo "  USE_STREAMING=$new_value"
echo "✓ Streaming enabled"
echo ""

echo "Step 3: Checking existing services..."
echo ""
if systemctl is-active --quiet homeguard-omr 2>/dev/null; then
    echo "  ✓ homeguard-omr is running (will be stopped)"
else
    echo "  - homeguard-omr is not running"
fi

if systemctl is-active --quiet homeguard-ramp 2>/dev/null; then
    echo "  ✓ homeguard-ramp is running (will be stopped)"
else
    echo "  - homeguard-ramp is not running"
fi

if systemctl is-active --quiet homeguard-multi 2>/dev/null; then
    echo "  ! homeguard-multi is already running (will be restarted)"
else
    echo "  - homeguard-multi is not running (will be created)"
fi
echo ""

echo "Step 4: Stopping old services..."
# Stop and disable old separate services
if systemctl list-unit-files | grep -q "homeguard-omr.service"; then
    echo "  Stopping homeguard-omr..."
    sudo systemctl stop homeguard-omr || true
    sudo systemctl disable homeguard-omr || true
    echo "  ✓ homeguard-omr stopped and disabled"
fi

if systemctl list-unit-files | grep -q "homeguard-ramp.service"; then
    echo "  Stopping homeguard-ramp..."
    sudo systemctl stop homeguard-ramp || true
    sudo systemctl disable homeguard-ramp || true
    echo "  ✓ homeguard-ramp stopped and disabled"
fi
echo ""

echo "Step 5: Installing multi-strategy service..."
# Copy service file to systemd directory
sudo cp scripts/ec2/homeguard-multi.service /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/homeguard-multi.service

# Reload systemd
sudo systemctl daemon-reload
echo "  ✓ Service file installed"
echo ""

echo "Step 6: Starting multi-strategy service..."
sudo systemctl enable homeguard-multi
sudo systemctl start homeguard-multi
sleep 3
echo "  ✓ Service started"
echo ""

echo "Step 7: Verifying service is running..."
if systemctl is-active --quiet homeguard-multi; then
    echo "  ✓ homeguard-multi is active"
else
    echo "  ✗ homeguard-multi failed to start"
    echo ""
    echo "Checking logs for errors:"
    journalctl -u homeguard-multi -n 50 --no-pager
    exit 1
fi
echo ""

echo "Step 8: Checking initialization logs..."
echo ""
echo "Recent logs (last 40 lines):"
echo "============================"
journalctl -u homeguard-multi -n 40 --no-pager
echo ""

echo "Step 9: Verifying streaming is active..."
echo ""
sleep 2  # Give it a moment to initialize
echo "Streaming-related logs:"
echo "======================="
journalctl -u homeguard-multi -n 100 --no-pager | grep -E "STREAMING|WebSocket|LiveDataProvider|symbols" | tail -20 || echo "  (Streaming initialization in progress...)"
echo ""

echo "=================================================================="
echo "DEPLOYMENT COMPLETE"
echo "=================================================================="
echo ""
echo "Multi-strategy service is now running:"
echo "  - OMR + RAMP in single process"
echo "  - Shared IEX WebSocket connection"
echo "  - Smart fallback enabled (90% threshold)"
echo ""
echo "Service: homeguard-multi"
echo "  OMR Entry:  3:50 PM ET"
echo "  RAMP:       3:55 PM ET"
echo "  OMR Exit:   9:31 AM ET (next day)"
echo ""
echo "To monitor logs:"
echo "  journalctl -u homeguard-multi -f"
echo ""
echo "To check streaming status:"
echo "  journalctl -u homeguard-multi | grep -i streaming"
echo ""
echo "To check smart fallback activations:"
echo "  journalctl -u homeguard-multi | grep 'Falling back to REST API'"
echo ""
echo "To check service status:"
echo "  sudo systemctl status homeguard-multi"
echo ""
echo "To restart service:"
echo "  sudo systemctl restart homeguard-multi"
echo ""
echo "Old services (disabled):"
echo "  homeguard-omr:  stopped and disabled"
echo "  homeguard-ramp: stopped and disabled"
echo ""
echo "If you need to rollback:"
echo "  1. sudo systemctl stop homeguard-multi"
echo "  2. sudo systemctl disable homeguard-multi"
echo "  3. sudo systemctl enable homeguard-omr"
echo "  4. sudo systemctl enable homeguard-ramp"
echo "  5. sudo systemctl start homeguard-omr"
echo "  6. sudo systemctl start homeguard-ramp"
echo ""
