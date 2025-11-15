#!/bin/bash
# Daily Health Check for Homeguard Trading Bot
# Quick automated health verification

echo "========================================"
echo "Homeguard Trading Bot Health Check"
echo "Date: $(date)"
echo "========================================"
echo ""

# 1. Instance State
echo "[1/6] Instance State:"
INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids i-02500fe2392631ff2 --query 'Reservations[0].Instances[0].State.Name' --output text 2>&1)
if [ "$INSTANCE_STATE" == "running" ]; then
    echo "✓ Instance is RUNNING"
elif [ "$INSTANCE_STATE" == "stopped" ]; then
    echo "⊙ Instance is STOPPED (expected outside market hours)"
else
    echo "⚠ Instance state: $INSTANCE_STATE"
fi
echo ""

# 2. Bot Service Status (only if instance is running)
if [ "$INSTANCE_STATE" == "running" ]; then
    echo "[2/6] Bot Service Status:"
    BOT_STATUS=$(ssh -i ~/.ssh/homeguard-trading.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@100.30.95.146 "sudo systemctl is-active homeguard-trading" 2>&1)
    if [ "$BOT_STATUS" == "active" ]; then
        echo "✓ Bot service is ACTIVE"
    else
        echo "✗ Bot service is $BOT_STATUS"
    fi
    echo ""

    # 3. Recent Errors
    echo "[3/6] Recent Errors (last hour):"
    ERROR_COUNT=$(ssh -i ~/.ssh/homeguard-trading.pem -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -p err --since '1 hour ago' --no-pager 2>&1 | wc -l")
    if [ "$ERROR_COUNT" -eq "0" ]; then
        echo "✓ No errors in last hour"
    else
        echo "⚠ $ERROR_COUNT error lines found"
        echo "  View with: scripts/ec2/view_logs.sh"
    fi
    echo ""

    # 4. Memory Usage
    echo "[4/6] Resource Usage:"
    ssh -i ~/.ssh/homeguard-trading.pem -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo systemctl status homeguard-trading --no-pager | grep -E 'Memory|CPU'" 2>&1 | sed 's/^/  /'
    echo ""

    # 5. Last Activity
    echo "[5/6] Last Activity:"
    ssh -i ~/.ssh/homeguard-trading.pem -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -n 3 --no-pager 2>&1" | tail -3 | sed 's/^/  /'
    echo ""

    # 6. Market Status
    echo "[6/6] Current Market Status:"
    MARKET_STATUS=$(ssh -i ~/.ssh/homeguard-trading.pem -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -n 1 --no-pager 2>&1 | grep -oP 'Market: \K[A-Z]+' | tail -1")
    if [ -n "$MARKET_STATUS" ]; then
        echo "  Market: $MARKET_STATUS"
    else
        echo "  (Unable to determine market status)"
    fi
    echo ""
else
    echo "[2-6] Skipped (instance not running)"
    echo ""
fi

echo "========================================"
echo "Health Check Complete"
echo "========================================"
echo ""
echo "For live monitoring, run:"
echo "  scripts/ec2/view_logs.sh"
