#!/bin/bash
# Show status of all strategy services with colored output
#
# Usage:
#   ./strategy_status.sh
#   strat-status  (alias)
#   bot-status    (alias)

# First show all service statuses (without logs)
echo "=== OMR ==="
if systemctl cat homeguard-omr >/dev/null 2>&1; then
    SYSTEMD_COLORS=1 sudo -E systemctl status homeguard-omr --no-pager -n 0 2>/dev/null
else
    echo "OMR: service not installed"
fi
echo ""

echo "=== MP ==="
if systemctl cat homeguard-mp >/dev/null 2>&1; then
    SYSTEMD_COLORS=1 sudo -E systemctl status homeguard-mp --no-pager -n 0 2>/dev/null
else
    echo "MP: service not installed"
fi
echo ""

# Then show recent logs from all running services
echo "=== Recent Logs ==="
sudo journalctl -u homeguard-omr -u homeguard-mp -n 15 --no-pager --output=short 2>/dev/null || echo "No logs available"
