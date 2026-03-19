#!/bin/bash
# CSCM Demo Trading Logs
#
# Stream live logs from the CSCM demo trading service.
# Uses DemoBroker with Binance streaming (self-contained paper trading).
#
# Usage:
#   ./cscm_demo_logs.sh        # Stream live
#   ./cscm_demo_logs.sh -n 50  # Last 50 lines
#   cscm-demo-logs  (alias)

set -e

if [ "$1" == "-n" ] && [ -n "$2" ]; then
    # Show last N lines
    echo "CSCM Demo Logs (last $2 lines)"
    echo "=========================================="
    sudo journalctl -u homeguard-cscm-demo -n "$2" --no-pager --output=short
else
    # Stream live
    echo "CSCM Demo Live Logs (Ctrl+C to stop)"
    echo "=========================================="
    sudo journalctl -u homeguard-cscm-demo -f --output=short
fi
