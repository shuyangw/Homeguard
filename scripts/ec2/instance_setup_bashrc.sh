#!/bin/bash
#
# Setup .bashrc for Homeguard Trading Bot
# Run this script ON the EC2 instance to configure shell aliases and banner
#
# Usage:
#   ./instance_setup_bashrc.sh
#

set -e

echo "=========================================="
echo "Setting up .bashrc for Homeguard Bot"
echo "=========================================="
echo ""

# Check if already set up
if grep -q "# Homeguard Trading Bot Shortcuts" ~/.bashrc 2>/dev/null; then
    echo "⚠️  .bashrc already configured"
    echo "   Remove the existing Homeguard section first if you want to update"
    exit 0
fi

# Add Homeguard bot configuration
cat >> ~/.bashrc << 'BASHRC_SETUP'

# Homeguard Trading Bot Shortcuts
alias bot-update='echo "🔄 Updating..."; (cd ~/Homeguard && git pull && echo "🔄 Restarting (~15s)..." && sudo systemctl restart homeguard-trading && echo "✅ Done!")'
alias bot-restart='echo "🔄 Restarting..."; sudo systemctl restart homeguard-trading && echo "✅ Restarted!"'
alias bot-start='sudo systemctl start homeguard-trading'
alias bot-stop='sudo systemctl stop homeguard-trading'
alias bot-status='SYSTEMD_COLORS=1 sudo -E systemctl status homeguard-trading --no-pager'
alias bot-logs='sudo journalctl -u homeguard-trading -f --output=cat --no-hostname'
alias bot-logs-recent='sudo journalctl -u homeguard-trading -n 100 --no-pager --output=cat --no-hostname'

# Welcome Banner (with colors!)
if [ -n "$PS1" ]; then
  echo ""
  echo -e "\033[1;36m==========================================\033[0m"
  echo -e "\033[1;36m  Homeguard Trading Bot\033[0m"
  echo -e "\033[1;36m==========================================\033[0m"
  if systemctl is-active --quiet homeguard-trading 2>/dev/null; then
    echo -e "  Status: \033[1;32m● RUNNING\033[0m"
  else
    echo -e "  Status: \033[1;31m● STOPPED\033[0m"
  fi
  echo ""
  echo -e "\033[1;33m  Quick Commands:\033[0m"
  echo -e "  \033[1;32mbot-update\033[0m   → Pull code + restart"
  echo -e "  \033[1;32mbot-restart\033[0m  → Restart bot"
  echo -e "  \033[1;32mbot-start\033[0m    → Start bot"
  echo -e "  \033[1;32mbot-stop\033[0m     → Stop bot"
  echo -e "  \033[1;32mbot-status\033[0m   → Check status"
  echo -e "  \033[1;32mbot-logs\033[0m     → Live logs"
  echo ""
  echo -e "\033[1;36m==========================================\033[0m"
  echo ""
fi
BASHRC_SETUP

echo "✅ .bashrc configured successfully!"
echo ""
echo "Aliases added:"
echo "  - bot-update   (pull code + restart)"
echo "  - bot-restart  (restart bot)"
echo "  - bot-start    (start bot)"
echo "  - bot-stop     (stop bot)"
echo "  - bot-status   (check status)"
echo "  - bot-logs     (live logs with colors)"
echo ""
echo "To activate, run: source ~/.bashrc"
echo "Or reconnect: exit and SSH again"
echo ""
echo "=========================================="
