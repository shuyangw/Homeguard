#!/bin/bash
#
# Setup .bashrc for Homeguard Trading Bot
# Run this script ON the EC2 instance to configure shell aliases and banner
#
# Usage:
#   ./instance_setup_bashrc.sh
#   ./instance_setup_bashrc.sh --force  # Overwrite existing config
#

set -e

FORCE=false
if [[ "$1" == "--force" ]]; then
    FORCE=true
fi

echo "=========================================="
echo "Setting up .bashrc for Homeguard Bot"
echo "=========================================="
echo ""

# Check if already set up
if grep -q "# Homeguard Trading Bot Shortcuts" ~/.bashrc 2>/dev/null; then
    if [[ "$FORCE" == false ]]; then
        echo "[!] .bashrc already configured"
        echo "    Use --force to overwrite"
        exit 0
    else
        echo "[*] Removing existing Homeguard config..."
        # Remove existing config block
        sed -i '/# Homeguard Trading Bot Shortcuts/,/# End Homeguard Config/d' ~/.bashrc
    fi
fi

# Add Homeguard bot configuration
cat >> ~/.bashrc << 'BASHRC_SETUP'

# Homeguard Trading Bot Shortcuts
# ================================

# Quick Commands (all strategies)
alias bot-update='echo "[*] Updating..."; (cd ~/Homeguard && git pull && echo "[*] Restarting services..." && sudo systemctl restart homeguard-omr homeguard-mp homeguard-cscm-demo 2>/dev/null; echo "[+] Done!")'
alias bot-restart='echo "[*] Restarting..."; sudo systemctl restart homeguard-omr homeguard-mp homeguard-cscm-demo 2>/dev/null; echo "[+] Restarted!"'
alias bot-start='sudo systemctl start homeguard-omr homeguard-mp homeguard-cscm-demo 2>/dev/null'
alias bot-stop='sudo systemctl stop homeguard-omr homeguard-mp homeguard-cscm-demo 2>/dev/null'
alias bot-status='echo ""; echo "Strategy Services:"; SYSTEMD_COLORS=1 systemctl is-active homeguard-omr homeguard-mp homeguard-cscm-demo 2>/dev/null | paste - - - <(echo -e "homeguard-omr\nhomeguard-mp\nhomeguard-cscm-demo") | awk "{print \"  \" \$2 \": \" \$1}"'
alias bot-logs='sudo journalctl -u homeguard-omr -u homeguard-mp -u homeguard-cscm-demo -f --output=cat --no-hostname'
alias bot-logs-recent='sudo journalctl -u homeguard-omr -u homeguard-mp -u homeguard-cscm-demo -n 100 --no-pager --output=cat --no-hostname'

# OMR (Overnight Mean Reversion) - Leveraged ETFs
alias omr-start='sudo systemctl start homeguard-omr && echo "[+] OMR started"'
alias omr-stop='sudo systemctl stop homeguard-omr && echo "[-] OMR stopped"'
alias omr-restart='sudo systemctl restart homeguard-omr && echo "[*] OMR restarted"'
alias omr-status='SYSTEMD_COLORS=1 sudo -E systemctl status homeguard-omr --no-pager'
alias omr-logs='sudo journalctl -u homeguard-omr -f --output=cat --no-hostname'

# MP/RAMP (Momentum Protection) - S&P 500
alias mp-start='sudo systemctl start homeguard-mp && echo "[+] MP started"'
alias mp-stop='sudo systemctl stop homeguard-mp && echo "[-] MP stopped"'
alias mp-restart='sudo systemctl restart homeguard-mp && echo "[*] MP restarted"'
alias mp-status='SYSTEMD_COLORS=1 sudo -E systemctl status homeguard-mp --no-pager'
alias mp-logs='sudo journalctl -u homeguard-mp -f --output=cat --no-hostname'

# CSCM Demo (Cross-Sectional Crypto Momentum) - Paper Trading
alias cscm-start='sudo systemctl start homeguard-cscm-demo && echo "[+] CSCM Demo started"'
alias cscm-stop='sudo systemctl stop homeguard-cscm-demo && echo "[-] CSCM Demo stopped"'
alias cscm-restart='sudo systemctl restart homeguard-cscm-demo && echo "[*] CSCM Demo restarted"'
alias cscm-status='~/Homeguard/scripts/ec2/cscm_demo_status.sh'
alias cscm-logs='sudo journalctl -u homeguard-cscm-demo -f --output=cat --no-hostname'
alias cscm-positions='~/Homeguard/scripts/ec2/cscm_demo_positions.sh'
alias cscm-reset='~/Homeguard/scripts/ec2/cscm_demo_reset.sh'
alias cscm-refresh='~/Homeguard/scripts/ec2/cscm_demo_refresh.sh'

# Config Commands
alias strat-status='~/Homeguard/scripts/ec2/strategy_status.sh'
alias strat-omr-on='~/Homeguard/scripts/ec2/toggle_strategy.sh omr on'
alias strat-omr-off='~/Homeguard/scripts/ec2/toggle_strategy.sh omr off'
alias strat-mp-on='~/Homeguard/scripts/ec2/toggle_strategy.sh mp on'
alias strat-mp-off='~/Homeguard/scripts/ec2/toggle_strategy.sh mp off'

# Welcome Banner (with colors!)
homeguard_banner() {
    # Colors
    local CYAN='\033[1;36m'
    local GREEN='\033[1;32m'
    local RED='\033[1;31m'
    local YELLOW='\033[1;33m'
    local GRAY='\033[0;37m'
    local NC='\033[0m'

    echo ""
    echo -e "${CYAN}===========================================${NC}"
    echo -e "${CYAN}  Homeguard Trading Bot${NC}"
    echo -e "${CYAN}===========================================${NC}"
    echo ""

    # Strategy Processes
    echo -e " ${YELLOW}Strategy Processes:${NC}"
    for svc in homeguard-omr homeguard-mp homeguard-cscm-demo; do
        if systemctl is-active --quiet $svc 2>/dev/null; then
            echo -e "  ${GREEN}o${NC} $svc\t${GREEN}running${NC}"
        elif systemctl is-enabled --quiet $svc 2>/dev/null; then
            echo -e "  ${GRAY}o${NC} $svc\t${GRAY}enabled${NC}"
        else
            echo -e "  ${GRAY}o${NC} $svc\t${GRAY}disabled${NC}"
        fi
    done
    echo ""

    # Strategy Config (if toggle script exists)
    if [ -f ~/Homeguard/config/trading/strategy_toggle.yaml ]; then
        echo -e " ${YELLOW}Strategy Config:${NC}"
        if grep -q "enabled: true" ~/Homeguard/config/trading/strategy_toggle.yaml 2>/dev/null | head -1; then
            local omr_status=$(grep -A1 "omr:" ~/Homeguard/config/trading/strategy_toggle.yaml | grep "enabled:" | awk '{print $2}')
            local mp_status=$(grep -A1 "mp:" ~/Homeguard/config/trading/strategy_toggle.yaml | grep "enabled:" | awk '{print $2}')
            if [ "$omr_status" == "true" ]; then
                echo -e "  ${GREEN}[+] OMR${NC}    enabled in config"
            else
                echo -e "  ${RED}[-] OMR${NC}    disabled in config"
            fi
            if [ "$mp_status" == "true" ]; then
                echo -e "  ${GREEN}[+] MP${NC}     enabled in config"
            else
                echo -e "  ${RED}[-] MP${NC}     disabled in config"
            fi
        fi
        echo ""
    fi

    # Quick Commands
    echo -e " ${YELLOW}Quick Commands (all strategies):${NC}"
    echo -e "  ${GREEN}bot-update${NC}        -> Pull code + restart all"
    echo -e "  ${GREEN}bot-restart${NC}       -> Restart all strategies"
    echo -e "  ${GREEN}bot-start${NC}         -> Start all strategies"
    echo -e "  ${GREEN}bot-stop${NC}          -> Stop all strategies"
    echo -e "  ${GREEN}bot-status${NC}        -> Status of all strategies"
    echo -e "  ${GREEN}bot-logs${NC}          -> Combined live logs"
    echo -e "  ${GREEN}bot-logs-recent${NC}   -> Last 100 logs (all)"
    echo ""

    # Per-Strategy Commands
    echo -e " ${YELLOW}Per-Strategy Commands:${NC}"
    echo -e "  ${GREEN}omr-start/stop/restart/status/logs${NC}"
    echo -e "  ${GREEN}mp-start/stop/restart/status/logs${NC}"
    echo -e "  ${CYAN}cscm-start/stop/restart/status/logs${NC}"
    echo -e "  ${CYAN}cscm-positions${NC}    -> Show CSCM positions"
    echo -e "  ${CYAN}cscm-reset${NC}        -> Reset CSCM portfolio"
    echo -e "  ${CYAN}cscm-refresh${NC}      -> Force CSCM rebalance"
    echo ""

    # Config Commands
    echo -e " ${YELLOW}Config Commands:${NC}"
    echo -e "  ${GREEN}strat-status${NC}      -> Show detailed status"
    echo -e "  ${GREEN}strat-omr-on/off${NC}  -> Toggle OMR in config"
    echo -e "  ${GREEN}strat-mp-on/off${NC}   -> Toggle MP in config"
    echo ""
    echo -e "${CYAN}===========================================${NC}"
    echo ""
}

# Show banner on interactive login
if [ -n "$PS1" ]; then
    homeguard_banner
fi

# End Homeguard Config
BASHRC_SETUP

echo "[+] .bashrc configured successfully!"
echo ""
echo "Aliases added:"
echo "  - bot-* commands (all strategies)"
echo "  - omr-* commands (OMR strategy)"
echo "  - mp-* commands (MP/RAMP strategy)"
echo "  - cscm-* commands (CSCM demo)"
echo "  - strat-* commands (config toggles)"
echo ""
echo "To activate, run: source ~/.bashrc"
echo "Or reconnect: exit and SSH again"
echo ""
echo "=========================================="
