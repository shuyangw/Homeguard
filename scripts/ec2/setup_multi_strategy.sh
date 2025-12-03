#!/bin/bash
#
# Setup Multi-Strategy Services
#
# This script installs the per-strategy systemd services.
# Run this once to migrate from single homeguard-trading.service
# to separate homeguard-omr.service and homeguard-mp.service
#
# Usage:
#   sudo ./setup_multi_strategy.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$SCRIPT_DIR/services"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Homeguard Multi-Strategy Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run with sudo${NC}"
    exit 1
fi

# Check if service files exist
if [ ! -f "$SERVICE_DIR/homeguard-omr.service" ]; then
    echo -e "${RED}Service files not found in $SERVICE_DIR${NC}"
    exit 1
fi

# Stop old service if running
echo -e "${YELLOW}Stopping old homeguard-trading.service...${NC}"
systemctl stop homeguard-trading.service 2>/dev/null || true
systemctl disable homeguard-trading.service 2>/dev/null || true

# Install setproctitle in venv
echo -e "${YELLOW}Installing setproctitle...${NC}"
/home/ec2-user/Homeguard/venv/bin/pip install setproctitle==1.3.3

# Copy service files
echo -e "${YELLOW}Installing service files...${NC}"
cp "$SERVICE_DIR/homeguard-trading.target" /etc/systemd/system/
cp "$SERVICE_DIR/homeguard-omr.service" /etc/systemd/system/
cp "$SERVICE_DIR/homeguard-mp.service" /etc/systemd/system/

# Set permissions
chmod 644 /etc/systemd/system/homeguard-trading.target
chmod 644 /etc/systemd/system/homeguard-omr.service
chmod 644 /etc/systemd/system/homeguard-mp.service

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
systemctl daemon-reload

# Enable target
systemctl enable homeguard-trading.target

# Enable OMR by default (since it was running before)
echo -e "${YELLOW}Enabling OMR service...${NC}"
systemctl enable homeguard-omr.service
systemctl start homeguard-omr.service

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Services installed:"
echo -e "  ${GREEN}homeguard-omr.service${NC}  - Overnight Mean Reversion"
echo -e "  ${YELLOW}homeguard-mp.service${NC}   - Momentum Protection (disabled)"
echo ""
echo -e "Commands:"
echo -e "  ${BLUE}sudo systemctl start homeguard-omr${NC}   - Start OMR"
echo -e "  ${BLUE}sudo systemctl stop homeguard-omr${NC}    - Stop OMR"
echo -e "  ${BLUE}sudo systemctl start homeguard-mp${NC}    - Start MP"
echo -e "  ${BLUE}sudo systemctl stop homeguard-mp${NC}     - Stop MP"
echo ""
echo -e "View logs:"
echo -e "  ${BLUE}journalctl -u homeguard-omr -f${NC}"
echo -e "  ${BLUE}journalctl -u homeguard-mp -f${NC}"
echo ""
echo -e "Process names will appear as:"
echo -e "  ${GREEN}homeguard-omr${NC} and ${GREEN}homeguard-mp${NC} in ps/htop"
