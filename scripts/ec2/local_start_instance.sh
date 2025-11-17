#!/bin/bash
#
# Start Homeguard Trading Bot EC2 Instance
#
# This script starts the EC2 instance from your local machine using AWS CLI.
# The instance will automatically start the trading bot via the systemd service.
#
# Prerequisites:
#   - AWS CLI installed and configured
#   - AWS credentials with EC2 permissions (ec2:StartInstances, ec2:DescribeInstances)
#
# Usage:
#   ./scripts/ec2/local_start_instance.sh
#

set -e

# Configuration
INSTANCE_ID="i-02500fe2392631ff2"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Start Homeguard Trading Bot EC2 Instance"
echo "=========================================="
echo ""
echo -e "${BLUE}Instance ID: $INSTANCE_ID${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ Error: AWS CLI is not installed${NC}"
    echo ""
    echo "Please install AWS CLI:"
    echo "  https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check current instance state
echo "Checking instance state..."
INSTANCE_STATE=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>&1)

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: Failed to check instance state${NC}"
    echo ""
    echo "Error details:"
    echo "$INSTANCE_STATE"
    echo ""
    echo "Please check:"
    echo "  - AWS CLI is configured (run: aws configure)"
    echo "  - Your AWS credentials have EC2 permissions"
    echo "  - Instance ID is correct: $INSTANCE_ID"
    exit 1
fi

echo -e "Current state: ${YELLOW}$INSTANCE_STATE${NC}"
echo ""

# Start instance if stopped
if [ "$INSTANCE_STATE" == "stopped" ]; then
    echo "Starting instance..."
    aws ec2 start-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION" \
        --output text > /dev/null

    echo -e "${GREEN}✅ Instance start command sent${NC}"
    echo ""
    echo "Waiting for instance to start (this may take 1-2 minutes)..."

    # Wait for instance to be running
    aws ec2 wait instance-running \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION"

    echo -e "${GREEN}✅ Instance is now running!${NC}"
    echo ""

    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo "Instance Details:"
    echo "  Public IP: $PUBLIC_IP"
    echo ""
    echo "Note: Wait ~30 seconds for the trading bot to start"
    echo ""
    echo "To connect:"
    echo -e "  ${BLUE}ssh -i ~/.ssh/homeguard-trading.pem ec2-user@$PUBLIC_IP${NC}"
    echo ""
    echo "To check bot status:"
    echo -e "  ${BLUE}./scripts/ec2/local_check_bot.sh${NC}"

elif [ "$INSTANCE_STATE" == "running" ]; then
    echo -e "${GREEN}✅ Instance is already running${NC}"
    echo ""

    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo "Instance Details:"
    echo "  Public IP: $PUBLIC_IP"
    echo ""
    echo "To connect:"
    echo -e "  ${BLUE}ssh -i ~/.ssh/homeguard-trading.pem ec2-user@$PUBLIC_IP${NC}"

elif [ "$INSTANCE_STATE" == "pending" ]; then
    echo -e "${YELLOW}⏳ Instance is starting...${NC}"
    echo ""
    echo "Waiting for instance to be running..."

    aws ec2 wait instance-running \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION"

    echo -e "${GREEN}✅ Instance is now running!${NC}"

else
    echo -e "${YELLOW}⚠️  Instance is in state: $INSTANCE_STATE${NC}"
    echo ""
    echo "Cannot start instance from this state."
    echo "Please check the AWS console for more details."
fi

echo ""
echo "=========================================="
