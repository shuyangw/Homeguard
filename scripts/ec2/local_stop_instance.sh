#!/bin/bash
#
# Stop Homeguard Trading Bot EC2 Instance
#
# This script stops the EC2 instance from your local machine using AWS CLI.
# Use this to manually stop the instance during off-hours.
#
# Prerequisites:
#   - AWS CLI installed and configured
#   - AWS credentials with EC2 permissions (ec2:StopInstances, ec2:DescribeInstances)
#
# Usage:
#   ./scripts/ec2/local_stop_instance.sh
#

set -e

# Load EC2 configuration from .env
source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh" || exit 1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Stop Homeguard Trading Bot EC2 Instance"
echo "=========================================="
echo ""
echo -e "${BLUE}Instance ID: $EC2_INSTANCE_ID${NC}"
echo -e "${BLUE}Region: $EC2_REGION${NC}"
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
set +e  # Temporarily disable exit on error to capture AWS errors
INSTANCE_STATE=$(aws ec2 describe-instances \
    --instance-ids "$EC2_INSTANCE_ID" \
    --region "$EC2_REGION" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>&1)
AWS_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $AWS_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Error: Failed to check instance state${NC}"
    echo ""
    echo "Error details:"
    echo "$INSTANCE_STATE"
    echo ""
    echo "Please check:"
    echo "  - AWS CLI is configured (run: aws configure)"
    echo "  - Your AWS credentials have EC2 permissions"
    echo "  - Instance ID is correct: $EC2_INSTANCE_ID"
    exit 1
fi

echo -e "Current state: ${YELLOW}$INSTANCE_STATE${NC}"
echo ""

# Stop instance if running
if [ "$INSTANCE_STATE" == "running" ]; then
    echo -e "${YELLOW}⚠️  This will stop the trading bot and shut down the instance.${NC}"
    echo ""
    read -p "Are you sure you want to stop the instance? (y/N): " -n 1 -r
    echo ""
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping instance..."
        aws ec2 stop-instances \
            --instance-ids "$EC2_INSTANCE_ID" \
            --region "$EC2_REGION" \
            --output text > /dev/null

        echo -e "${GREEN}✅ Instance stop command sent${NC}"
        echo ""
        echo "Waiting for instance to stop (this may take 1-2 minutes)..."

        # Wait for instance to be stopped
        aws ec2 wait instance-stopped \
            --instance-ids "$EC2_INSTANCE_ID" \
            --region "$EC2_REGION"

        echo -e "${GREEN}✅ Instance is now stopped${NC}"
        echo ""
        echo "To start the instance again:"
        echo -e "  ${BLUE}./scripts/ec2/local_start_instance.sh${NC}"
    else
        echo "Operation cancelled."
    fi

elif [ "$INSTANCE_STATE" == "stopped" ]; then
    echo -e "${GREEN}✅ Instance is already stopped${NC}"
    echo ""
    echo "To start the instance:"
    echo -e "  ${BLUE}./scripts/ec2/local_start_instance.sh${NC}"

elif [ "$INSTANCE_STATE" == "stopping" ]; then
    echo -e "${YELLOW}⏳ Instance is already stopping...${NC}"
    echo ""
    echo "Waiting for instance to be stopped..."

    aws ec2 wait instance-stopped \
        --instance-ids "$EC2_INSTANCE_ID" \
        --region "$EC2_REGION"

    echo -e "${GREEN}✅ Instance is now stopped${NC}"

else
    echo -e "${YELLOW}⚠️  Instance is in state: $INSTANCE_STATE${NC}"
    echo ""
    echo "Cannot stop instance from this state."
    echo "Please check the AWS console for more details."
fi

echo ""
echo "=========================================="
