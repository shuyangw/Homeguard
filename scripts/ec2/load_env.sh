#!/bin/bash
# ============================================================================
# Load environment variables from .env file
#
# This helper script parses the root .env file and exports environment variables
# for use in shell scripts. Source this at the start of EC2 scripts.
#
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"
# ============================================================================

# Find project root (two levels up from scripts/ec2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: .env file not found at $PROJECT_ROOT/.env"
    echo ""
    echo "Please create .env from the template:"
    echo "  cp $PROJECT_ROOT/.env.example $PROJECT_ROOT/.env"
    echo ""
    echo "Then edit .env with your EC2 instance details:"
    echo "  EC2_IP=your_instance_ip"
    echo "  EC2_INSTANCE_ID=your_instance_id"
    echo "  EC2_REGION=us-east-1"
    echo "  EC2_SSH_KEY_PATH=~/.ssh/homeguard-trading.pem"
    echo "  EC2_USER=ec2-user"
    return 1 2>/dev/null || exit 1
fi

# Parse .env file and export variables
# Handles KEY="value" and KEY=value formats, skips comments and empty lines
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue

    # Remove leading/trailing whitespace from key
    key=$(echo "$key" | xargs)

    # Remove quotes from value if present
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

    # Export the variable
    export "$key=$value"
done < "$PROJECT_ROOT/.env"

# Validate required EC2 variables
if [ -z "$EC2_IP" ]; then
    echo "ERROR: EC2_IP not set in .env file"
    return 1 2>/dev/null || exit 1
fi
if [ "$EC2_IP" = "<YOUR_EC2_IP>" ]; then
    echo "ERROR: EC2_IP is still set to placeholder value"
    echo "Please edit .env and set your actual EC2 IP address"
    return 1 2>/dev/null || exit 1
fi

if [ -z "$EC2_INSTANCE_ID" ]; then
    echo "ERROR: EC2_INSTANCE_ID not set in .env file"
    return 1 2>/dev/null || exit 1
fi
if [ "$EC2_INSTANCE_ID" = "<YOUR_INSTANCE_ID>" ]; then
    echo "ERROR: EC2_INSTANCE_ID is still set to placeholder value"
    echo "Please edit .env and set your actual EC2 instance ID"
    return 1 2>/dev/null || exit 1
fi

# Set defaults if not specified
: "${EC2_REGION:=us-east-1}"
: "${EC2_USER:=ec2-user}"
: "${EC2_SSH_KEY_PATH:=$HOME/.ssh/homeguard-trading.pem}"

# Expand ~ to $HOME in SSH key path
EC2_SSH_KEY_PATH="${EC2_SSH_KEY_PATH/#\~/$HOME}"

# Export all EC2 variables
export EC2_IP EC2_INSTANCE_ID EC2_REGION EC2_USER EC2_SSH_KEY_PATH
