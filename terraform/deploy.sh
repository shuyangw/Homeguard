#!/bin/bash
# Quick deployment script for Homeguard Trading Bot

set -e

echo "=========================================="
echo "Homeguard Trading Bot - Terraform Deploy"
echo "=========================================="
echo ""

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "ERROR: Terraform is not installed"
    echo "Install from: https://www.terraform.io/downloads"
    exit 1
fi

echo "✓ Terraform installed: $(terraform version | head -n 1)"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS CLI not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "✓ AWS credentials configured"
echo "  Account: $(aws sts get-caller-identity --query Account --output text)"
echo "  Region: $(aws configure get region)"
echo ""

# Check if terraform.tfvars exists
if [ ! -f terraform.tfvars ]; then
    echo "ERROR: terraform.tfvars not found"
    echo ""
    echo "Create it from the example:"
    echo "  cp terraform.tfvars.example terraform.tfvars"
    echo "  nano terraform.tfvars"
    echo ""
    echo "Then set required variables:"
    echo "  - key_pair_name"
    echo "  - ssh_allowed_cidrs"
    echo "  - alpaca_key_id (or use TF_VAR_alpaca_key_id env var)"
    echo "  - alpaca_secret (or use TF_VAR_alpaca_secret env var)"
    exit 1
fi

echo "✓ terraform.tfvars found"
echo ""

# Check if Alpaca credentials are set
if [ -z "$TF_VAR_alpaca_key_id" ]; then
    echo "WARNING: TF_VAR_alpaca_key_id not set as environment variable"
    echo "Make sure alpaca_key_id is set in terraform.tfvars"
fi

if [ -z "$TF_VAR_alpaca_secret" ]; then
    echo "WARNING: TF_VAR_alpaca_secret not set as environment variable"
    echo "Make sure alpaca_secret is set in terraform.tfvars"
fi

echo ""
echo "=========================================="
echo "Starting Deployment"
echo "=========================================="
echo ""

# Initialize Terraform
echo "[1/3] Initializing Terraform..."
terraform init

echo ""
echo "[2/3] Planning deployment..."
terraform plan -out=tfplan

echo ""
echo "=========================================="
echo "Deployment Plan Summary"
echo "=========================================="
echo ""
echo "Review the plan above. This will create:"
echo "  - EC2 instance (t4g.small)"
echo "  - Security group"
echo "  - EBS volume (8 GB)"
echo ""
echo "Estimated monthly cost: ~\$13"
echo ""
read -p "Continue with deployment? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Deployment cancelled"
    rm -f tfplan
    exit 0
fi

echo ""
echo "[3/3] Applying changes..."
terraform apply tfplan

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""

# Show outputs
terraform output post_deployment_instructions

echo ""
echo "Quick Commands:"
echo "==============="
echo ""
echo "Get SSH command:"
echo "  terraform output ssh_connection_command"
echo ""
echo "Get instance IP:"
echo "  terraform output instance_public_ip"
echo ""
echo "View all outputs:"
echo "  terraform output"
echo ""
echo "Destroy infrastructure:"
echo "  terraform destroy"
echo ""

# Clean up plan file
rm -f tfplan
