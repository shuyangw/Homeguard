# Pre-Flight Checklist - Before Deploying

Use this checklist to ensure you're ready to deploy the Homeguard trading bot to AWS.

---

## Prerequisites

### ✅ 1. Terraform Installed

```bash
terraform --version
# Should show: Terraform v1.0.0 or higher
```

**If not installed**:
- macOS: `brew install terraform`
- Linux: Download from https://www.terraform.io/downloads

---

### ✅ 2. AWS CLI Configured

```bash
aws sts get-caller-identity
# Should show your AWS account details
```

**If not configured**:
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

---

### ✅ 3. EC2 Key Pair Created

```bash
# Check if key exists
aws ec2 describe-key-pairs --key-names homeguard-trading-bot
```

**If not exists**:
```bash
# Create key pair
aws ec2 create-key-pair \
  --key-name homeguard-trading-bot \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/homeguard-trading-bot.pem

# Set permissions
chmod 400 ~/.ssh/homeguard-trading-bot.pem
```

---

### ✅ 4. Get Your Public IP

```bash
curl https://checkip.amazonaws.com
# Note this IP - you'll use it in terraform.tfvars
```

---

### ✅ 5. Alpaca API Credentials

**Paper Trading Account**:
1. Sign up at https://alpaca.markets/
2. Go to Paper Trading dashboard
3. Generate API keys
4. Copy Key ID and Secret Key

**Verify keys work**:
```bash
curl -X GET https://paper-api.alpaca.markets/v2/account \
  -H "APCA-API-KEY-ID: YOUR_KEY_ID" \
  -H "APCA-API-SECRET-KEY: YOUR_SECRET"

# Should return account details (not an error)
```

---

## Configuration

### ✅ 6. Create terraform.tfvars

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars
```

**Minimal required settings**:
```hcl
key_pair_name     = "homeguard-trading-bot"
ssh_allowed_cidrs = ["YOUR_IP/32"]  # Replace with IP from step 4
```

**Set Alpaca credentials** (choose one method):

**Method A: Environment variables (recommended)**:
```bash
export TF_VAR_alpaca_key_id="your_key_here"
export TF_VAR_alpaca_secret="your_secret_here"
```

**Method B: In terraform.tfvars** (less secure):
```hcl
alpaca_key_id = "your_key_here"
alpaca_secret = "your_secret_here"
```

---

### ✅ 7. Verify Repository Access

```bash
# Test that repository is accessible
git ls-remote https://github.com/shuyangw/Homeguard.git
# Should list branches
```

---

## Cost Verification

### ✅ 8. Understand Costs

**Monthly costs** (default configuration):
- EC2 t4g.small: $12.26
- EBS 8 GB: $0.64
- Data transfer: ~$0.10
- **Total: ~$13/month**

**Optional additions**:
- Elastic IP: $0 (if instance running), $3.60/month (if stopped)
- CloudWatch Logs: +$0.50-1/month
- Detailed monitoring: +$2.10/month

---

## Security Verification

### ✅ 9. Security Best Practices

**Check your configuration**:

```bash
# Verify SSH is restricted to your IP
grep ssh_allowed_cidrs terraform.tfvars
# Should show YOUR_IP/32, NOT 0.0.0.0/0
```

**Verify volume protection**:
```bash
grep delete_volume_on_termination terraform.tfvars
# Should be: false (protects your data)
```

**Verify credentials are secure**:
```bash
# Check if using environment variables
echo $TF_VAR_alpaca_key_id
# If set, you're using the secure method

# Or check if .gitignore excludes tfvars
cat .gitignore | grep tfvars
# Should include: *.tfvars
```

---

## Final Checks

### ✅ 10. Test Terraform Configuration

```bash
cd terraform

# Initialize
terraform init

# Validate configuration
terraform validate
# Should show: "Success! The configuration is valid."

# Check what will be created
terraform plan
# Review the output - should show ~3-5 resources to create
```

---

## Deployment Readiness

### ✅ All Checks Passed?

If you've completed all steps above, you're ready to deploy!

**Deploy with**:
```bash
# Option 1: Interactive script
./deploy.sh

# Option 2: Manual
terraform apply
```

---

## Troubleshooting

### Common Issues

**Issue: "Error: No valid credential sources found"**
- Fix: Run `aws configure` and enter credentials

**Issue: "Error: InvalidKeyPair.NotFound"**
- Fix: Create key pair (see step 3)

**Issue: "Error: UnauthorizedOperation"**
- Fix: Check AWS credentials have EC2 permissions

**Issue: Terraform plan shows unexpected resources**
- Fix: Run `terraform init -upgrade`

**Issue: Can't find terraform.tfvars**
- Fix: Copy from example: `cp terraform.tfvars.example terraform.tfvars`

---

## Post-Deployment Checklist

After successful deployment:

### ✅ 1. Wait for Initialization (3-5 minutes)

```bash
# Get instance IP
terraform output instance_public_ip

# SSH to instance
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<INSTANCE_IP>

# Watch installation progress
tail -f /var/log/cloud-init-output.log
```

### ✅ 2. Verify Installation

```bash
# Check if service is installed
systemctl list-unit-files | grep homeguard
# Should show: homeguard-trading.service

# Check repository is cloned
ls -la ~/Homeguard
# Should show the codebase
```

### ✅ 3. Start Trading Bot

```bash
# Start service
sudo systemctl start homeguard-trading

# Check status
sudo systemctl status homeguard-trading
# Should show: "active (running)"

# View logs
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

### ✅ 4. Verify Alpaca Connection

```bash
# Check logs for successful broker connection
grep "Broker connected successfully" ~/logs/trading_*.log

# Or check systemd journal
sudo journalctl -u homeguard-trading | grep -i broker
```

### ✅ 5. Monitor First Trading Window

**Entry signals** (3:50 PM EST):
```bash
# Watch logs during entry window
tail -f ~/logs/executions_$(date +%Y%m%d).log
```

**Exit signals** (9:31 AM EST next day):
```bash
# Check exit executions
tail -f ~/logs/executions_$(date +%Y%m%d).log
```

---

## Support

If you encounter issues:

1. **Check logs**:
   - Installation: `/var/log/cloud-init-output.log`
   - Bot logs: `~/logs/trading_$(date +%Y%m%d).log`
   - System logs: `sudo journalctl -u homeguard-trading`

2. **Verify connectivity**:
   ```bash
   # Test Alpaca API
   curl -X GET https://paper-api.alpaca.markets/v2/account \
     -H "APCA-API-KEY-ID: $YOUR_KEY" \
     -H "APCA-API-SECRET-KEY: $YOUR_SECRET"
   ```

3. **Review Terraform state**:
   ```bash
   terraform show
   ```

---

## Emergency Rollback

If deployment fails:

```bash
# Destroy everything
terraform destroy

# Fix configuration
nano terraform.tfvars

# Redeploy
terraform apply
```

**Note**: If `delete_volume_on_termination = false`, your EBS volume persists and can be reattached.

---

## Next Steps

After successful deployment:

1. ✅ Monitor bot during first week
2. ✅ Set up EBS snapshots for backups
3. ✅ Document any custom configuration
4. ⚙️ Optional: Enable CloudWatch Logs
5. ⚙️ Optional: Set up SNS email alerts

---

## Resources

- [Terraform README](./README.md) - Complete documentation
- [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
- [Alpaca Dashboard](https://app.alpaca.markets/)
- [Deployment Guide](../docs/EC2_DEPLOYMENT.md) - Manual deployment (fallback)
