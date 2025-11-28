# Homeguard Trading Bot - Terraform Infrastructure

This Terraform module deploys the Homeguard trading bot to AWS EC2 with automated setup.

## Features

- ✅ **Automated deployment** - `terraform apply` -> fully configured in 5 minutes
- ✅ **ARM64 optimized** - t4g instances (40% cheaper than x86)
- ✅ **Production-ready** - systemd service, auto-restart, log rotation
- ✅ **Secure** - IMDSv2, SSH-only access, encrypted EBS volumes
- ✅ **Cost-optimized** - ~$13/month for complete infrastructure
- ✅ **Optional features** - CloudWatch, SNS alerts, Elastic IP
- ✅ **Discord bot addon** - Optional read-only observability via natural language

---

## Prerequisites

### 1. Install Terraform

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip
unzip terraform_1.7.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify installation
terraform --version
```

### 2. Configure AWS Credentials

```bash
# Install AWS CLI
brew install awscli  # macOS
# or: pip install awscli

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Verify
aws sts get-caller-identity
```

### 3. Create EC2 Key Pair

```bash
# Via AWS CLI
aws ec2 create-key-pair --key-name homeguard-trading-bot --query 'KeyMaterial' --output text > ~/.ssh/homeguard-trading-bot.pem
chmod 400 ~/.ssh/homeguard-trading-bot.pem

# Or via AWS Console:
# EC2 → Key Pairs → Create Key Pair → Download .pem file
```

### 4. Get Your IP Address

```bash
# Get your public IP
curl https://checkip.amazonaws.com

# You'll use this in terraform.tfvars for SSH access
```

---

## Quick Start

### 1. Configure Variables

```bash
cd terraform

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings
nano terraform.tfvars
```

**Minimal configuration** (`terraform.tfvars`):
```hcl
# Required settings
key_pair_name    = "homeguard-trading-bot"
ssh_allowed_cidrs = ["YOUR_IP_HERE/32"]  # Replace with your IP

# Alpaca credentials (or set via environment variables - see below)
alpaca_key_id = "your_paper_key_id"
alpaca_secret = "your_paper_secret"
```

**Recommended: Use environment variables for secrets**:
```bash
export TF_VAR_alpaca_key_id="your_paper_key_id"
export TF_VAR_alpaca_secret="your_paper_secret"
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Review Plan

```bash
terraform plan
```

This shows what will be created:
- EC2 instance (t4g.small)
- Security group (SSH access)
- EBS volume (8 GB)

### 4. Deploy Infrastructure

```bash
terraform apply

# Type 'yes' when prompted
```

**Deployment takes ~5 minutes**:
- Terraform creates resources (~30 seconds)
- User-data script installs software (~3-4 minutes)

### 5. Connect and Verify

```bash
# Get SSH command from output
terraform output ssh_connection_command

# Or manually:
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<INSTANCE_IP>

# Check installation progress
tail -f /var/log/cloud-init-output.log

# When complete, start the bot
sudo systemctl start homeguard-trading

# Verify it's running
sudo systemctl status homeguard-trading

# View logs
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

---

## Configuration Options

### Instance Types

| Instance | vCPUs | RAM | Cost/Month | Use Case |
|----------|-------|-----|------------|----------|
| **t4g.small** | 2 | 2 GB | $12.26 | Trading only (recommended) |
| **t4g.medium** | 2 | 4 GB | $24.53 | Trading + occasional backtests |
| **t4g.large** | 2 | 8 GB | $49.06 | Heavy backtesting |

Set in `terraform.tfvars`:
```hcl
instance_type = "t4g.small"
```

### Storage

| Size | Cost/Month | Use Case |
|------|------------|----------|
| **8 GB** | $0.64 | Trading only (default) |
| **20 GB** | $1.60 | Trading + historical data |
| **50 GB** | $4.00 | Extended backtesting |

Set in `terraform.tfvars`:
```hcl
root_volume_size = 8
```

### Optional Features

Enable in `terraform.tfvars`:

```hcl
# Static IP address (recommended for production)
create_elastic_ip = true  # Costs $0/month if instance running

# CloudWatch Logs (remote log access)
create_cloudwatch_logs = true
log_retention_days     = 30  # $0.50-1/month

# Email alerts
create_sns_alerts = true
alert_email       = "you@example.com"  # Free tier eligible

# CloudWatch alarms
create_cloudwatch_alarms = true  # Requires create_sns_alerts = true
```

### Discord Bot (Optional Addon)

The Discord bot provides read-only observability through natural language queries.

**Configuration**:
```hcl
# In terraform.tfvars (recommended: use environment variables instead)
discord_token            = ""  # Get from Discord Developer Portal
anthropic_api_key        = ""  # Get from console.anthropic.com
discord_allowed_channels = ""  # Comma-separated channel IDs
```

**Recommended: Use environment variables**:
```bash
export TF_VAR_discord_token="your_bot_token"
export TF_VAR_anthropic_api_key="sk-ant-api03-..."
export TF_VAR_discord_allowed_channels="123456789012345678"
```

**Post-deployment setup**:
1. Follow the Discord bot setup guide: `docs/guides/20251127_DISCORD_BOT_SETUP.md`
2. Start the service: `sudo systemctl start homeguard-discord`
3. Verify: `sudo systemctl status homeguard-discord`

**Cost**: ~$5-15/month in Claude API usage (pay-per-query)

---

## Common Operations

### View Outputs

```bash
# Show all outputs
terraform output

# Show specific output
terraform output instance_public_ip
terraform output ssh_connection_command
```

### Update Instance

```bash
# Change instance type in terraform.tfvars
instance_type = "t4g.medium"

# Apply changes (requires instance restart)
terraform apply
```

### Update Code on Instance

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@$(terraform output -raw instance_public_ip)

# Pull latest code
cd ~/Homeguard
git pull

# Restart service
sudo systemctl restart homeguard-trading
```

### Destroy Infrastructure

```bash
# WARNING: This deletes everything (if delete_on_termination = true)
terraform destroy

# Type 'yes' to confirm
```

**Note**: If `delete_volume_on_termination = false` (recommended), the EBS volume will persist and can be reattached later.

---

## Cost Breakdown

### Minimal Configuration (Default)

| Resource | Specification | Cost/Month |
|----------|--------------|------------|
| EC2 Instance | t4g.small | $12.26 |
| EBS Volume | 8 GB GP3 | $0.64 |
| Data Transfer | Minimal | $0.10 |
| **Total** | | **$13.00** |

### With Optional Features

| Feature | Additional Cost |
|---------|----------------|
| Elastic IP | $0 (if running) |
| CloudWatch Logs | +$0.50-1.00 |
| SNS Alerts | $0 (free tier) |
| Detailed Monitoring | +$2.10 |

---

## Security Best Practices

### 1. Restrict SSH Access

```hcl
# In terraform.tfvars
ssh_allowed_cidrs = ["YOUR_IP/32"]  # Only your IP
```

### 2. Use Environment Variables for Secrets

```bash
# Don't put credentials in terraform.tfvars
export TF_VAR_alpaca_key_id="..."
export TF_VAR_alpaca_secret="..."

terraform apply
```

### 3. Enable Volume Encryption

Already enabled by default:
```hcl
encrypted = true  # In main.tf
```

### 4. Use IMDSv2

Already configured:
```hcl
http_tokens = "required"  # In main.tf
```

### 5. Protect Root Volume

```hcl
# In terraform.tfvars
delete_volume_on_termination = false
```

---

## Disaster Recovery

### Backup Strategy

**Option 1: EBS Snapshots (Recommended)**

```bash
# Create snapshot via CLI
aws ec2 create-snapshot \
  --volume-id $(terraform output -raw volume_id) \
  --description "Homeguard trading bot backup"

# Or use AWS Backup (automated)
```

**Option 2: Terraform State + Git**

Your infrastructure is version-controlled:
- Terraform state tracks all resources
- Code is in Git
- `terraform apply` recreates everything

### Recovery Process

If instance fails:

```bash
# 1. Destroy failed instance
terraform destroy -target=aws_instance.trading_bot

# 2. Recreate from scratch
terraform apply

# 3. Or restore from snapshot (manual)
```

---

## Troubleshooting

### Issue: "Error: InvalidKeyPair.NotFound"

**Fix**: Create key pair first:
```bash
aws ec2 create-key-pair --key-name homeguard-trading-bot --query 'KeyMaterial' --output text > ~/.ssh/homeguard-trading-bot.pem
chmod 400 ~/.ssh/homeguard-trading-bot.pem
```

### Issue: Can't SSH to instance

**Check**:
1. Security group allows your IP:
   ```bash
   curl https://checkip.amazonaws.com  # Verify your current IP
   ```

2. Key pair permissions:
   ```bash
   chmod 400 ~/.ssh/homeguard-trading-bot.pem
   ```

3. Instance is running:
   ```bash
   terraform output instance_state  # Should be "running"
   ```

### Issue: Bot not starting

**Check user-data logs**:
```bash
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@$(terraform output -raw instance_public_ip)

# View installation log
tail -f /var/log/cloud-init-output.log

# Check for errors
grep -i error /var/log/cloud-init-output.log
```

### Issue: High costs

**Verify resources**:
```bash
# Check what's deployed
terraform show

# View estimated costs
terraform output estimated_monthly_cost

# Ensure you're using t4g.small
terraform output | grep instance_type
```

---

## Terraform State Management

### Remote State (Recommended for Teams)

Store state in S3 for collaboration:

```hcl
# Create backend.tf
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "homeguard/terraform.tfstate"
    region = "us-east-1"
  }
}
```

### Local State (Default)

State stored in `terraform.tfstate` (don't commit to Git!)

Add to `.gitignore`:
```
terraform.tfstate
terraform.tfstate.backup
.terraform/
terraform.tfvars  # Contains secrets
```

---

## Advanced Usage

### Multiple Environments

```bash
# Production
terraform workspace new production
terraform apply -var-file=production.tfvars

# Staging
terraform workspace new staging
terraform apply -var-file=staging.tfvars
```

### Import Existing Resources

If you created an instance manually:

```bash
terraform import aws_instance.trading_bot i-1234567890abcdef0
```

### Customize User Data

Edit `user-data.sh` and reapply:

```bash
# User data only runs on first boot
# To re-run, terminate and recreate instance
terraform taint aws_instance.trading_bot
terraform apply
```

---

## Support

### Getting Help

1. **Terraform errors**: Check `terraform.log`
   ```bash
   TF_LOG=DEBUG terraform apply
   ```

2. **Instance issues**: Check cloud-init logs
   ```bash
   sudo tail -f /var/log/cloud-init-output.log
   ```

3. **Bot issues**: Check application logs
   ```bash
   tail -f ~/logs/trading_$(date +%Y%m%d).log
   ```

### Clean Up

```bash
# Destroy everything
terraform destroy

# Remove Terraform files
rm -rf .terraform terraform.tfstate*
```

---

## What's Deployed

This Terraform module creates:

- ✅ **EC2 instance** (t4g.small, Amazon Linux 2023 ARM64)
- ✅ **Security group** (SSH access only)
- ✅ **EBS volume** (8 GB GP3, encrypted)
- ✅ **Automated setup** (installs Python, clones repo, creates systemd services)
- ✅ **Lambda scheduling** (auto-start 9:00 AM, auto-stop 4:30 PM ET Mon-Fri)
- ✅ **EventBridge rules** (cron triggers for Lambda functions)
- ✅ **IAM roles & policies** (Lambda permissions for EC2 control)
- ✅ **CloudWatch Log Groups** (Lambda execution logs, 90-day retention)
- ⚙️ **Optional**: Elastic IP, SNS alerts
- ⚙️ **Optional**: Discord bot (read-only observability addon)

**Not included**:
- Load balancer (not needed)
- Auto-scaling (single instance sufficient)
- RDS database (no external DB required)
- VPC (uses default VPC)

---

## Post-Deployment Management

Once deployed, use the SSH management scripts for easy instance management:

### Quick-Access Scripts

**Location**: `scripts/ec2/` (repository root)

**Windows**:
```bash
# Check bot status
scripts\ec2\check_bot.bat

# View live logs
scripts\ec2\view_logs.bat

# Restart bot service
scripts\ec2\restart_bot.bat

# Daily health check (6-point validation)
scripts\ec2\daily_health_check.bat

# SSH into instance
scripts\ec2\connect.bat
```

**Linux/Mac**:
```bash
# Check bot status
scripts/ec2/check_bot.sh

# View live logs
scripts/ec2/view_logs.sh

# Restart bot service
scripts/ec2/restart_bot.sh

# Daily health check (6-point validation)
scripts/ec2/daily_health_check.sh

# SSH into instance
scripts/ec2/connect.sh
```

### Health Monitoring

See [`HEALTH_CHECK_CHEATSHEET.md`](../docs/HEALTH_CHECK_CHEATSHEET.md) for comprehensive monitoring guide:
- Daily health check routines
- Common issues and fixes
- Advanced monitoring commands
- Lambda scheduler verification
- Git repository sync checks

### Automated Scheduling

The deployment includes Lambda functions that automatically manage the instance:

**Start Function** (`homeguard-start-instance`):
- Triggers: 9:00 AM ET Monday-Friday
- Action: Starts EC2 instance if stopped
- Logs: `/aws/lambda/homeguard-start-instance` (CloudWatch)

**Stop Function** (`homeguard-stop-instance`):
- Triggers: 4:30 PM ET Monday-Friday
- Action: Stops EC2 instance if running
- Logs: `/aws/lambda/homeguard-stop-instance` (CloudWatch)

**Cost Savings**: Running only during market hours (157.5 hrs/month) saves ~46% vs 24/7 operation.

**Verify Scheduling**:
```bash
# Check if Lambda functions exist
aws lambda list-functions | grep homeguard

# View Lambda logs
aws logs tail /aws/lambda/homeguard-start-instance --since 24h
aws logs tail /aws/lambda/homeguard-stop-instance --since 24h

# Check EventBridge rules
aws events list-rules | grep homeguard
```

### Bot Service Management

The trading bot runs as a systemd service with auto-restart:

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<INSTANCE_IP>

# Check service status
sudo systemctl status homeguard-trading

# View live logs
sudo journalctl -u homeguard-trading -f

# Restart service
sudo systemctl restart homeguard-trading

# View recent errors
sudo journalctl -u homeguard-trading -p err -n 20
```

**Service Features**:
- Auto-restart on failure (10-second delay)
- Resource limits (1GB RAM, 150% CPU)
- Logging to systemd journal + file logs
- Starts automatically when instance boots

---

## Next Steps

After deployment:

1. ✅ Deploy with `terraform apply`
2. ✅ Wait for automated setup (~3-4 minutes)
3. ✅ Run health check: `scripts\ec2\daily_health_check.bat` (or `.sh`)
4. ✅ Verify bot is running: `scripts\ec2\check_bot.bat` (or `.sh`)
5. ✅ Monitor first trading window: `scripts\ec2\view_logs.bat` (or `.sh`)
6. ✅ Verify Lambda scheduling: Check CloudWatch logs for start/stop events
7. ✅ Set up backups: Enable EBS snapshots (optional)
8. 📖 Read [`HEALTH_CHECK_CHEATSHEET.md`](../docs/HEALTH_CHECK_CHEATSHEET.md) for monitoring best practices

**Daily Routine**:
- Morning: Run `daily_health_check.bat/.sh` before market open
- During market: Check `view_logs.bat/.sh` occasionally
- Issues: Use `restart_bot.bat/.sh` if needed

---

## Cost Optimization Tips

1. **✅ Automated scheduling** - Already configured! Lambda stops instance after market close (saves ~46%)
2. **Use t4g.small** (not t3.small) - 40% cheaper ARM instances
3. **Keep default 8 GB volume** - sufficient for trading only
4. **Elastic IP** included - no additional cost when instance running, minimal when stopped
5. **Start with local logs** - systemd journal sufficient for most needs
6. **Use free tier SNS** for alerts (if you enable it)
7. **Monitor costs** - Check AWS billing dashboard regularly

**Current Setup Savings**:
- Running only market hours: ~$7/month (vs ~$13/month 24/7)
- t4g.small ARM: 40% cheaper than equivalent x86 instance
- **Total monthly cost: ~$7.00**

---

## References

- [Terraform AWS Provider Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/)
- [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
