# Homeguard Trading Bot - Complete Deployment Guide

This guide covers deploying the Homeguard trading bot to AWS EC2 on **Windows**, **macOS**, and **Linux**, for both **first-time** and **subsequent** deployments.

---

## Table of Contents

1. [First-Time Deployment](#first-time-deployment)
   - [Windows](#windows-first-time)
   - [macOS](#macos-first-time)
   - [Linux](#linux-first-time)
2. [Subsequent Deployments](#subsequent-deployments)
3. [Post-Deployment Verification](#post-deployment-verification)
4. [Troubleshooting](#troubleshooting)

---

# First-Time Deployment

## Windows First-Time

### Prerequisites Installation

#### 1. Install Terraform

**Option A: Using Chocolatey (Recommended)**

```powershell
# Open PowerShell as Administrator
# Install Chocolatey if needed
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Terraform
choco install terraform -y

# Verify
terraform --version
```

**Option B: Manual Installation**

1. Download from https://www.terraform.io/downloads (Windows AMD64)
2. Extract `terraform.exe` to `C:\Program Files\Terraform\`
3. Add to PATH:
   - Press Win + X → System
   - Advanced system settings → Environment Variables
   - Under System variables, select Path → Edit
   - Add `C:\Program Files\Terraform\`
   - Click OK
4. Open new PowerShell: `terraform --version`

#### 2. Install AWS CLI

```powershell
# Download and install
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

# Or use Chocolatey
choco install awscli -y

# Close and reopen PowerShell, then verify
aws --version
```

#### 3. Install Git

```powershell
# Using Chocolatey
choco install git -y

# Or download from https://git-scm.com/download/win

# Verify
git --version
```

---

### AWS Setup

#### 4. Configure AWS Credentials

```powershell
# Run AWS configure
aws configure

# Enter when prompted:
AWS Access Key ID: [paste your key]
AWS Secret Access Key: [paste your secret]
Default region name: us-east-1
Default output format: json

# Verify it works
aws sts get-caller-identity
```

**Get AWS Credentials**:
1. AWS Console → IAM → Users → [Your username]
2. Security credentials tab
3. Create access key → CLI
4. Download CSV or copy keys

#### 5. Create EC2 Key Pair

```powershell
# Create .ssh directory
New-Item -ItemType Directory -Force -Path $env:USERPROFILE\.ssh

# Create key pair
aws ec2 create-key-pair `
  --key-name homeguard-trading-bot `
  --query 'KeyMaterial' `
  --output text | Out-File -Encoding ascii -FilePath $env:USERPROFILE\.ssh\homeguard-trading-bot.pem

# Verify
aws ec2 describe-key-pairs --key-names homeguard-trading-bot
```

#### 6. Get Your Public IP

```powershell
(Invoke-WebRequest -Uri "https://checkip.amazonaws.com").Content.Trim()
# Output example: 123.45.67.89
# You'll use this as: 123.45.67.89/32
```

---

### Deployment

#### 7. Clone Repository

```powershell
# Navigate to documents folder
cd $env:USERPROFILE\Documents

# Clone repository
git clone https://github.com/shuyangw/Homeguard.git
cd Homeguard

# Switch to main branch (or deployment branch)
git checkout main

# Navigate to terraform directory
cd terraform
```

#### 8. Configure Terraform Variables

```powershell
# Copy example configuration
Copy-Item terraform.tfvars.example terraform.tfvars

# Edit with Notepad
notepad terraform.tfvars

# Or use VS Code
code terraform.tfvars
```

**Edit `terraform.tfvars`**:

```hcl
# ===== REQUIRED =====
aws_region = "us-east-1"
key_pair_name = "homeguard-trading-bot"
ssh_allowed_cidrs = ["YOUR_IP/32"]  # Replace with IP from step 6

git_repo_url = "https://github.com/shuyangw/Homeguard.git"
git_branch = "main"

# ===== INSTANCE CONFIG =====
instance_type = "t4g.small"
root_volume_size = 8
delete_volume_on_termination = false

# ===== OPTIONAL (disable for minimal cost) =====
create_elastic_ip = false
enable_detailed_monitoring = false
create_cloudwatch_logs = false
create_sns_alerts = false
```

**Save and close**

#### 9. Set Alpaca Credentials

```powershell
# Set environment variables (recommended)
$env:TF_VAR_alpaca_key_id = "YOUR_ALPACA_KEY_ID_HERE"
$env:TF_VAR_alpaca_secret = "YOUR_ALPACA_SECRET_HERE"

# To persist across sessions (optional):
[System.Environment]::SetEnvironmentVariable('TF_VAR_alpaca_key_id', 'YOUR_KEY', 'User')
[System.Environment]::SetEnvironmentVariable('TF_VAR_alpaca_secret', 'YOUR_SECRET', 'User')
```

#### 10. Deploy Infrastructure

```powershell
# Initialize Terraform
terraform init

# Validate configuration
terraform validate
# Should show: "Success! The configuration is valid."

# Preview deployment
terraform plan

# Deploy (type 'yes' when prompted)
terraform apply
```

**Wait 3-5 minutes** for instance to boot and install software.

#### 11. Get Connection Information

```powershell
# View all outputs
terraform output

# Get instance IP
terraform output instance_public_ip

# Get SSH command
terraform output ssh_connection_command
```

#### 12. Connect to Instance

**Using PowerShell SSH**:
```powershell
ssh -i $env:USERPROFILE\.ssh\homeguard-trading-bot.pem ec2-user@<INSTANCE_IP>
```

**Using PuTTY**:
1. Download PuTTY: https://www.putty.org/
2. Convert .pem to .ppk using PuTTYgen:
   - Load .pem file
   - Save private key as .ppk
3. Open PuTTY:
   - Host: `ec2-user@<INSTANCE_IP>`
   - Connection → SSH → Auth → Browse to .ppk
   - Click Open

#### 13. Verify Installation

```bash
# Check installation progress
tail -f /var/log/cloud-init-output.log

# Wait for: "Installation completed at: ..."

# Verify service is installed
systemctl list-unit-files | grep homeguard

# Start trading bot
sudo systemctl start homeguard-trading

# Check status
sudo systemctl status homeguard-trading

# View logs
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

---

## macOS First-Time

### Prerequisites Installation

#### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Terraform

```bash
brew install terraform
terraform --version
```

#### 3. Install AWS CLI

```bash
brew install awscli
aws --version
```

#### 4. Install Git (usually pre-installed)

```bash
brew install git
git --version
```

---

### AWS Setup

#### 5. Configure AWS Credentials

```bash
aws configure

# Enter when prompted:
# AWS Access Key ID: [paste your key]
# AWS Secret Access Key: [paste your secret]
# Default region: us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

#### 6. Create EC2 Key Pair

```bash
# Create .ssh directory if needed
mkdir -p ~/.ssh

# Create key pair
aws ec2 create-key-pair \
  --key-name homeguard-trading-bot \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/homeguard-trading-bot.pem

# Set permissions
chmod 400 ~/.ssh/homeguard-trading-bot.pem

# Verify
aws ec2 describe-key-pairs --key-names homeguard-trading-bot
```

#### 7. Get Your Public IP

```bash
curl https://checkip.amazonaws.com
# Output: 123.45.67.89
# Use as: 123.45.67.89/32
```

---

### Deployment

#### 8. Clone Repository

```bash
cd ~/Documents  # or your preferred location

git clone https://github.com/shuyangw/Homeguard.git
cd Homeguard

git checkout main  # or deployment branch

cd terraform
```

#### 9. Configure Variables

```bash
# Copy example
cp terraform.tfvars.example terraform.tfvars

# Edit with your preferred editor
nano terraform.tfvars
# Or: vim terraform.tfvars
# Or: code terraform.tfvars (VS Code)
```

**Edit `terraform.tfvars`**:

```hcl
aws_region = "us-east-1"
key_pair_name = "homeguard-trading-bot"
ssh_allowed_cidrs = ["YOUR_IP/32"]

git_repo_url = "https://github.com/shuyangw/Homeguard.git"
git_branch = "main"

instance_type = "t4g.small"
root_volume_size = 8
delete_volume_on_termination = false
```

#### 10. Set Alpaca Credentials

```bash
# Export environment variables
export TF_VAR_alpaca_key_id="YOUR_KEY"
export TF_VAR_alpaca_secret="YOUR_SECRET"

# To persist, add to ~/.zshrc or ~/.bash_profile:
echo 'export TF_VAR_alpaca_key_id="YOUR_KEY"' >> ~/.zshrc
echo 'export TF_VAR_alpaca_secret="YOUR_SECRET"' >> ~/.zshrc
source ~/.zshrc
```

#### 11. Deploy

```bash
terraform init
terraform validate
terraform plan
terraform apply  # type 'yes' when prompted
```

#### 12. Connect

```bash
# Get instance IP
terraform output instance_public_ip

# SSH
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<INSTANCE_IP>

# Or use output command
$(terraform output -raw ssh_connection_command)
```

#### 13. Verify Installation

```bash
tail -f /var/log/cloud-init-output.log
# Wait for completion

sudo systemctl start homeguard-trading
sudo systemctl status homeguard-trading
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

---

## Linux First-Time

### Prerequisites Installation

**Ubuntu/Debian**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install prerequisites
sudo apt install -y wget unzip curl git

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip
unzip terraform_1.7.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform --version

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

**RHEL/Fedora/Amazon Linux**:

```bash
sudo yum update -y
sudo yum install -y wget unzip curl git

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip
unzip terraform_1.7.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

---

### AWS Setup & Deployment

**Steps 5-13 are identical to macOS** (see above)

The commands are the same for:
- AWS configuration
- Key pair creation
- Repository cloning
- Terraform deployment
- SSH connection

---

# Subsequent Deployments

After your first deployment, future deployments are much faster.

## All Platforms

### Update Code Only (No Infrastructure Changes)

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<INSTANCE_IP>

# Stop service
sudo systemctl stop homeguard-trading

# Update code
cd ~/Homeguard
git pull origin main

# Reinstall dependencies (if requirements.txt changed)
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl start homeguard-trading
sudo systemctl status homeguard-trading
```

---

### Redeploy Infrastructure

**If you need to recreate EC2 instance**:

#### Windows

```powershell
cd $env:USERPROFILE\Documents\Homeguard\terraform

# Set credentials (if not persisted)
$env:TF_VAR_alpaca_key_id = "YOUR_KEY"
$env:TF_VAR_alpaca_secret = "YOUR_SECRET"

# Redeploy
terraform apply
```

#### macOS/Linux

```bash
cd ~/Documents/Homeguard/terraform

# Set credentials (if not persisted)
export TF_VAR_alpaca_key_id="YOUR_KEY"
export TF_VAR_alpaca_secret="YOUR_SECRET"

# Redeploy
terraform apply
```

**Time**: ~2-5 minutes (vs 30+ minutes first time)

---

### Change Instance Type

**Example: Upgrade from t4g.small to t4g.medium**

```bash
# Edit terraform.tfvars
nano terraform.tfvars  # or notepad on Windows

# Change:
instance_type = "t4g.medium"

# Apply changes
terraform apply
# Note: This requires instance stop/start (brief downtime)
```

---

### Destroy and Rebuild

**Complete teardown and rebuild**:

#### Windows

```powershell
cd $env:USERPROFILE\Documents\Homeguard\terraform

# Destroy
terraform destroy  # type 'yes'

# Rebuild
terraform apply  # type 'yes'
```

#### macOS/Linux

```bash
cd ~/Documents/Homeguard/terraform

terraform destroy
terraform apply
```

---

# Post-Deployment Verification

## Checklist

After every deployment:

### 1. Service Status

```bash
sudo systemctl status homeguard-trading
# Should show: "active (running)"
```

### 2. Logs Check

```bash
# Check for errors
grep ERROR ~/logs/trading_$(date +%Y%m%d).log

# Verify Alpaca connection
grep "Broker connected" ~/logs/trading_*.log
```

### 3. Disk Space

```bash
df -h /home/ec2-user
# Should have several GB free
```

### 4. Memory Usage

```bash
free -h
# Used should be <1.5 GB on t4g.small
```

### 5. Network Connectivity

```bash
# Test Alpaca API
curl -X GET https://paper-api.alpaca.markets/v2/clock
# Should return JSON with market status
```

---

## Management Scripts (Quick Access)

After deployment, use the SSH management scripts for easy instance monitoring:

### Windows

```bash
# From repository root
scripts\ec2\check_bot.bat          # Check bot status + recent activity
scripts\ec2\view_logs.bat          # Stream live logs
scripts\ec2\restart_bot.bat        # Restart bot service
scripts\ec2\daily_health_check.bat # Automated 6-point health check
scripts\ec2\connect.bat            # SSH into instance
```

### Linux/Mac

```bash
# From repository root
scripts/ec2/check_bot.sh           # Check bot status + recent activity
scripts/ec2/view_logs.sh           # Stream live logs
scripts/ec2/restart_bot.sh         # Restart bot service
scripts/ec2/daily_health_check.sh  # Automated 6-point health check
scripts/ec2/connect.sh             # SSH into instance
```

### Automated Health Check

The `daily_health_check` script performs 6-point validation:

1. **Instance State** - Verifies EC2 instance is running
2. **Bot Service Status** - Checks systemd service is active
3. **Recent Errors** - Counts errors in last hour
4. **Resource Usage** - Shows memory and CPU utilization
5. **Last Activity** - Displays recent log entries
6. **Market Status** - Shows current market state (OPEN/CLOSED)

**Recommended Daily Routine**:
```bash
# Morning (before market open)
scripts\ec2\daily_health_check.bat  # Windows
scripts/ec2/daily_health_check.sh   # Linux/Mac

# If issues found, restart:
scripts\ec2\restart_bot.bat         # Windows
scripts/ec2/restart_bot.sh          # Linux/Mac
```

**Comprehensive Monitoring Guide**: See [`HEALTH_CHECK_CHEATSHEET.md`](../../HEALTH_CHECK_CHEATSHEET.md) for:
- Complete monitoring commands
- Common issues and fixes
- Lambda scheduler verification
- Git repository sync checks

---

## Monitor First Trading Window

### Entry Signals (3:50 PM EST)

```bash
# Start watching logs before 3:50 PM
tail -f ~/logs/trading_$(date +%Y%m%d).log

# Look for:
# - "Entry window active"
# - "Generating Entry Signals"
# - Order executions
```

### Exit Signals (9:31 AM EST next day)

```bash
tail -f ~/logs/executions_$(date +%Y%m%d).log

# Look for SELL orders
```

---

# Troubleshooting

## Common Issues

### 1. Terraform: "InvalidKeyPair.NotFound"

**Problem**: Key pair doesn't exist

**Solution**:
```bash
# Create key pair first
aws ec2 create-key-pair --key-name homeguard-trading-bot \
  --query 'KeyMaterial' --output text > ~/.ssh/homeguard-trading-bot.pem
chmod 400 ~/.ssh/homeguard-trading-bot.pem
```

---

### 2. SSH: "Permission denied (publickey)"

**Problem**: Wrong key or permissions

**Solution**:
```bash
# Check permissions
chmod 400 ~/.ssh/homeguard-trading-bot.pem

# Use correct key
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<IP>
```

---

### 3. Can't Connect: "Connection timed out"

**Problem**: Security group blocking SSH

**Solution**:
1. Get your current IP: `curl https://checkip.amazonaws.com`
2. Update `terraform.tfvars`:
   ```hcl
   ssh_allowed_cidrs = ["NEW_IP/32"]
   ```
3. Reapply: `terraform apply`

---

### 4. Bot Not Starting

**Check logs**:
```bash
sudo journalctl -u homeguard-trading -n 100
tail -n 100 ~/logs/trading_$(date +%Y%m%d).log
```

**Common causes**:
- Alpaca credentials incorrect
- Config files missing
- Python dependencies not installed

**Fix**:
```bash
# Verify .env file
cat ~/Homeguard/.env

# Reinstall dependencies
cd ~/Homeguard
source venv/bin/activate
pip install -r requirements.txt

# Restart
sudo systemctl restart homeguard-trading
```

---

### 5. Out of Memory

**Check usage**:
```bash
free -h
```

**If >80% used**:
```bash
# Upgrade instance type
# Edit terraform.tfvars:
instance_type = "t4g.medium"

# Apply
terraform apply
```

---

### 6. Disk Full

**Check disk**:
```bash
df -h /home/ec2-user
```

**Clean old logs**:
```bash
find ~/logs -name "*.log*" -mtime +30 -delete
```

**Increase volume size**:
```hcl
# terraform.tfvars
root_volume_size = 20  # Changed from 8

# Apply
terraform apply
```

---

## Getting Help

### Log Files

**Application logs**:
- Main: `~/logs/trading_YYYYMMDD.log`
- Executions: `~/logs/executions_YYYYMMDD.log`

**System logs**:
- Installation: `/var/log/cloud-init-output.log`
- Service: `sudo journalctl -u homeguard-trading`

### Useful Commands

```bash
# Service status
sudo systemctl status homeguard-trading

# Restart service
sudo systemctl restart homeguard-trading

# View recent logs
sudo journalctl -u homeguard-trading -n 100 --no-pager

# Test Alpaca connection
cd ~/Homeguard
source venv/bin/activate
python scripts/trading/test_alpaca_connection.py
```

---

## Cost Management

### Monitor Costs

1. AWS Console → Billing & Cost Management
2. View by service (should see EC2, EBS)
3. Expected: ~$13/month

### Set Billing Alert

1. Billing → Budgets → Create budget
2. Monthly cost budget
3. Set amount: $20
4. Add email alert

### Stop Instance Temporarily

```bash
# Stop (still charges for EBS, saves EC2 cost)
terraform apply -var="instance_type=t4g.nano"  # Smallest type
# Or use AWS console to stop

# Start again
terraform apply  # Restores original instance type
```

---

## Next Steps

After successful deployment:

1. ✅ Monitor for one week
2. ✅ Set up EBS snapshots (weekly backups)
3. ✅ Document any customizations
4. ⚙️ Optional: Enable CloudWatch Logs
5. ⚙️ Optional: Set up SNS email alerts
6. ⚙️ Optional: Create staging environment

---

## Quick Reference

### Daily Operations

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<IP>

# Check status
sudo systemctl status homeguard-trading

# View logs
tail -f ~/logs/trading_$(date +%Y%m%d).log

# Check trades
tail ~/logs/executions_$(date +%Y%m%d).log
```

### Terraform Commands

```bash
cd terraform

terraform init      # First time only
terraform validate  # Check config
terraform plan      # Preview changes
terraform apply     # Deploy/update
terraform destroy   # Remove everything
terraform output    # View outputs
```

### Estimated Times

| Task | First Time | Subsequent |
|------|------------|------------|
| Prerequisites | 30 min | 0 min |
| AWS Setup | 10 min | 0 min |
| Deployment | 5 min | 2 min |
| Verification | 5 min | 2 min |
| **Total** | **50 min** | **4 min** |

---

## Platform-Specific Notes

### Windows

- Use PowerShell (not CMD)
- Paths use `$env:USERPROFILE` instead of `~`
- Line endings: Use `-Encoding ascii` for .pem files
- SSH: Built into Windows 10/11, or use PuTTY

### macOS

- Use Terminal or iTerm2
- Homebrew makes installation easy
- Paths use `~` for home directory
- SSH: Built-in, works perfectly

### Linux

- Most native experience
- Package managers vary (apt, yum, etc.)
- All commands work as documented
- SSH: Native and seamless

---

**That's it! You're ready to deploy on any platform.** 🚀
