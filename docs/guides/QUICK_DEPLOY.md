# Quick Deploy Reference

**For users who have already completed first-time setup.**

---

## Prerequisites (Already Installed)

- ✅ Terraform
- ✅ AWS CLI configured
- ✅ EC2 key pair created
- ✅ Repository cloned

---

## Windows Quick Deploy

```powershell
# 1. Navigate to terraform directory
cd $env:USERPROFILE\Documents\Homeguard\terraform

# 2. Set Alpaca credentials
$env:TF_VAR_alpaca_key_id = "YOUR_KEY"
$env:TF_VAR_alpaca_secret = "YOUR_SECRET"

# 3. Deploy
terraform apply
# Type 'yes' when prompted

# 4. Get instance IP
terraform output instance_public_ip

# 5. Connect
ssh -i $env:USERPROFILE\.ssh\homeguard-trading-bot.pem ec2-user@<IP>

# 6. Verify
sudo systemctl status homeguard-trading
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

**Time**: 2-5 minutes

---

## macOS/Linux Quick Deploy

```bash
# 1. Navigate to terraform directory
cd ~/Documents/Homeguard/terraform

# 2. Set Alpaca credentials
export TF_VAR_alpaca_key_id="YOUR_KEY"
export TF_VAR_alpaca_secret="YOUR_SECRET"

# 3. Deploy
terraform apply
# Type 'yes' when prompted

# 4. Get instance IP
terraform output instance_public_ip

# 5. Connect
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<IP>

# 6. Verify
sudo systemctl status homeguard-trading
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

**Time**: 2-5 minutes

---

## Update Code Only (No Redeploy)

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<IP>

# Stop bot
sudo systemctl stop homeguard-trading

# Pull latest code
cd ~/Homeguard
git pull

# Update dependencies (if needed)
source venv/bin/activate
pip install -r requirements.txt

# Restart
sudo systemctl start homeguard-trading

# Verify
sudo systemctl status homeguard-trading
```

**Time**: 1-2 minutes

---

## Common Operations

### Change Instance Type

```bash
# Edit terraform.tfvars
instance_type = "t4g.medium"  # Change from t4g.small

# Apply
terraform apply
```

### Destroy Instance

```bash
terraform destroy
# Type 'yes'
```

### View Logs

```bash
# SSH to instance first
tail -f ~/logs/trading_$(date +%Y%m%d).log
tail -f ~/logs/executions_$(date +%Y%m%d).log
```

### Restart Bot

```bash
sudo systemctl restart homeguard-trading
```

---

## Troubleshooting

### Can't SSH

```bash
# Get current IP
curl https://checkip.amazonaws.com

# Update terraform.tfvars
ssh_allowed_cidrs = ["NEW_IP/32"]

# Apply
terraform apply
```

### Bot Not Running

```bash
sudo systemctl status homeguard-trading
sudo journalctl -u homeguard-trading -n 50
```

### Check Alpaca Connection

```bash
grep "Broker connected" ~/logs/trading_*.log
```

---

## Cost Estimate

- **t4g.small**: $12.26/month
- **8 GB EBS**: $0.64/month
- **Total**: ~$13/month

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Deploy | `terraform apply` |
| Destroy | `terraform destroy` |
| SSH | `ssh -i ~/.ssh/homeguard-trading-bot.pem ec2-user@<IP>` |
| Status | `sudo systemctl status homeguard-trading` |
| Logs | `tail -f ~/logs/trading_$(date +%Y%m%d).log` |
| Restart | `sudo systemctl restart homeguard-trading` |
| Update Code | `cd ~/Homeguard && git pull` |
