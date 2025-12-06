# Trading Bot Health Check Cheatsheet

Quick reference for monitoring your Homeguard trading bot on EC2.

> **Note**: Replace placeholders with your actual values from `.env` file or `terraform output`.
> Use the pre-configured scripts in `scripts/ec2/` which automatically read from `.env`.

**Instance IP**: `<YOUR_EC2_IP>` (see `EC2_IP` in `.env`)
**Instance ID**: `<YOUR_INSTANCE_ID>` (see `EC2_INSTANCE_ID` in `.env`)
**SSH Key**: `~/.ssh/homeguard-trading.pem` (see `EC2_SSH_KEY_PATH` in `.env`)

---

## Quick Health Checks

### 1. Check if Bot is Running ⚡

**Quick Script** (recommended):
```bash
scripts/ec2/check_bot.sh       # Linux/Mac
scripts\ec2\check_bot.bat      # Windows
```

**Manual SSH**:
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo systemctl status homeguard-trading --no-pager"
```

**What to look for**:
- ✅ `Active: active (running)` in green
- ✅ Recent timestamp (not stuck)
- ❌ `Active: failed` or `inactive (dead)` = problem

---

### 2. View Live Activity 📊

**Quick Script**:
```bash
scripts/ec2/view_logs.sh       # Linux/Mac
scripts\ec2\view_logs.bat      # Windows
```

**Manual SSH**:
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading -f"
```

**Press Ctrl+C to stop**

**What to look for**:
- ✅ Regular updates every 15 seconds
- ✅ `Market: OPEN` during trading hours (9:30 AM - 4:00 PM ET)
- ✅ `Signals: N` and `Orders: N/N` when trading
- ❌ No updates = bot is stuck
- ❌ Repeated errors = problem

---

### 3. Check Recent Activity (Last 10 Lines) 📝

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading -n 10 --no-pager"
```

---

### 4. Check for Errors ⚠️

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading -p err -n 20 --no-pager"
```

**What to look for**:
- ✅ No output = no errors
- ❌ API errors = check Alpaca credentials
- ❌ Connection errors = network issue
- ❌ Import errors = missing dependencies

---

### 5. Check Resource Usage 💻

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo systemctl status homeguard-trading --no-pager | grep -E 'Memory|CPU'"
```

**What to look for**:
- ✅ Memory: < 500M (normal)
- ✅ CPU: < 5s total time
- ❌ Memory: > 900M = approaching limit (1GB max)
- ❌ CPU: very high = possible infinite loop

---

### 6. Check Instance State 🖥️

```bash
aws ec2 describe-instances --instance-ids <YOUR_INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name' --output text
```

**Expected**:
- ✅ `running` during market hours (9 AM - 4:30 PM ET weekdays)
- ✅ `stopped` outside market hours
- ❌ `stopping` or `pending` for too long = problem

---

## Daily Health Check Routine (2 minutes)

### Morning (Before Market Opens - 9:00-9:30 AM ET)

```bash
# 1. Check instance is running
aws ec2 describe-instances --instance-ids <YOUR_INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name'

# 2. Check bot service status
scripts/ec2/check_bot.sh

# 3. Verify no recent errors
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading -p err --since '1 hour ago' --no-pager"
```

**Expected**: Instance running, bot active, no errors

---

### During Market Hours (9:30 AM - 4:00 PM ET)

```bash
# View live activity (watch for 30 seconds)
scripts/ec2/view_logs.sh
```

**Expected**:
- Market status: `OPEN`
- Regular checks every 15 seconds
- Trade signals and orders (if conditions met)

---

### After Market Close (4:00-4:30 PM ET)

```bash
# Check if logs were flushed
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "ls -lh ~/logs/live_trading/paper/ | tail -5"
```

**Expected**: Today's log file updated with recent timestamp

---

### Evening (After 4:30 PM ET)

```bash
# Verify instance was stopped
aws ec2 describe-instances --instance-ids <YOUR_INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name'
```

**Expected**: `stopped` or `stopping`

---

## Common Issues & Quick Fixes

### Issue: Bot is not running

```bash
# Check status
sudo systemctl status homeguard-trading

# Check recent logs for errors
sudo journalctl -u homeguard-trading -n 50

# Restart bot
scripts/ec2/restart_bot.sh
```

---

### Issue: Bot stuck (no updates)

```bash
# Force restart
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo systemctl restart homeguard-trading"

# Verify it restarted
scripts/ec2/check_bot.sh
```

---

### Issue: API errors (Alpaca credentials)

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>

# Check .env file exists and has correct format
cat ~/Homeguard/.env

# Should see:
# ALPACA_PAPER_KEY_ID=PK...
# ALPACA_PAPER_SECRET_KEY=...

# If missing or wrong, fix it:
nano ~/Homeguard/.env

# Restart bot
sudo systemctl restart homeguard-trading
exit
```

---

### Issue: Out of memory

```bash
# Check memory usage
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "free -h"

# If memory is full, restart bot
scripts/ec2/restart_bot.sh
```

---

### Issue: Instance didn't start on schedule

```bash
# Check Lambda logs for start function
aws logs tail /aws/lambda/homeguard-start-instance --since 1h

# Manually start instance
aws ec2 start-instances --instance-ids <YOUR_INSTANCE_ID>

# Wait 2 minutes, then verify bot started
scripts/ec2/check_bot.sh
```

---

### Issue: Instance didn't stop on schedule

```bash
# Check Lambda logs for stop function
aws logs tail /aws/lambda/homeguard-stop-instance --since 1h

# Manually stop instance
aws ec2 stop-instances --instance-ids <YOUR_INSTANCE_ID>
```

---

## Advanced Monitoring

### View Today's Trading Logs

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "tail -100 ~/logs/live_trading/paper/trading_\$(date +%Y%m%d).log"
```

---

### Download All Logs

```bash
scp -i ~/.ssh/homeguard-trading.pem -r ec2-user@<YOUR_EC2_IP>:~/logs/live_trading/paper/ ./downloaded_logs/
```

---

### Check if Bot is Trading

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading --since today | grep -E 'SIGNAL|ORDER|TRADE'"
```

---

### View System Performance

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "top -b -n 1 | head -20"
```

---

### Check Disk Space

```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "df -h"
```

**Expected**: `/` has > 2GB free

---

### Monitor Bot in Real-Time (detailed)

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>

# Watch live with colors
sudo journalctl -u homeguard-trading -f --output=cat

# Or with timestamps
sudo journalctl -u homeguard-trading -f -o short-precise

# Filter for specific events
sudo journalctl -u homeguard-trading -f | grep "Market:"
sudo journalctl -u homeguard-trading -f | grep "ORDER"
```

---

## Lambda Scheduler Health Checks

### Check Start Lambda Logs

```bash
aws logs tail /aws/lambda/homeguard-start-instance --since 24h --format short
```

**Expected**: Successful starts at 9:00 AM ET Monday-Friday

---

### Check Stop Lambda Logs

```bash
aws logs tail /aws/lambda/homeguard-stop-instance --since 24h --format short
```

**Expected**: Successful stops at 4:30 PM ET Monday-Friday

---

### Verify EventBridge Rules are Enabled

```bash
# Check start rule
aws events describe-rule --name homeguard-start-instance --query 'State'

# Check stop rule
aws events describe-rule --name homeguard-stop-instance --query 'State'
```

**Expected**: Both should return `"ENABLED"`

---

## Git Repository Health

### Check if Code is Up-to-Date

```bash
# On EC2 instance
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "cd ~/Homeguard && git log -1 --oneline"

# Compare with local
git log -1 --oneline
```

**If different**: EC2 is behind, run `git pull` on EC2

---

### Update Code on EC2

```bash
# SSH to instance
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>

# Pull latest code
cd ~/Homeguard
git pull

# Reinstall dependencies if needed
source venv/bin/activate
pip install -r requirements.txt

# Restart bot
sudo systemctl restart homeguard-trading

# Verify
sudo systemctl status homeguard-trading
exit
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| **SSH to instance** | `ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>` |
| **Check bot status** | `scripts/ec2/check_bot.sh` |
| **View live logs** | `scripts/ec2/view_logs.sh` |
| **Restart bot** | `scripts/ec2/restart_bot.sh` |
| **Stop bot** | `sudo systemctl stop homeguard-trading` |
| **Start bot** | `sudo systemctl start homeguard-trading` |
| **Check errors** | `sudo journalctl -u homeguard-trading -p err -n 20` |
| **Check instance state** | `aws ec2 describe-instances --instance-ids <YOUR_INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name'` |
| **Start instance** | `aws ec2 start-instances --instance-ids <YOUR_INSTANCE_ID>` |
| **Stop instance** | `aws ec2 stop-instances --instance-ids <YOUR_INSTANCE_ID>` |

---

## Health Status Indicators

### ✅ Healthy

- Bot status: `Active: active (running)`
- Logs updating every 15 seconds
- No errors in recent logs
- Memory usage < 500M
- Instance state matches schedule (running during market hours, stopped after)
- Lambda functions executing on schedule

### ⚠️ Warning

- Memory usage 500-900M (approaching limit)
- Occasional API timeouts (retry attempts visible)
- Logs updating but slower than 15 seconds
- Instance state transitions taking longer than expected

### ❌ Critical

- Bot status: `failed` or `inactive`
- No log updates for > 5 minutes
- Memory usage > 900M (near 1GB limit)
- Repeated API authentication errors
- Instance failed to start on schedule
- Python exceptions or stack traces in logs

---

## Automated Health Check Script

Create this for daily automated checks:

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Homeguard Trading Bot Health Check ==="
echo "Date: $(date)"
echo ""

echo "1. Instance State:"
aws ec2 describe-instances --instance-ids <YOUR_INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name'
echo ""

echo "2. Bot Service Status:"
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo systemctl is-active homeguard-trading"
echo ""

echo "3. Recent Errors (last hour):"
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading -p err --since '1 hour ago' --no-pager | wc -l"
echo ""

echo "4. Memory Usage:"
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo systemctl status homeguard-trading --no-pager | grep Memory"
echo ""

echo "5. Last Activity:"
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> "sudo journalctl -u homeguard-trading -n 1 --no-pager"
echo ""

echo "=== Health Check Complete ==="
```

**Usage**:
```bash
chmod +x daily_health_check.sh
./daily_health_check.sh
```

---

## Emergency Contacts & Resources

- **AWS Console**: https://console.aws.amazon.com/ec2
- **CloudWatch Logs**: https://console.aws.amazon.com/cloudwatch
- **Instance ID**: `<YOUR_INSTANCE_ID>` (see `.env`)
- **Region**: `us-east-1` (N. Virginia)
- **Alpaca Dashboard**: https://app.alpaca.markets/paper/dashboard/overview

---

**Last Updated**: 2025-12-06
**Instance IP**: `<YOUR_EC2_IP>` (see `.env`)
**Bot Version**: OMR Live Adapter (continuous mode)
