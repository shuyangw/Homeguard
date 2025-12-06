# Troubleshooting Guide

Common issues and solutions for Homeguard trading bot.

**Last Updated**: 2025-12-06

---

## Quick Diagnostics

Before diving into specific issues, run this quick check:

```bash
# Windows
scripts\ec2\daily_health_check.bat

# Linux/Mac
scripts/ec2/daily_health_check.sh
```

This performs a 6-point validation: instance state, bot status, recent errors, memory usage, last activity, and market status.

---

## EC2 Instance Issues

### Instance Won't Start

**Symptoms**: Instance stuck in "stopped" state or fails to start.

**Solutions**:
1. Check AWS Console for instance state
2. Verify Lambda start function logs:
   ```bash
   aws logs tail /aws/lambda/homeguard-start-instance --since 1h
   ```
3. Manually start:
   ```bash
   aws ec2 start-instances --instance-ids <YOUR_INSTANCE_ID>
   ```
4. Check if EventBridge rule is enabled:
   ```bash
   aws events describe-rule --name homeguard-start-instance --query 'State'
   ```

### Instance Won't Stop

**Symptoms**: Instance running outside market hours, high costs.

**Solutions**:
1. Check Lambda stop function logs:
   ```bash
   aws logs tail /aws/lambda/homeguard-stop-instance --since 1h
   ```
2. Manually stop:
   ```bash
   aws ec2 stop-instances --instance-ids <YOUR_INSTANCE_ID>
   ```
3. Verify EventBridge rule is enabled

### Cannot SSH to Instance

**Symptoms**: Connection refused or timeout.

**Solutions**:
1. Verify instance is running
2. Check your IP is whitelisted in security group
3. Verify SSH key path in `.env`:
   ```bash
   ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>
   ```
4. If IP changed, update security group or use `scripts/ec2/get_my_ip.sh`

---

## Trading Bot Issues

### Bot Not Running

**Symptoms**: Service shows `inactive` or `failed`.

**Solutions**:
1. Check service status:
   ```bash
   sudo systemctl status homeguard-trading
   ```
2. View recent logs for errors:
   ```bash
   sudo journalctl -u homeguard-trading -n 50
   ```
3. Restart the bot:
   ```bash
   sudo systemctl restart homeguard-trading
   ```

### Bot Stuck (No Updates)

**Symptoms**: Logs not updating, last activity > 5 minutes old.

**Solutions**:
1. Force restart:
   ```bash
   sudo systemctl restart homeguard-trading
   ```
2. Check for Python errors:
   ```bash
   sudo journalctl -u homeguard-trading -p err -n 20
   ```
3. Check memory usage (may be OOM):
   ```bash
   free -h
   ```

### No Trades Being Placed

**Symptoms**: Bot running but no orders on Alpaca.

**Solutions**:
1. Verify market is open (trading hours: 9:30 AM - 4:00 PM ET)
2. Check for signals in logs:
   ```bash
   sudo journalctl -u homeguard-trading --since today | grep -E 'SIGNAL|ORDER'
   ```
3. Verify Alpaca credentials in `~/Homeguard/.env`
4. Check Alpaca API status: https://status.alpaca.markets

---

## API & Credentials Issues

### Alpaca API Errors

**Symptoms**: Authentication failures, 401/403 errors.

**Solutions**:
1. SSH to instance and verify `.env`:
   ```bash
   cat ~/Homeguard/.env | grep ALPACA
   ```
2. Verify keys match Alpaca dashboard (Paper Trading section)
3. Check for expired keys
4. Verify you're using Paper (not Live) credentials

### VIX Data Fetch Failures

**Symptoms**: "Failed to fetch VIX data" in logs.

**Solutions**:
1. Bot has built-in fallback chain (Yahoo Finance → FRED → cached)
2. Check if internet connectivity is working
3. VIX failures are non-fatal; bot continues with cached/default values

---

## Memory & Resource Issues

### Out of Memory (OOM)

**Symptoms**: Bot killed, memory > 900MB.

**Solutions**:
1. Restart bot to clear memory:
   ```bash
   sudo systemctl restart homeguard-trading
   ```
2. Check current usage:
   ```bash
   free -h
   ```
3. If recurring, consider upgrading instance type

### Disk Full

**Symptoms**: Write failures, log rotation issues.

**Solutions**:
1. Check disk space:
   ```bash
   df -h
   ```
2. Clean old logs:
   ```bash
   find ~/logs -name "*.log" -mtime +30 -delete
   ```

---

## Code & Git Issues

### Code Out of Date

**Symptoms**: Missing features, bug not fixed.

**Solutions**:
1. SSH to instance
2. Update code:
   ```bash
   cd ~/Homeguard
   git pull
   sudo systemctl restart homeguard-trading
   ```

### Import Errors

**Symptoms**: ModuleNotFoundError in logs.

**Solutions**:
1. SSH to instance
2. Reinstall dependencies:
   ```bash
   cd ~/Homeguard
   source venv/bin/activate
   pip install -r requirements.txt
   sudo systemctl restart homeguard-trading
   ```

---

## Local Development Issues

### Backtest Fails to Run

**Symptoms**: ImportError, path issues.

**Solutions**:
1. Activate conda environment:
   ```bash
   conda activate fintech
   ```
2. Run from project root
3. Check data directory exists (see `settings.ini`)

### GUI Won't Start

**Symptoms**: Tkinter errors, display issues.

**Solutions**:
1. Ensure running on local machine (not SSH)
2. Install Tkinter if missing:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-tk

   # macOS
   brew install python-tk
   ```

---

## Health Check Reference

### Healthy Indicators
- Bot status: `Active: active (running)`
- Logs updating every 15 seconds
- Memory usage < 500MB
- No errors in recent logs

### Warning Indicators
- Memory usage 500-900MB
- Occasional API timeouts
- Logs updating slower than normal

### Critical Indicators
- Bot status: `failed` or `inactive`
- No log updates > 5 minutes
- Memory usage > 900MB
- Repeated authentication errors

---

## Getting Help

1. Check [Health Check Cheatsheet](../HEALTH_CHECK_CHEATSHEET.md) for monitoring commands
2. Review [Infrastructure Overview](../INFRASTRUCTURE_OVERVIEW.md) for architecture details
3. Check [SSH Scripts README](../../scripts/ec2/SSH_SCRIPTS_README.md) for management scripts
4. Open an issue on GitHub for bugs or feature requests
