# SSH Quick Access Scripts

Quick scripts to manage your Homeguard Trading Bot on AWS EC2.

**Location**: `scripts/ec2/`
**Instance IP**: 100.30.95.146
**Key File**: `~/.ssh/homeguard-trading.pem`

---

## Available Scripts

### Windows (.bat files)

| Script | Purpose | Usage |
|--------|---------|-------|
| **connect.bat** | SSH into instance | Double-click or `connect.bat` |
| **check_bot.bat** | Check bot status | Double-click or `check_bot.bat` |
| **view_logs.bat** | View live logs | Double-click or `view_logs.bat` |
| **restart_bot.bat** | Restart bot service | Double-click or `restart_bot.bat` |
| **daily_health_check.bat** | Automated health check | Double-click or `daily_health_check.bat` |

### Linux/Mac (.sh files)

| Script | Purpose | Usage |
|--------|---------|-------|
| **connect.sh** | SSH into instance | `./connect.sh` |
| **check_bot.sh** | Check bot status | `./check_bot.sh` |
| **view_logs.sh** | View live logs | `./view_logs.sh` |
| **restart_bot.sh** | Restart bot service | `./restart_bot.sh` |
| **daily_health_check.sh** | Automated health check | `./daily_health_check.sh` |

---

## Quick Start

### Windows

```powershell
# Navigate to scripts directory (from project root)
cd scripts\ec2

# SSH into instance
connect.bat

# Check if bot is running
check_bot.bat

# View live logs (press Ctrl+C to stop)
view_logs.bat

# Restart bot
restart_bot.bat
```

### Linux/Mac

```bash
# Navigate to scripts directory (from project root)
cd scripts/ec2

# SSH into instance
./connect.sh

# Check if bot is running
./check_bot.sh

# View live logs (press Ctrl+C to stop)
./view_logs.sh

# Restart bot
./restart_bot.sh
```

### Or run directly from project root:

**Windows**:
```powershell
scripts\ec2\check_bot.bat
scripts\ec2\view_logs.bat
```

**Linux/Mac**:
```bash
scripts/ec2/check_bot.sh
scripts/ec2/view_logs.sh
```

---

## Script Details

### 1. connect.bat / connect.sh
**Opens SSH session to EC2 instance**

Once connected, you can:
```bash
# Check bot status
sudo systemctl status homeguard-trading

# View logs
sudo journalctl -u homeguard-trading -f

# Navigate to project
cd ~/Homeguard

# Check trading logs
ls -lh ~/logs/live_trading/paper/

# Exit SSH
exit
```

---

### 2. check_bot.bat / check_bot.sh
**Shows bot status and recent activity**

Output includes:
- Service status (running/stopped)
- Process ID (PID)
- Memory usage
- CPU time
- Recent log entries (last 10 lines)

Example output:
```
● homeguard-trading.service - Homeguard Trading Bot
   Active: active (running)
   Main PID: 27292
   Memory: 74.2M

Recent Activity:
[08:24:21] Market: CLOSED | Checks: 4 | Runs: 0 | Signals: 0
```

---

### 3. view_logs.bat / view_logs.sh
**Streams live bot logs to your terminal**

Shows:
- Market status checks
- Strategy execution
- Trade signals
- Order placements
- Errors and warnings

**Press Ctrl+C to stop**

Example output:
```
[08:23:36] Market: CLOSED | Checks: 1 | Runs: 0 | Signals: 0 | Orders: 0/0
[08:23:51] Market: CLOSED | Checks: 2 | Runs: 0 | Signals: 0 | Orders: 0/0
[08:24:06] Market: CLOSED | Checks: 3 | Runs: 0 | Signals: 0 | Orders: 0/0
```

---

### 4. restart_bot.bat / restart_bot.sh
**Restarts the trading bot service**

Use when:
- Bot appears stuck
- After updating configuration
- After code changes
- To reload environment variables

Safely stops and restarts the bot with a 3-second wait.

---

## Manual Commands Reference

If you prefer to run commands manually:

### SSH to Instance
```bash
# Windows
ssh -i %USERPROFILE%\.ssh\homeguard-trading.pem ec2-user@100.30.95.146

# Linux/Mac
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146
```

### Check Status
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo systemctl status homeguard-trading"
```

### View Logs
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -f"
```

### Restart Bot
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146 "sudo systemctl restart homeguard-trading"
```

---

## Additional Useful Commands

Once SSH'd into the instance:

```bash
# Check instance state
aws ec2 describe-instances --instance-ids i-02500fe2392631ff2 --query 'Reservations[0].Instances[0].State.Name'

# View saved trading logs
ls -lh ~/logs/live_trading/paper/
cat ~/logs/live_trading/paper/trading_$(date +%Y%m%d).log

# Check Python environment
source ~/Homeguard/venv/bin/activate
python --version
pip list | grep alpaca

# View .env file (credentials)
cat ~/Homeguard/.env

# Stop bot
sudo systemctl stop homeguard-trading

# Start bot
sudo systemctl start homeguard-trading

# Disable auto-start
sudo systemctl disable homeguard-trading

# Enable auto-start
sudo systemctl enable homeguard-trading

# View all systemd logs
sudo journalctl -u homeguard-trading --no-pager

# View errors only
sudo journalctl -u homeguard-trading -p err -n 50

# Follow logs and filter
sudo journalctl -u homeguard-trading -f | grep "TRADE"
```

---

## Troubleshooting

### Script doesn't work on Windows
- Make sure you're running from the project root directory
- Verify key file exists at `C:\Users\qwqw1\.ssh\homeguard-trading.pem`

### Script doesn't work on Linux/Mac
- Make scripts executable: `chmod +x *.sh`
- Verify key file exists at `~/.ssh/homeguard-trading.pem`
- Verify key permissions: `chmod 400 ~/.ssh/homeguard-trading.pem`

### "Permission denied" error
- Check key file permissions (should be 400)
- Verify you're using the correct key file path

### "Connection refused" error
- Instance may be stopped (scheduled stop at 4:30 PM ET)
- Check instance state in AWS Console or:
  ```bash
  aws ec2 describe-instances --instance-ids i-02500fe2392631ff2
  ```

### Bot is not running
- SSH to instance: `./connect.sh` or `connect.bat`
- Check status: `sudo systemctl status homeguard-trading`
- View errors: `sudo journalctl -u homeguard-trading -p err -n 50`
- Restart: `./restart_bot.sh` or `restart_bot.bat`

---

## Security Notes

⚠️ **Keep your SSH key secure!**
- Never commit `.pem` files to git
- Set proper permissions: `chmod 400 ~/.ssh/homeguard-trading.pem`
- Don't share your key file

⚠️ **IP Address Changes**
- If your IP changes, update `terraform.tfvars` and run `terraform apply`
- Current allowed IP: 73.68.21.247/32

---

## Next Steps

**Daily monitoring**:
- Run `check_bot.bat` (Windows) or `./check_bot.sh` (Linux/Mac) each morning
- Verify bot is trading during market hours

**Weekly review**:
- SSH to instance and download trading logs
- Review trade history and performance

**After code updates**:
- SSH to instance
- `cd ~/Homeguard && git pull`
- `source venv/bin/activate && pip install -r requirements.txt`
- `./restart_bot.sh` or `restart_bot.bat`

---

**Instance Details**:
- **IP**: 100.30.95.146
- **Instance ID**: i-02500fe2392631ff2
- **Region**: us-east-1
- **Schedule**: Runs Monday-Friday 9:00 AM - 4:30 PM ET
