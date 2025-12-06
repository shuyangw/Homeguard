# Homeguard Trading Bot - Infrastructure Overview

**Deployment Date**: November 15, 2025
**Region**: us-east-1 (N. Virginia)
**Total Resources**: 16
**Monthly Cost**: ~$7.00

> **Note**: This document uses placeholders for sensitive values. Replace with your actual values from `.env` or `terraform output`.

---

## Quick Summary

| Component | Status | Details |
|-----------|--------|---------|
| **EC2 Instance** | ✓ Running | `<YOUR_INSTANCE_ID>` (t4g.small) |
| **Public IP** | ✓ Active | `<YOUR_EC2_IP>` (Elastic IP) |
| **Scheduled Start/Stop** | ✓ Enabled | 9:00 AM - 4:30 PM ET (Mon-Fri) |
| **Security** | ✓ Active | SSH from `<YOUR_IP_CIDR>` only |
| **Trading Bot** | ✓ Running | systemd service active |
| **Management Scripts** | ✓ Available | 10 scripts in `scripts/ec2/` |
| **Health Monitoring** | ✓ Configured | Automated 6-point health check |

---

## Infrastructure Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS Account (us-east-1)                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      COMPUTE & NETWORKING                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Security Group: homeguard-trading-bot-sg            │          │
│  │  <YOUR_SECURITY_GROUP_ID>                            │          │
│  ├──────────────────────────────────────────────────────┤          │
│  │  Inbound:  SSH (22) from <YOUR_IP_CIDR>             │          │
│  │  Outbound: ALL traffic to 0.0.0.0/0                 │          │
│  └──────────────────────────────────────────────────────┘          │
│                            │                                        │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  EC2 Instance: homeguard-trading-bot                 │          │
│  │  <YOUR_INSTANCE_ID>                                  │          │
│  ├──────────────────────────────────────────────────────┤          │
│  │  Type:     t4g.small (ARM64)                         │          │
│  │  AMI:      Amazon Linux 2023                         │          │
│  │  State:    running                                   │          │
│  │  Key Pair: homeguard-trading                         │          │
│  ├──────────────────────────────────────────────────────┤          │
│  │  Attached Storage:                                   │          │
│  │  ├─ 8 GB gp3 EBS (encrypted)                        │          │
│  │  └─ Delete on termination: false                    │          │
│  ├──────────────────────────────────────────────────────┤          │
│  │  Running Services:                                   │          │
│  │  └─ homeguard-trading.service (systemd)             │          │
│  │     └─ Python trading bot (OMR strategy)            │          │
│  └──────────────────────────────────────────────────────┘          │
│                            │                                        │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Elastic IP: homeguard-trading-bot-eip               │          │
│  │  <YOUR_EC2_IP>                                       │          │
│  │  (Static IP - persists when instance stopped)       │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    SERVERLESS SCHEDULING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  IAM Role: homeguard-ec2-scheduler-role              │          │
│  ├──────────────────────────────────────────────────────┤          │
│  │  Permissions:                                        │          │
│  │  ├─ ec2:StartInstances                              │          │
│  │  ├─ ec2:StopInstances                               │          │
│  │  ├─ ec2:DescribeInstances                           │          │
│  │  └─ logs:* (CloudWatch Logs)                        │          │
│  └──────────────────────────────────────────────────────┘          │
│                            │                                        │
│          ┌─────────────────┴──────────────────┐                    │
│          ▼                                    ▼                    │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │  Lambda Function │              │  Lambda Function │            │
│  │  START Instance  │              │  STOP Instance   │            │
│  ├──────────────────┤              ├──────────────────┤            │
│  │  Name:           │              │  Name:           │            │
│  │  homeguard-      │              │  homeguard-      │            │
│  │  start-instance  │              │  stop-instance   │            │
│  ├──────────────────┤              ├──────────────────┤            │
│  │  Runtime:        │              │  Runtime:        │            │
│  │  Python 3.11     │              │  Python 3.11     │            │
│  ├──────────────────┤              ├──────────────────┤            │
│  │  Triggered by:   │              │  Triggered by:   │            │
│  │  EventBridge     │              │  EventBridge     │            │
│  │  9:00 AM ET      │              │  4:30 PM ET      │            │
│  │  (Mon-Fri)       │              │  (Mon-Fri)       │            │
│  └──────────────────┘              └──────────────────┘            │
│          ▲                                    ▲                    │
│          │                                    │                    │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │  EventBridge     │              │  EventBridge     │            │
│  │  Rule (START)    │              │  Rule (STOP)     │            │
│  ├──────────────────┤              ├──────────────────┤            │
│  │  Schedule:       │              │  Schedule:       │            │
│  │  cron(0 14 ?     │              │  cron(30 21 ?    │            │
│  │  * MON-FRI *)    │              │  * MON-FRI *)    │            │
│  │                  │              │                  │            │
│  │  14:00 UTC =     │              │  21:30 UTC =     │            │
│  │  9:00 AM ET      │              │  4:30 PM ET      │            │
│  └──────────────────┘              └──────────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      MONITORING & LOGGING                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  CloudWatch Log Group                                │          │
│  │  /aws/lambda/homeguard-start-instance                │          │
│  │  Retention: 90 days                                  │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  CloudWatch Log Group                                │          │
│  │  /aws/lambda/homeguard-stop-instance                 │          │
│  │  Retention: 90 days                                  │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Local Logs (on EC2 instance)                        │          │
│  │  ~/logs/trading_YYYYMMDD.log                         │          │
│  │  (Flushed to disk at 4:00 PM ET daily)              │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL CONNECTIONS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EC2 Instance connects to:                                         │
│  ├─ Alpaca API (paper.api.alpaca.markets) - Trading                │
│  ├─ Yahoo Finance - Market data downloads                          │
│  ├─ GitHub - Code repository updates                               │
│  └─ AWS services - CloudWatch, Systems Manager                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Resource Breakdown by Category

### Compute (2 resources)
1. **EC2 Instance** (`aws_instance.homeguard_trading`)
   - ID: `<YOUR_INSTANCE_ID>`
   - Type: t4g.small
   - State: running
   - Cost: $2.65/month (157 hrs/month scheduled)

2. **EBS Volume** (attached to instance)
   - Size: 8 GB
   - Type: gp3 (encrypted)
   - Cost: $0.64/month

### Networking (2 resources)
3. **Security Group** (`aws_security_group.homeguard_trading`)
   - ID: `<YOUR_SECURITY_GROUP_ID>`
   - Rules: SSH from `<YOUR_IP_CIDR>`

4. **Elastic IP** (`aws_eip.homeguard_trading`)
   - IP: `<YOUR_EC2_IP>`
   - Cost: $3.60/month (when stopped)

### Serverless Functions (2 resources)
5. **Start Instance Lambda** (`aws_lambda_function.start_instance`)
   - Function: homeguard-start-instance
   - Runtime: Python 3.11
   - Trigger: 9:00 AM ET (Mon-Fri)

6. **Stop Instance Lambda** (`aws_lambda_function.stop_instance`)
   - Function: homeguard-stop-instance
   - Runtime: Python 3.11
   - Trigger: 4:30 PM ET (Mon-Fri)

### Event Scheduling (4 resources)
7. **Start Schedule** (`aws_cloudwatch_event_rule.start_instance`)
   - Cron: 0 14 ? * MON-FRI *

8. **Stop Schedule** (`aws_cloudwatch_event_rule.stop_instance`)
   - Cron: 30 21 ? * MON-FRI *

9. **Start Target** (`aws_cloudwatch_event_target.start_instance`)
   - Links rule to Lambda

10. **Stop Target** (`aws_cloudwatch_event_target.stop_instance`)
    - Links rule to Lambda

### IAM Permissions (4 resources)
11. **IAM Role** (`aws_iam_role.ec2_scheduler`)
    - Role: homeguard-ec2-scheduler-role

12. **IAM Policy** (`aws_iam_role_policy.ec2_scheduler_policy`)
    - Permissions: EC2 start/stop, CloudWatch logs

13. **Start Lambda Permission** (`aws_lambda_permission.allow_eventbridge_start`)
    - Allows EventBridge to invoke start Lambda

14. **Stop Lambda Permission** (`aws_lambda_permission.allow_eventbridge_stop`)
    - Allows EventBridge to invoke stop Lambda

### Logging (2 resources)
15. **Start Lambda Logs** (`aws_cloudwatch_log_group.start_instance_logs`)
    - Path: /aws/lambda/homeguard-start-instance
    - Retention: 90 days

16. **Stop Lambda Logs** (`aws_cloudwatch_log_group.stop_instance_logs`)
    - Path: /aws/lambda/homeguard-stop-instance
    - Retention: 90 days

---

## Daily Operation Flow

### Monday - Friday

**9:00 AM ET**:
1. EventBridge triggers start Lambda
2. Lambda checks instance state
3. If stopped → starts instance
4. Instance boots (~30 seconds)
5. Systemd auto-starts trading bot
6. Bot begins market monitoring

**9:30 AM ET**:
- Market opens
- Bot starts executing OMR strategy
- Places trades via Alpaca API

**4:00 PM ET**:
- Market closes
- Bot stops trading
- Logs flush to disk
- Bot continues monitoring

**4:30 PM ET**:
1. EventBridge triggers stop Lambda
2. Lambda checks instance state
3. If running → stops instance
4. Instance shuts down gracefully
5. Elastic IP preserved
6. All data saved to EBS

### Saturday - Sunday
- Instance remains stopped
- No scheduled starts
- No trading activity
- Minimal costs (EBS + Elastic IP only)

---

## Cost Breakdown

| Component | Hours/Month | Rate | Monthly Cost |
|-----------|-------------|------|--------------|
| EC2 t4g.small | 157.5 | $0.0168/hr | $2.65 |
| EBS 8 GB gp3 | 730 | $0.08/GB | $0.64 |
| Elastic IP (stopped) | ~572.5 | $0.005/hr | $3.60 |
| Lambda invocations | 42 | Free tier | $0.01 |
| CloudWatch Logs | ~10 MB | $0.50/GB | $0.01 |
| Data transfer | ~100 MB | $0.09/GB | $0.10 |
| **TOTAL** | | | **~$7.00** |

---

## Connection Information

**SSH Access**:
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>
```

**Public IP**: `<YOUR_EC2_IP>` (static - see `.env` for actual value)
**Public DNS**: `<YOUR_PUBLIC_DNS>` (available from AWS Console or `terraform output`)
**Instance ID**: `<YOUR_INSTANCE_ID>` (see `.env` for actual value)
**Security Group**: `<YOUR_SECURITY_GROUP_ID>` (see `terraform output`)

---

## Quick Commands

### Check Infrastructure Status
```bash
# List all resources
terraform state list

# Show all outputs
terraform output

# Get specific output
terraform output instance_public_ip
terraform output instance_state
```

### Monitor Instance
```bash
# Check if instance is running (use EC2_INSTANCE_ID from .env)
aws ec2 describe-instances --instance-ids <YOUR_INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name'

# View Lambda logs
aws logs tail /aws/lambda/homeguard-start-instance --follow
aws logs tail /aws/lambda/homeguard-stop-instance --follow
```

### Manual Control
```bash
# Manually start instance (use EC2_INSTANCE_ID from .env)
aws ec2 start-instances --instance-ids <YOUR_INSTANCE_ID>

# Manually stop instance
aws ec2 stop-instances --instance-ids <YOUR_INSTANCE_ID>

# SSH to instance (use EC2_IP from .env)
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP>

# Check bot status
sudo systemctl status homeguard-trading

# View trading logs
tail -f ~/logs/trading_$(date +%Y%m%d).log
```

---

## Management Tools

### SSH Quick-Access Scripts

Pre-configured scripts for easy instance management (Windows & Linux/Mac):

**Location**: `scripts/ec2/`

| Script | Purpose | Usage |
|--------|---------|-------|
| `connect.bat` / `connect.sh` | SSH into instance | Double-click or run directly |
| `check_bot.bat` / `check_bot.sh` | Check bot status + recent activity | Shows systemd status + last 10 log lines |
| `view_logs.bat` / `view_logs.sh` | Stream live bot logs | Real-time log monitoring (Ctrl+C to stop) |
| `restart_bot.bat` / `restart_bot.sh` | Restart trading bot service | Restarts systemd service and shows status |
| `daily_health_check.bat` / `daily_health_check.sh` | Automated 6-point health check | Instance state, bot status, errors, resources |
| `view_logs_plain.bat` | View logs without ANSI colors | For Windows CMD compatibility |

**Windows Quick Start**:
```bash
# From repository root
scripts\ec2\check_bot.bat
scripts\ec2\view_logs.bat
scripts\ec2\daily_health_check.bat
```

**Linux/Mac Quick Start**:
```bash
# From repository root
scripts/ec2/check_bot.sh
scripts/ec2/view_logs.sh
scripts/ec2/daily_health_check.sh
```

### Health Monitoring

**Comprehensive Health Check Cheatsheet**: See [`HEALTH_CHECK_CHEATSHEET.md`](HEALTH_CHECK_CHEATSHEET.md) for:
- Daily health check routine (morning, during market, after close)
- Common issues and quick fixes
- Advanced monitoring commands
- Lambda scheduler health checks
- Git repository sync verification

**6-Point Daily Health Check** (automated script):
1. Instance State (running/stopped)
2. Bot Service Status (active/failed)
3. Recent Errors (last hour)
4. Resource Usage (memory/CPU)
5. Last Activity (recent logs)
6. Market Status (open/closed)

### Bot Service Management

The trading bot runs as a systemd service with auto-restart capabilities:

```bash
# Service status
sudo systemctl status homeguard-trading

# Start service
sudo systemctl start homeguard-trading

# Stop service
sudo systemctl stop homeguard-trading

# Restart service
sudo systemctl restart homeguard-trading

# View service logs (live)
sudo journalctl -u homeguard-trading -f

# View service logs (last 50 lines)
sudo journalctl -u homeguard-trading -n 50

# View errors only
sudo journalctl -u homeguard-trading -p err
```

**Service Configuration**:
- **File**: `/etc/systemd/system/homeguard-trading.service`
- **Auto-restart**: Enabled (10-second delay between restarts)
- **Resource Limits**: 1GB RAM max, 150% CPU quota
- **Logging**: systemd journal + file logs
- **User**: ec2-user (non-root)

### Discord Bot Service (Optional)

The Discord monitoring bot runs as a separate systemd service for observability:

```bash
# Service status
sudo systemctl status homeguard-discord

# Start/stop/restart
sudo systemctl start homeguard-discord
sudo systemctl stop homeguard-discord
sudo systemctl restart homeguard-discord

# View logs
sudo journalctl -u homeguard-discord -f
```

**Service Configuration**:
- **File**: `/etc/systemd/system/homeguard-discord.service`
- **Purpose**: Read-only observability via Discord
- **Auto-restart**: Enabled (10-second delay)
- **Security**: Read-only filesystem access, no privilege escalation
- **Dependencies**: discord.py, anthropic

**Important**: The Discord bot is fully isolated from the trading bot. Discord bot failures have zero impact on trading operations.

**Management Scripts**:
- `scripts/ec2/discord_bot_status.bat` - Check status
- `scripts/ec2/discord_bot_restart.bat` - Restart service
- `scripts/ec2/discord_bot_logs.bat` - View logs

### Code Updates

To update the bot code on EC2:

```bash
# Method 1: Using SSH script
scripts\ec2\connect.bat   # or connect.sh

# Then on the instance:
cd ~/Homeguard
git pull
sudo systemctl restart homeguard-trading

# Method 2: Remote one-liner (use EC2_IP from .env)
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<YOUR_EC2_IP> \
  "cd ~/Homeguard && git pull && sudo systemctl restart homeguard-trading"
```

### Log Locations

**On EC2 Instance**:
- **Systemd logs**: `sudo journalctl -u homeguard-trading`
- **File logs**: `~/logs/live_trading/paper/trading_YYYYMMDD.log`
- **Service file**: `/etc/systemd/system/homeguard-trading.service`
- **Environment**: `~/Homeguard/.env` (credentials)

**On AWS**:
- **Lambda start logs**: `/aws/lambda/homeguard-start-instance`
- **Lambda stop logs**: `/aws/lambda/homeguard-stop-instance`

---

## Security Summary

**Network Security**:
- ✓ SSH restricted to single IP (`<YOUR_IP_CIDR>` - your home/office IP)
- ✓ No inbound ports except SSH (22)
- ✓ Outbound traffic allowed (for API calls)

**Data Security**:
- ✓ EBS volume encrypted at rest
- ✓ IMDSv2 required (metadata service protection)
- ✓ Alpaca credentials in environment variables (not code)
- ✓ Key pair required for SSH access

**IAM Security**:
- ✓ Lambda has minimal permissions (EC2 start/stop only)
- ✓ No root credentials stored
- ✓ Service-specific roles

---

## Architecture Highlights

**Automated**: Fully hands-off operation during trading week
**Cost-Optimized**: Runs only during market hours (46% savings)
**Resilient**: Auto-restart on failure, data persists across reboots
**Monitorable**: CloudWatch logs, systemd logs, trading logs
**Secure**: Minimal attack surface, encrypted storage, restricted access

---

**Last Updated**: November 15, 2025
**Managed by**: Terraform
**Configuration**: `terraform/terraform.tfvars`
