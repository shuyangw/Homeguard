# Discord Bot Setup Guide

**Date**: 2025-11-27
**Purpose**: Setup guide for the Homeguard Discord monitoring bot

---

## Overview

The Discord bot provides read-only observability for the Homeguard trading system through natural language queries powered by Claude. It runs as an optional, fully isolated addon that has zero impact on trading operations.

**Key Features**:
- **Slash Commands**: Type `/` to see all available commands with autocomplete
- Natural language queries: `/ask Why didn't the bot trade today?`
- Read-only access to logs, status, and configurations
- Channel-restricted for security
- Automatic secret masking in outputs

---

## Prerequisites

1. A Discord account
2. A Discord server you can add bots to
3. An Anthropic API key (from https://console.anthropic.com)
4. Access to the EC2 instance running Homeguard

---

## Step 1: Create Discord Application

1. Go to https://discord.com/developers/applications
2. Click **"New Application"**
3. Name: `Homeguard Trading Monitor` (or your preference)
4. Accept Terms of Service
5. Click **"Create"**
6. Note the **Application ID** (needed for invite URL)

---

## Step 2: Configure Bot User

1. Click **"Bot"** in left sidebar
2. Click **"Add Bot"** → **"Yes, do it!"**
3. Configure settings:
   - **Username**: `Homeguard Monitor`
   - **Public Bot**: OFF
   - **Require OAuth2 Code Grant**: OFF

4. **Get Bot Token**:
   - Click **"Reset Token"** → **"Yes, do it!"**
   - **COPY TOKEN IMMEDIATELY** (you won't see it again)
   - Save securely for later

---

## Step 3: Enable Message Content Intent

Under **"Privileged Gateway Intents"**, enable:

| Intent | Required | Purpose |
|--------|----------|---------|
| **MESSAGE CONTENT INTENT** | YES | Block DM spam |
| PRESENCE INTENT | No | Not needed |
| SERVER MEMBERS INTENT | No | Not needed |

> Note: Slash commands don't require MESSAGE CONTENT INTENT, but we use it to block DMs.
> Intent requires verification if bot joins 100+ servers. For private use, no verification needed.

---

## Step 4: Generate Invite URL

1. Go to **OAuth2 → URL Generator**
2. **Select Scopes**:
   - `bot`
   - `applications.commands`

3. **Select Bot Permissions**:

| Permission | Purpose |
|------------|---------|
| Read Messages/View Channels | See channels |
| Send Messages | Reply to queries |
| Send Messages in Threads | Thread support |
| Embed Links | Rich embeds |
| Attach Files | Send log excerpts |
| Read Message History | Context |
| Add Reactions | Acknowledge commands |

**Permissions Integer**: `377957124160`

4. Copy the generated URL

---

## Step 5: Add Bot to Server

1. Open the invite URL in your browser
2. Select your Discord server
3. Review permissions → Click **"Authorize"**
4. Complete CAPTCHA if prompted

---

## Step 6: Get Channel ID

1. In Discord: **User Settings → Advanced → Developer Mode: ON**
2. Right-click your monitoring channel
3. Click **"Copy Channel ID"**
4. Save this ID for configuration

---

## Step 7: Configure Environment

Add to your `.env` file (local or EC2):

```bash
# Discord Bot
DISCORD_TOKEN=your_bot_token_from_step_2
ANTHROPIC_API_KEY=sk-ant-api03-your_key_here
ALLOWED_CHANNELS=your_channel_id_from_step_6
ALLOWED_USERS=
```

---

## Step 8: Install Dependencies

```bash
# Activate environment
conda activate fintech  # or source venv/bin/activate

# Install Discord bot dependencies
pip install discord.py anthropic
```

---

## Step 9: Test Locally (Optional)

Before deploying to EC2, test locally:

```bash
cd /path/to/Homeguard
python -m src.discord_bot.main
```

In Discord, type `/` and you should see the bot's commands appear:
- `/help` - Should show available commands
- `/ping` - Check bot latency
- `/status` - Will fail (no trading logs locally, but confirms bot works)

> Note: Slash commands are automatically registered when the bot starts. It may take up to 1 minute for commands to appear in Discord.

Press Ctrl+C to stop.

---

## Step 10: Deploy to EC2

### SSH to Instance
```bash
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@100.30.95.146
```

### Update Repository
```bash
cd ~/Homeguard
git pull
```

### Install Dependencies
```bash
source venv/bin/activate
pip install discord.py anthropic
```

### Add Tokens to .env
```bash
# Append to existing .env
cat >> .env << 'EOF'
DISCORD_TOKEN=your_bot_token
ANTHROPIC_API_KEY=your_anthropic_key
ALLOWED_CHANNELS=your_channel_id
EOF
```

### Install Service
```bash
sudo cp scripts/systemd/homeguard-discord.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable homeguard-discord
sudo systemctl start homeguard-discord
```

### Verify
```bash
sudo systemctl status homeguard-discord
```

---

## Step 11: Test in Discord

Type `/` in your Discord channel to see all available commands with autocomplete.

Try these commands:

| Command | Expected Result |
|---------|-----------------|
| `/bothelp` | Shows available commands |
| `/status` | Trading bot health check |
| `/signals` | Today's trading signals |
| `/trades` | Today's executed trades |
| `/ask` | Ask Claude any question about the trading system |
| `/logs` | Show recent log entries |
| `/errors` | Search for errors in logs |
| `/ping` | Check bot latency |
| `/botstats` | Show bot statistics |

---

## Troubleshooting

### Bot Not Responding

1. Check service status:
   ```bash
   sudo systemctl status homeguard-discord
   ```

2. Check logs:
   ```bash
   sudo journalctl -u homeguard-discord -n 50
   ```

3. Verify tokens in `.env`:
   ```bash
   grep DISCORD_TOKEN ~/Homeguard/.env
   grep ANTHROPIC_API_KEY ~/Homeguard/.env
   ```

### Commands Not Appearing

- Slash commands may take up to 1 minute to register after bot starts
- Try restarting the bot: `sudo systemctl restart homeguard-discord`
- Check logs for "Successfully synced X slash commands"

### Permission Denied in Discord

- Error message says "not available in this channel" = wrong channel
- Check `ALLOWED_CHANNELS` in `.env`
- Ensure channel ID is correct (right-click → Copy ID)

### Claude Errors

- Check Anthropic API key is valid
- Check API quota at console.anthropic.com

---

## Management Commands

### Windows (from local machine)
```batch
scripts\ec2\discord_bot_status.bat
scripts\ec2\discord_bot_restart.bat
scripts\ec2\discord_bot_logs.bat
```

### Linux/Mac (from local machine)
```bash
scripts/ec2/local_discord_status.sh
scripts/ec2/local_discord_restart.sh
scripts/ec2/local_discord_logs.sh
```

### On EC2 Instance
```bash
sudo systemctl status homeguard-discord
sudo systemctl restart homeguard-discord
sudo journalctl -u homeguard-discord -f
```

---

## Security Notes

1. **Read-Only**: Bot cannot modify files or control services
2. **Channel-Restricted**: Only responds in configured channels
3. **Secret Masking**: API keys/passwords masked in Discord output
4. **Isolated**: Separate service, trading bot unaffected by failures
5. **No Sudo**: Bot runs as ec2-user with minimal privileges

---

## Cost Estimate

- **Claude API**: ~$0.05-0.10 per investigation
- **Expected usage**: 5-10 queries/day
- **Monthly cost**: ~$5-15/month
- **AWS cost**: $0 (runs on existing EC2)

---

## Quick Reference

| Item | Value |
|------|-------|
| Command Type | Slash Commands (`/`) |
| Bot Token Source | Discord Developer Portal → Bot → Reset Token |
| API Key Source | console.anthropic.com |
| Permissions Integer | 377957124160 |
| Service Name | homeguard-discord |
| Service File | /etc/systemd/system/homeguard-discord.service |
| Log Command | `journalctl -u homeguard-discord -f` |

## Available Commands

| Command | Description |
|---------|-------------|
| `/ask <question>` | Ask any question about the trading system |
| `/status` | Check if trading bot is running |
| `/signals` | Show today's trading signals |
| `/trades` | Show today's executed trades |
| `/logs [lines]` | Show recent log entries (default 50) |
| `/errors` | Search for errors in logs |
| `/bothelp` | Show available commands |
| `/ping` | Check bot latency |
| `/botstats` | Show bot statistics |
