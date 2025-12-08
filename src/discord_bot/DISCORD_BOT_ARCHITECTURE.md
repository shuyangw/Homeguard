# Discord Bot Architecture

**A read-only observability bot for monitoring the Homeguard trading system through natural language queries powered by Claude.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Provides read-only monitoring of the trading system via Discord
- Uses Claude AI to interpret queries and investigate issues
- Executes shell commands on EC2 to gather information
- Reports service status, logs, trades, and errors

### Key Features
- **Claude-Powered Investigation**: Natural language query interpretation
- **Read-Only Access**: No modifications, no service control
- **Rate Limiting**: Per-user and global rate limits
- **Security**: Allowed channels, roles, and command filtering
- **Multi-Strategy**: Monitors both OMR and MP services

### Use Cases
- Check trading bot status remotely
- Investigate errors from Discord
- Review today's trades and signals
- Monitor service health

---

## Architecture

```
src/discord_bot/
├── __init__.py              # Package marker
├── main.py                  # Bot entry point, Discord setup
├── investigator.py          # Claude-powered investigation logic
├── executor.py              # Shell command execution
├── security.py              # Channel/user/command filtering
├── formatters.py            # Discord message formatting
└── config.py                # Bot configuration loading
```

### Design Philosophy

1. **Read-Only**: Bot cannot modify system state
2. **Claude as Investigator**: AI interprets queries and runs commands
3. **Security First**: Multiple layers of access control
4. **Rate Limited**: Prevent abuse and API exhaustion
5. **Multi-Strategy Aware**: Monitors both OMR and MP services

---

## Key Components

### Main Bot (`main.py`)

**Purpose**: Discord bot entry point with slash commands.

**Key Functions**:
- `/investigate <query>`: Start Claude-powered investigation
- `/status`: Quick status check of trading services
- `/health`: Full health check

**Rate Limits**:
| Limit | Value |
|-------|-------|
| Max concurrent investigations | 5 |
| User cooldown | 10 seconds |
| User rate per minute | 5 |
| User rate per hour | 20 |
| Global rate per minute | 15 |

**Trading Services**:
```python
TRADING_SERVICES = ["homeguard-omr", "homeguard-mp"]
```

**Usage**:
```bash
# Start the bot
python -m src.discord_bot.main
```

**Environment Variables**:
- `DISCORD_TOKEN`: Discord bot token
- `ANTHROPIC_API_KEY`: Anthropic API key
- `ALLOWED_CHANNELS`: Comma-separated channel IDs (optional)

### Investigator (`investigator.py`)

**Purpose**: Claude-powered investigation using tool use API.

**How It Works**:
1. User sends query via `/investigate`
2. Query sent to Claude with system prompt
3. Claude requests shell commands via tool use
4. Commands executed via Executor
5. Results sent back to Claude
6. Claude summarizes findings

**System Prompt Highlights**:
- READ-ONLY access emphasized
- Strategy schedules documented (OMR: 3:50 PM, MP: Friday 3:55 PM)
- Log structure explained
- Key commands provided
- Times in ET (UTC-5/UTC-4)

**Tool Definition**:
```python
TOOLS = [{
    "name": "run_shell_command",
    "description": "Execute a read-only shell command...",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "default": 30}
        }
    }
}]
```

**Usage**:
```python
from src.discord_bot.investigator import TradingInvestigator

investigator = TradingInvestigator(config)
result = await investigator.investigate(
    query="What errors occurred today?",
    user_id="12345"
)
```

### Executor (`executor.py`)

**Purpose**: Safe shell command execution with security filtering.

**Allowed Commands**:
- `tail`, `cat`, `head` - View files
- `grep`, `awk` - Search and filter
- `ps`, `top` - Process info
- `systemctl status` - Service status
- `journalctl` - View logs
- `ls`, `find` - List files
- `date`, `uptime` - System info

**Blocked Commands**:
- `sudo` - Elevated privileges
- `rm`, `mv`, `cp` - File modification
- `systemctl start/stop/restart` - Service control
- `kill`, `pkill` - Process termination
- `chmod`, `chown` - Permission changes
- `>`, `>>`, `|` (to destructive commands)

**Usage**:
```python
from src.discord_bot.executor import CommandExecutor

executor = CommandExecutor()
result = await executor.execute("tail -100 /path/to/log.txt")
# Returns: {"stdout": "...", "stderr": "...", "exit_code": 0}
```

### Security (`security.py`)

**Purpose**: Multi-layer access control.

**Checks**:
1. **Channel Allowed**: Only configured channels
2. **User Allowed**: Role-based access (Bot Admin role)
3. **Command Allowed**: Whitelist of safe commands

**Required Role**:
```python
REQUIRED_ROLE_ID = 1446546349163286570  # "Bot Admin" role
```

**Usage**:
```python
from src.discord_bot.security import is_channel_allowed, is_user_allowed

if not is_channel_allowed(channel_id):
    return "Access denied"

if not is_user_allowed(user):
    return "You don't have permission"
```

### Formatters (`formatters.py`)

**Purpose**: Format investigation results for Discord.

**Features**:
- Markdown formatting
- Code block wrapping
- Length truncation (Discord 2000 char limit)
- Error message formatting

**Usage**:
```python
from src.discord_bot.formatters import format_investigation_result, format_error

message = format_investigation_result(result)
error_msg = format_error("Connection failed", exception)
```

### Config (`config.py`)

**Purpose**: Load bot configuration.

**Configuration**:
```python
@dataclass
class DiscordBotConfig:
    discord_token: str
    anthropic_api_key: str
    allowed_channels: List[int]
    allowed_users: List[int]
    max_command_timeout: int = 120
    max_iterations: int = 10
```

**Usage**:
```python
from src.discord_bot.config import load_config

config = load_config()  # Loads from environment
```

---

## Data Flow

```
Discord User
        ↓
  Slash Command (/investigate "query")
        ↓
  Security Checks (channel, user, rate limit)
        ↓
  TradingInvestigator
        ↓
  Claude API (with tool use)
        ↓
  CommandExecutor (read-only shell)
        ↓
  Claude Summarizes Findings
        ↓
  Format for Discord
        ↓
  Response to User
```

---

## Public API

### Main Entry Point

```python
# Start the bot
python -m src.discord_bot.main
```

### Investigation

```python
from src.discord_bot.investigator import TradingInvestigator

investigator = TradingInvestigator(config)
result = await investigator.investigate(query, user_id)
```

---

## Configuration

### Environment Variables

```bash
# Required
DISCORD_TOKEN=your_discord_bot_token
ANTHROPIC_API_KEY=your_anthropic_key

# Optional
ALLOWED_CHANNELS=123456789,987654321
```

### Log Directories

Monitored directories on EC2:
```
~/logs/live_trading/paper/YYYYMMDD/
├── *_session.json   # Signals, orders, checks
├── *_trades.csv     # Executed trades
└── *.log            # Buffered output
```

---

## Dependencies

### Internal (src/ modules)
- `src.settings` - Log directory paths

### External (pip packages)
- `discord.py` - Discord API
- `anthropic` - Claude API
- `python-dotenv` - Environment loading
- `aiohttp` - Async HTTP

---

## Security Considerations

### Command Filtering

The executor uses a whitelist approach:
- Only explicitly allowed commands can run
- No shell metacharacters that could chain commands
- No file modification or service control

### Rate Limiting

Prevents abuse:
- Per-user cooldowns
- Hourly/minute limits
- Global rate limiting
- Maximum concurrent investigations

### Access Control

- Channel-based restrictions
- Role-based user access
- Bot Admin role required

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `Rate limited` | Too many requests | Wait for cooldown |
| `Access denied` | Wrong channel/role | Use allowed channel |
| `Command blocked` | Unsafe command | Only read-only allowed |
| `Investigation timeout` | Claude taking too long | Simplify query |

---

## Testing

### Test Location
- `tests/discord_bot/` - Unit tests

### Running Tests
```bash
pytest tests/discord_bot/ -v
```

### Manual Testing

```bash
# Set environment
export DISCORD_TOKEN=your_token
export ANTHROPIC_API_KEY=your_key

# Run bot
python -m src.discord_bot.main
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Infrastructure Overview](../../docs/INFRASTRUCTURE_OVERVIEW.md)
- [Live Trading System](../trading/LIVE_TRADING_SYSTEM.md)

---

## Changelog

- **2025-12-08**: Initial documentation created
- **2025-12-06**: Multi-strategy support (OMR + MP)
- **2025-11-XX**: Claude tool use integration
- **2025-10-XX**: Initial Discord bot
