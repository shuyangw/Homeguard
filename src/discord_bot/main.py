"""
Homeguard Discord Bot - Main entry point.

A read-only observability bot for monitoring the Homeguard trading system
through natural language queries powered by Claude.

Uses Discord Slash Commands for all interactions.

Usage:
    python -m src.discord_bot.main

Environment Variables Required:
    DISCORD_TOKEN - Discord bot token
    ANTHROPIC_API_KEY - Anthropic API key
    ALLOWED_CHANNELS - Comma-separated channel IDs (optional)
"""

import asyncio
import logging
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from .config import load_config, DiscordBotConfig
from .formatters import format_error, format_investigation_result
from .investigator import TradingInvestigator
from .security import is_channel_allowed, is_user_allowed
from ..settings import get_discord_bot_log_dir

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_BASE = 5  # seconds
LOG_DIR = get_discord_bot_log_dir()

# Rate limiting constants
MAX_CONCURRENT_INVESTIGATIONS = 5
USER_COOLDOWN_SECONDS = 10  # Per-user cooldown between commands
USER_RATE_LIMIT_PER_MINUTE = 5
USER_RATE_LIMIT_PER_HOUR = 20
GLOBAL_RATE_LIMIT_PER_MINUTE = 15

# Trading service name
TRADING_SERVICE_NAME = "homeguard-trading"


async def check_trading_process_running() -> tuple[bool, str]:
    """
    Check if the trading process/service is running.

    Supports:
        - Linux: Checks systemd service status, falls back to pgrep
        - macOS: Uses pgrep to find trading processes
        - Windows: Checks for Python processes (dev environment)

    Returns:
        (is_running: bool, status_message: str)
    """
    system = platform.system()

    if system == "Linux":
        # Check systemd service on Linux (EC2)
        try:
            result = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", TRADING_SERVICE_NAME,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            status = stdout.decode().strip()

            if status == "active":
                return True, f"Trading service `{TRADING_SERVICE_NAME}` is running"
            else:
                return False, f"Trading service `{TRADING_SERVICE_NAME}` is not running (status: {status})"
        except FileNotFoundError:
            # systemctl not available, try pgrep
            return await _check_trading_process_pgrep()
        except Exception as e:
            return False, f"Error checking trading service: {e}"

    elif system == "Darwin":
        # macOS - use pgrep to find trading processes
        return await _check_trading_process_pgrep()

    elif system == "Windows":
        # On Windows, check for Python process running trading module
        try:
            result = await asyncio.create_subprocess_exec(
                "tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            output = stdout.decode()

            # This is a basic check - on Windows dev environment
            # we can't easily distinguish which Python process is which
            if "python.exe" in output.lower():
                return True, "Python processes running (dev environment - trading status unknown)"
            else:
                return False, "No Python processes found"
        except Exception as e:
            return False, f"Error checking processes: {e}"

    else:
        return False, f"Unsupported platform: {system}"


async def _check_trading_process_pgrep() -> tuple[bool, str]:
    """
    Check for trading process using pgrep (Unix/macOS).

    Returns:
        (is_running: bool, status_message: str)
    """
    try:
        result = await asyncio.create_subprocess_exec(
            "pgrep", "-f", "homeguard.*trading|trading.*homeguard",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await result.communicate()

        if result.returncode == 0 and stdout.strip():
            pids = stdout.decode().strip().split('\n')
            return True, f"Trading process found (PID: {', '.join(pids)})"
        else:
            return False, "No trading process found"
    except FileNotFoundError:
        return False, "pgrep not available - cannot check trading process"
    except Exception as e:
        return False, f"Unable to check trading process: {e}"


class RateLimiter:
    """
    Token bucket rate limiter with per-user and global limits.

    Tracks request timestamps to enforce rate limits without external dependencies.
    """

    def __init__(self):
        # Per-user tracking: {user_id: [timestamps]}
        self._user_requests: dict[int, list[float]] = defaultdict(list)
        # Global tracking: [timestamps]
        self._global_requests: list[float] = []
        # Lock for thread-safe access
        self._lock = asyncio.Lock()

    def _clean_old_requests(self, requests: list[float], window_seconds: float) -> list[float]:
        """Remove requests older than the time window."""
        cutoff = time.time() - window_seconds
        return [ts for ts in requests if ts > cutoff]

    async def check_rate_limit(self, user_id: int) -> tuple[bool, str | None]:
        """
        Check if a request from user_id is allowed.

        Returns:
            (allowed: bool, error_message: str | None)
        """
        async with self._lock:
            now = time.time()

            # Clean old requests
            self._user_requests[user_id] = self._clean_old_requests(
                self._user_requests[user_id], 3600  # Keep 1 hour of history
            )
            self._global_requests = self._clean_old_requests(
                self._global_requests, 60  # Keep 1 minute of history
            )

            # Check global rate limit (per minute)
            if len(self._global_requests) >= GLOBAL_RATE_LIMIT_PER_MINUTE:
                return False, f"Global rate limit reached ({GLOBAL_RATE_LIMIT_PER_MINUTE}/min). Try again in a moment."

            # Check per-user rate limit (per minute)
            user_last_minute = [ts for ts in self._user_requests[user_id] if ts > now - 60]
            if len(user_last_minute) >= USER_RATE_LIMIT_PER_MINUTE:
                return False, f"Rate limit reached ({USER_RATE_LIMIT_PER_MINUTE}/min). Please wait before trying again."

            # Check per-user rate limit (per hour)
            user_last_hour = self._user_requests[user_id]
            if len(user_last_hour) >= USER_RATE_LIMIT_PER_HOUR:
                return False, f"Hourly limit reached ({USER_RATE_LIMIT_PER_HOUR}/hour). Please try again later."

            # Check per-user cooldown
            if user_last_minute:
                time_since_last = now - max(user_last_minute)
                if time_since_last < USER_COOLDOWN_SECONDS:
                    remaining = USER_COOLDOWN_SECONDS - time_since_last
                    return False, f"Please wait {remaining:.1f}s before your next request."

            return True, None

    async def record_request(self, user_id: int) -> None:
        """Record a successful request for rate limiting."""
        async with self._lock:
            now = time.time()
            self._user_requests[user_id].append(now)
            self._global_requests.append(now)

    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        now = time.time()
        return {
            "global_requests_last_minute": len([ts for ts in self._global_requests if ts > now - 60]),
            "unique_users_last_hour": len(self._user_requests),
        }


def setup_logging() -> logging.Logger:
    """
    Set up logging with both file and stdout handlers.

    Creates rotating log files in ~/logs/discord_bot/ with format:
    discord_bot_YYYYMMDD.log
    """
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("discord_bot")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # Log format
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler - rotating, max 10MB per file, keep 5 backups
    log_file = LOG_DIR / f"discord_bot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(log_format)
    logger.addHandler(stdout_handler)

    # Also configure discord.py logging
    discord_logger = logging.getLogger("discord")
    discord_logger.setLevel(logging.WARNING)
    discord_logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


# Initialize logging
logger = setup_logging()


def validate_discord_token(token: str) -> tuple[bool, str]:
    """
    Validate Discord token format.

    Discord tokens have a specific format:
    - Base64 encoded user ID
    - Timestamp
    - HMAC
    """
    if not token:
        return False, "Discord token is empty"

    # Basic format check - tokens have 3 parts separated by dots
    parts = token.split(".")
    if len(parts) != 3:
        return False, "Discord token format invalid (expected 3 parts separated by dots)"

    # First part should be base64 encoded
    try:
        import base64
        # Pad the base64 string if needed
        padded = parts[0] + "=" * (4 - len(parts[0]) % 4)
        base64.b64decode(padded)
    except Exception:
        return False, "Discord token format invalid (first part not valid base64)"

    return True, "Token format valid"


def validate_anthropic_key(key: str) -> tuple[bool, str]:
    """Validate Anthropic API key format."""
    if not key:
        return False, "Anthropic API key is empty"

    if not key.startswith("sk-ant-"):
        return False, "Anthropic API key should start with 'sk-ant-'"

    if len(key) < 40:
        return False, "Anthropic API key appears too short"

    return True, "API key format valid"


class TradingMonitorBot(commands.Bot):
    """Discord bot for monitoring the Homeguard trading system using slash commands."""

    def __init__(self, config: DiscordBotConfig):
        """Initialize the bot with configuration."""
        intents = discord.Intents.default()
        intents.message_content = True  # Still needed for DM blocking

        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self.investigator = TradingInvestigator(config)
        self._reconnect_attempts = 0
        self._is_ready = False

        # Rate limiting and concurrency control
        self.rate_limiter = RateLimiter()
        self.investigation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INVESTIGATIONS)
        self._active_investigations = 0

    async def setup_hook(self):
        """Called when bot is ready to set up. Registers slash commands."""
        logger.info("Bot setup hook called - preparing slash commands...")

        # Set up error handler for app commands
        self.tree.on_error = self.on_app_command_error

        # Note: Commands are added per-guild in sync_commands_to_guilds()
        # This avoids duplicates between global and guild commands

    async def on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        """Handle errors from slash commands."""
        logger.error(f"Slash command error: {error}", exc_info=True)
        try:
            if interaction.response.is_done():
                await interaction.followup.send(f"\u274c Error: {str(error)[:200]}", ephemeral=True)
            else:
                await interaction.response.send_message(f"\u274c Error: {str(error)[:200]}", ephemeral=True)
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

    async def sync_commands_to_guilds(self):
        """Sync slash commands to all guilds (instant propagation)."""
        # First, clear global commands to avoid duplicates
        logger.info("Clearing global slash commands...")
        try:
            self.tree.clear_commands(guild=None)
            await self.tree.sync()  # Sync empty global commands
            logger.info("Global commands cleared")
        except Exception as e:
            logger.error(f"Failed to clear global commands: {e}", exc_info=True)

        # Now sync to each guild
        logger.info("Syncing slash commands to guilds...")
        for guild in self.guilds:
            try:
                # Re-add commands for this guild
                self.tree.clear_commands(guild=guild)
                self.tree.add_command(ask_command, guild=guild)
                self.tree.add_command(status_command, guild=guild)
                self.tree.add_command(signals_command, guild=guild)
                self.tree.add_command(trades_command, guild=guild)
                self.tree.add_command(logs_command, guild=guild)
                self.tree.add_command(errors_command, guild=guild)
                self.tree.add_command(help_command, guild=guild)
                self.tree.add_command(ping_command, guild=guild)
                self.tree.add_command(botstats_command, guild=guild)

                synced = await self.tree.sync(guild=guild)
                logger.info(f"Synced {len(synced)} commands to guild: {guild.name} (ID: {guild.id})")
                for cmd in synced:
                    logger.info(f"  - /{cmd.name}: {cmd.description}")
            except Exception as e:
                logger.error(f"Failed to sync commands to guild {guild.name}: {e}", exc_info=True)

    async def on_ready(self):
        """Called when bot successfully connects to Discord."""
        self._is_ready = True
        self._reconnect_attempts = 0  # Reset on successful connection

        logger.info("=" * 60)
        logger.info("DISCORD BOT CONNECTED SUCCESSFULLY")
        logger.info(f"  Bot User: {self.user} (ID: {self.user.id})")
        logger.info(f"  Guilds: {len(self.guilds)}")
        for guild in self.guilds:
            logger.info(f"    - {guild.name} (ID: {guild.id})")
        logger.info(f"  Latency: {self.latency * 1000:.2f}ms")
        logger.info(f"  Command Type: Slash Commands (/)")

        if self.config.allowed_channel_ids:
            logger.info(f"  Restricted to channels: {self.config.allowed_channel_ids}")
        else:
            logger.info("  Channel restrictions: None (all channels allowed)")

        # Log rate limiting configuration
        logger.info("Rate Limiting:")
        logger.info(f"  Max concurrent investigations: {MAX_CONCURRENT_INVESTIGATIONS}")
        logger.info(f"  Per-user cooldown: {USER_COOLDOWN_SECONDS}s")
        logger.info(f"  Per-user limit: {USER_RATE_LIMIT_PER_MINUTE}/min, {USER_RATE_LIMIT_PER_HOUR}/hour")
        logger.info(f"  Global limit: {GLOBAL_RATE_LIMIT_PER_MINUTE}/min")
        logger.info("=" * 60)

        # Sync commands to guilds (instant propagation vs global which takes up to 1 hour)
        await self.sync_commands_to_guilds()

    async def on_connect(self):
        """Called when bot connects to Discord gateway."""
        logger.info("Connected to Discord gateway")

    async def on_disconnect(self):
        """Called when bot disconnects from Discord."""
        logger.warning("Disconnected from Discord gateway")
        self._is_ready = False

    async def on_resumed(self):
        """Called when bot resumes a session after disconnect."""
        logger.info("Session resumed after disconnect")
        self._is_ready = True

    async def on_error(self, event_method: str, *args, **kwargs):
        """Called when an error occurs in an event handler."""
        logger.error(f"Error in event {event_method}", exc_info=True)

    async def on_interaction(self, interaction: discord.Interaction):
        """Log all interactions for debugging."""
        logger.info(
            f"INTERACTION received: type={interaction.type.name}, "
            f"user={interaction.user}, channel={interaction.channel}, "
            f"data={interaction.data}"
        )
        # discord.py handles interactions automatically via the command tree

    async def on_message(self, message: discord.Message):
        """
        Handle messages - log all and block DMs.

        Slash commands are handled separately via interactions.
        """
        # Ignore messages from bots (including self)
        if message.author.bot:
            return

        # Log all messages for debugging
        logger.info(
            f"MESSAGE received: channel={message.channel} ({message.channel.id}), "
            f"author={message.author}, content={message.content[:100]}"
        )

        # Block DMs entirely - only respond in guild channels
        if message.guild is None:
            logger.warning(
                f"Blocked DM from {message.author} ({message.author.id}): "
                f"{message.content[:50]}..."
            )
            try:
                await message.channel.send(
                    "\u26d4 This bot only works in server channels using slash commands. "
                    "Type `/` to see available commands."
                )
            except discord.Forbidden:
                pass
            return


# Global bot instance (set during create_bot)
_bot: Optional[TradingMonitorBot] = None


def get_bot() -> TradingMonitorBot:
    """Get the current bot instance."""
    if _bot is None:
        raise RuntimeError("Bot not initialized")
    return _bot


async def check_permissions(interaction: discord.Interaction) -> tuple[bool, str | None]:
    """
    Check if interaction is allowed based on channel and user restrictions.

    Returns:
        (allowed: bool, error_message: str | None)
    """
    bot = get_bot()

    # Check channel restrictions
    if not is_channel_allowed(interaction.channel_id, bot.config.allowed_channel_ids):
        logger.warning(
            f"Blocked request from unauthorized channel {interaction.channel_id} "
            f"(user: {interaction.user}, guild: {interaction.guild})"
        )
        return False, "This command is not available in this channel."

    # Check user restrictions
    if not is_user_allowed(interaction.user.id, bot.config.allowed_user_ids):
        logger.warning(
            f"Blocked request from unauthorized user {interaction.user.id} ({interaction.user})"
        )
        return False, "You are not authorized to use this bot."

    return True, None


async def run_investigation(
    interaction: discord.Interaction, question: str, use_sonnet: bool = False
):
    """Core investigation logic with rate limiting, concurrency control, and ephemeral responses.

    Args:
        interaction: Discord interaction
        question: The query to investigate
        use_sonnet: If True, use Sonnet 4.5. If False (default), use Haiku for cost savings.
    """
    bot = get_bot()
    user_id = interaction.user.id

    # Check permissions
    allowed, perm_error = await check_permissions(interaction)
    if not allowed:
        await interaction.response.send_message(
            f"\u26d4 {perm_error}",
            ephemeral=True
        )
        return

    # Check rate limits
    allowed, rate_error = await bot.rate_limiter.check_rate_limit(user_id)
    if not allowed:
        logger.warning(f"Rate limited user {interaction.user} ({user_id}): {rate_error}")
        await interaction.response.send_message(
            f"\u23f3 {rate_error}",
            ephemeral=True
        )
        return

    # Check concurrent investigation limit
    if bot._active_investigations >= MAX_CONCURRENT_INVESTIGATIONS:
        logger.warning(
            f"Concurrent limit reached ({bot._active_investigations}/{MAX_CONCURRENT_INVESTIGATIONS}), "
            f"queuing request from {interaction.user}"
        )

    # Check if trading process is running (informational - not blocking)
    trading_running, trading_status = await check_trading_process_running()
    trading_warning = None
    if not trading_running:
        logger.warning(f"Trading process not running: {trading_status}")
        trading_warning = f"\u26a0\ufe0f **Warning**: {trading_status}\n_Historical data and logs may still be available._\n"

    # Defer the response since investigation takes time
    await interaction.response.defer(thinking=True)

    # Acquire semaphore for concurrency control
    async with bot.investigation_semaphore:
        bot._active_investigations += 1

        try:
            # Record the request for rate limiting
            await bot.rate_limiter.record_request(user_id)

            # Select model based on use_sonnet flag
            model = bot.config.model_sonnet if use_sonnet else bot.config.model_haiku
            model_name = "Sonnet 4.5" if use_sonnet else "Haiku"

            logger.info(
                f"Investigation request from {interaction.user} ({user_id}) "
                f"in #{interaction.channel.name if interaction.channel else 'unknown'}: {question[:100]}... "
                f"[{bot._active_investigations}/{MAX_CONCURRENT_INVESTIGATIONS} active] [Model: {model_name}]"
            )

            # Run the investigation
            result = await bot.investigator.investigate(
                query=question, user_id=str(user_id), model=model
            )

            if not result.success:
                error_msg = format_error(
                    result.error or "Unknown error", context=question[:50]
                )
                await interaction.followup.send(error_msg)
                logger.error(f"Investigation failed: {result.error}")
                return

            # Format and send result
            messages = format_investigation_result(
                answer=result.answer,
                commands_executed=result.commands_executed,
                iterations=result.iterations,
                duration_seconds=result.duration_seconds,
            )

            # Send trading process warning first if applicable
            if trading_warning:
                try:
                    await interaction.followup.send(trading_warning)
                except discord.HTTPException as e:
                    logger.error(f"Failed to send trading warning: {e}")

            # Send result messages
            for i, msg in enumerate(messages[:5]):  # Max 5 messages
                try:
                    await interaction.followup.send(msg)
                except discord.HTTPException as e:
                    logger.error(f"Failed to send result message {i}: {e}")
                    break

            if len(messages) > 5:
                await interaction.followup.send("_[Output truncated - too long for Discord]_")

            logger.info(
                f"Investigation completed: {result.iterations} iterations, "
                f"{len(result.commands_executed)} commands, {result.duration_seconds:.1f}s"
            )

        except aiohttp.ClientError as e:
            logger.error(f"Network error during investigation: {e}", exc_info=True)
            await interaction.followup.send("\u274c Network error - please try again")

        except discord.HTTPException as e:
            logger.error(f"Discord API error during investigation: {e}", exc_info=True)
            try:
                await interaction.followup.send(f"\u274c Discord error: {str(e)[:100]}")
            except discord.HTTPException:
                pass

        except Exception as e:
            logger.error(f"Investigation failed: {e}", exc_info=True)
            try:
                await interaction.followup.send(f"\u274c Investigation failed: {str(e)[:200]}")
            except discord.HTTPException:
                pass

        finally:
            bot._active_investigations -= 1


# ============================================================================
# SLASH COMMANDS
# ============================================================================

@app_commands.command(name="ask", description="Ask any question about the trading system")
@app_commands.describe(question="What would you like to know about the trading system?")
async def ask_command(interaction: discord.Interaction, question: str):
    """Ask a natural language question about the trading system. Uses Sonnet 4.5."""
    await run_investigation(interaction, question, use_sonnet=True)


@app_commands.command(name="status", description="Check if the trading bot is running")
async def status_command(interaction: discord.Interaction):
    """Quick health check of the trading bot."""
    await run_investigation(
        interaction, "Check if the trading bot is running and show any recent errors"
    )


@app_commands.command(name="signals", description="Show today's trading signals")
async def signals_command(interaction: discord.Interaction):
    """Show today's trading signals."""
    await run_investigation(
        interaction, "Show today's trading signals from the session file"
    )


@app_commands.command(name="trades", description="Show today's executed trades")
async def trades_command(interaction: discord.Interaction):
    """Show today's executed trades."""
    await run_investigation(
        interaction, "Show today's executed trades from the trades CSV"
    )


@app_commands.command(name="logs", description="Show recent log entries")
@app_commands.describe(lines="Number of log lines to show (default 50, max 200)")
async def logs_command(interaction: discord.Interaction, lines: int = 50):
    """Show recent log entries."""
    lines = min(lines, 200)  # Cap at 200 lines
    await run_investigation(
        interaction, f"Show the last {lines} lines of today's log file"
    )


@app_commands.command(name="errors", description="Search for errors in today's logs")
async def errors_command(interaction: discord.Interaction):
    """Search for errors in today's logs."""
    await run_investigation(
        interaction, "Search for any errors or warnings in today's logs"
    )


@app_commands.command(name="bothelp", description="Show available commands and usage")
async def help_command(interaction: discord.Interaction):
    """Show available commands."""
    bot = get_bot()

    # Check permissions
    allowed, perm_error = await check_permissions(interaction)
    if not allowed:
        await interaction.response.send_message(
            f"\u26d4 {perm_error}",
            ephemeral=True
        )
        return

    help_text = f"""**Homeguard Trading Monitor - Slash Commands**

**Investigation Commands:**
`/ask <question>` - Ask any question about the trading system
`/status` - Check if trading bot is running
`/signals` - Show today's trading signals
`/trades` - Show today's executed trades
`/logs [lines]` - Show recent log entries (default 50)
`/errors` - Search for errors in logs

**Utility Commands:**
`/bothelp` - Show this help message
`/ping` - Check bot latency
`/botstats` - Show bot statistics and rate limits

**Rate Limits:**
- {USER_COOLDOWN_SECONDS}s cooldown between requests
- {USER_RATE_LIMIT_PER_MINUTE} requests/minute, {USER_RATE_LIMIT_PER_HOUR}/hour per user
- {MAX_CONCURRENT_INVESTIGATIONS} concurrent investigations max

**About:**
This bot is READ-ONLY. It can observe logs, status, and configurations but cannot modify anything or control services.

_Powered by Claude_"""

    await interaction.response.send_message(help_text)


@app_commands.command(name="ping", description="Check bot latency")
async def ping_command(interaction: discord.Interaction):
    """Check bot latency."""
    bot = get_bot()

    # Check permissions
    allowed, perm_error = await check_permissions(interaction)
    if not allowed:
        await interaction.response.send_message(
            f"\u26d4 {perm_error}",
            ephemeral=True
        )
        return

    latency_ms = bot.latency * 1000
    await interaction.response.send_message(f"\U0001f3d3 Pong! Latency: {latency_ms:.2f}ms")


@app_commands.command(name="botstats", description="Show bot statistics and rate limits")
async def botstats_command(interaction: discord.Interaction):
    """Show bot statistics and rate limit info."""
    bot = get_bot()

    # Check permissions
    allowed, perm_error = await check_permissions(interaction)
    if not allowed:
        await interaction.response.send_message(
            f"\u26d4 {perm_error}",
            ephemeral=True
        )
        return

    stats = bot.rate_limiter.get_stats()
    stats_text = f"""**Bot Statistics**

**Current Load:**
- Active investigations: {bot._active_investigations}/{MAX_CONCURRENT_INVESTIGATIONS}
- Requests (last minute): {stats['global_requests_last_minute']}/{GLOBAL_RATE_LIMIT_PER_MINUTE}
- Unique users (last hour): {stats['unique_users_last_hour']}

**Rate Limits:**
- Per-user: {USER_RATE_LIMIT_PER_MINUTE}/min, {USER_RATE_LIMIT_PER_HOUR}/hour
- Cooldown: {USER_COOLDOWN_SECONDS}s between requests
- Global: {GLOBAL_RATE_LIMIT_PER_MINUTE}/min

**Connection:**
- Latency: {bot.latency * 1000:.2f}ms
- Guilds: {len(bot.guilds)}
- Command Type: Slash Commands"""

    await interaction.response.send_message(stats_text)


# ============================================================================
# BOT CREATION AND LIFECYCLE
# ============================================================================

def create_bot(config: DiscordBotConfig) -> TradingMonitorBot:
    """Create and configure the bot instance."""
    global _bot
    _bot = TradingMonitorBot(config)
    return _bot


async def run_bot_with_reconnect(config: DiscordBotConfig):
    """
    Run the bot with automatic reconnection handling.

    Implements exponential backoff for reconnection attempts.
    """
    reconnect_attempts = 0

    while True:
        try:
            logger.info("Creating bot instance...")
            bot = create_bot(config)

            logger.info("Starting bot connection to Discord...")
            async with bot:
                await bot.start(config.discord_token)

        except discord.LoginFailure as e:
            logger.error(f"AUTHENTICATION FAILED: {e}")
            logger.error("Please check your DISCORD_TOKEN is valid")
            logger.error("Get a new token from: https://discord.com/developers/applications")
            sys.exit(1)  # Don't retry auth failures

        except discord.PrivilegedIntentsRequired as e:
            logger.error(f"PRIVILEGED INTENTS REQUIRED: {e}")
            logger.error("Enable MESSAGE CONTENT INTENT in Discord Developer Portal:")
            logger.error("  1. Go to https://discord.com/developers/applications")
            logger.error("  2. Select your application -> Bot")
            logger.error("  3. Enable 'MESSAGE CONTENT INTENT'")
            sys.exit(1)  # Don't retry intent errors

        except aiohttp.ClientConnectorError as e:
            reconnect_attempts += 1
            delay = min(RECONNECT_DELAY_BASE * (2 ** reconnect_attempts), 300)  # Max 5 min
            logger.error(f"Network connection error: {e}")
            logger.warning(
                f"Reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} "
                f"in {delay}s..."
            )

            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.error("Max reconnection attempts reached. Exiting.")
                sys.exit(1)

            await asyncio.sleep(delay)

        except discord.GatewayNotFound:
            logger.error("Discord gateway not found - Discord may be down")
            logger.warning("Retrying in 60 seconds...")
            await asyncio.sleep(60)

        except discord.HTTPException as e:
            if e.status == 429:  # Rate limited
                retry_after = e.retry_after if hasattr(e, 'retry_after') else 60
                logger.warning(f"Rate limited by Discord. Waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
            else:
                reconnect_attempts += 1
                logger.error(f"Discord HTTP error: {e}")
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logger.error("Max reconnection attempts reached. Exiting.")
                    sys.exit(1)
                await asyncio.sleep(RECONNECT_DELAY_BASE * reconnect_attempts)

        except asyncio.CancelledError:
            logger.info("Bot shutdown requested")
            break

        except Exception as e:
            reconnect_attempts += 1
            logger.error(f"Unexpected error: {e}", exc_info=True)

            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.error("Max reconnection attempts reached. Exiting.")
                sys.exit(1)

            delay = min(RECONNECT_DELAY_BASE * reconnect_attempts, 60)
            logger.warning(f"Retrying in {delay}s...")
            await asyncio.sleep(delay)


def main():
    """Main entry point with validation."""
    logger.info("=" * 60)
    logger.info("HOMEGUARD DISCORD BOT STARTING")
    logger.info(f"  Time: {datetime.now().isoformat()}")
    logger.info(f"  PID: {os.getpid()}")
    logger.info(f"  Log directory: {LOG_DIR}")
    logger.info(f"  Command Type: Slash Commands (/)")
    logger.info("=" * 60)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()

    # Validate Discord token format
    logger.info("Validating Discord token...")
    token_valid, token_msg = validate_discord_token(config.discord_token)
    if not token_valid:
        logger.error(f"Discord token validation failed: {token_msg}")
        sys.exit(1)
    logger.info(f"  Discord token: {token_msg}")

    # Validate Anthropic API key format
    logger.info("Validating Anthropic API key...")
    api_valid, api_msg = validate_anthropic_key(config.anthropic_api_key)
    if not api_valid:
        logger.error(f"Anthropic API key validation failed: {api_msg}")
        sys.exit(1)
    logger.info(f"  Anthropic key: {api_msg}")

    # Validate overall config
    is_valid, error_msg = config.validate()
    if not is_valid:
        logger.error(f"Configuration validation failed: {error_msg}")
        sys.exit(1)

    # Log configuration summary
    logger.info("Configuration loaded successfully:")
    logger.info(f"  Max iterations: {config.max_iterations}")
    logger.info(f"  Command timeout: {config.command_timeout}s")
    logger.info(f"  Max output size: {config.max_output_size}")
    logger.info(f"  Allowed channels: {config.allowed_channel_ids or 'All'}")
    logger.info(f"  Allowed users: {config.allowed_user_ids or 'All'}")

    # Run the bot
    try:
        asyncio.run(run_bot_with_reconnect(config))
    except KeyboardInterrupt:
        logger.info("Bot shutdown by keyboard interrupt (Ctrl+C)")
    except SystemExit as e:
        logger.info(f"Bot exiting with code {e.code}")
        raise
    except Exception as e:
        logger.error(f"Bot crashed with unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
