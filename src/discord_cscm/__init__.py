"""
CSCM Discord Bot - Simple slash commands for portfolio status.

A lightweight Discord bot for monitoring CSCM paper trading.
No LLM required - just direct command handlers.

Available Commands:
- /cscm-status     - Portfolio value, cash, positions, regime
- /cscm-positions  - Detailed position table
- /cscm-refresh    - Force update portfolio value
- /cscm-regime     - BTC regime and top momentum coins

Usage:
    # As a standalone script
    python -m src.discord_cscm.bot

    # Import and run
    from src.discord_cscm import CSCMDiscordBot
    bot = CSCMDiscordBot()
    bot.run()

Environment Variables:
    DISCORD_CSCM_TOKEN     - Discord bot token
    DISCORD_CSCM_GUILD_ID  - Optional: Guild ID for faster command sync
"""

from src.discord_cscm.bot import CSCMDiscordBot, run_bot

__all__ = [
    'CSCMDiscordBot',
    'run_bot',
]
