"""
CSCM Discord Bot - Simple slash commands for portfolio monitoring.

A lightweight Discord bot providing status commands for CSCM paper trading.
No LLM required - just direct command handlers with discord.py.

Commands:
- /cscm-status     - Portfolio value, cash, positions, regime
- /cscm-positions  - Detailed position table
- /cscm-refresh    - Force update portfolio value
- /cscm-regime     - BTC regime and top momentum coins

Usage:
    # Run as module
    python -m src.discord_cscm.bot

    # Or from script
    from src.discord_cscm.bot import run_bot
    run_bot()

Environment Variables:
    DISCORD_CSCM_TOKEN     - Bot token (required)
    DISCORD_CSCM_GUILD_ID  - Guild ID for instant command sync (optional)
"""

import os
import sys
from typing import Optional
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSCMDiscordBot(commands.Bot):
    """
    Simple Discord bot for CSCM paper trading status.

    Provides slash commands for monitoring portfolio without requiring LLM.
    Uses lazy initialization for trading adapter to minimize startup time.

    Commands are registered on startup and synced to the configured guild
    (or globally if no guild is specified).
    """

    def __init__(self):
        """Initialize the Discord bot."""
        intents = discord.Intents.default()
        super().__init__(
            command_prefix='!',
            intents=intents,
            description='CSCM Paper Trading Monitor'
        )

        # Lazy adapter initialization
        self._adapter = None
        self._tracker = None

        # Guild ID for faster command sync (optional)
        guild_id = os.getenv('DISCORD_CSCM_GUILD_ID')
        self._guild = discord.Object(id=int(guild_id)) if guild_id else None

    def _ensure_adapter(self):
        """Lazy initialization of trading adapter."""
        if self._adapter is None:
            try:
                from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
                from src.trading.portfolio.tracker import PortfolioTracker

                self._adapter = CSCMPaperAdapter()
                self._tracker = PortfolioTracker(self._adapter)
                logger.info("[Discord] Adapter initialized")
            except Exception as e:
                logger.error(f"[Discord] Failed to initialize adapter: {e}")
                raise

    async def setup_hook(self):
        """Register slash commands on startup."""
        # Add commands
        self.tree.add_command(cscm_status)
        self.tree.add_command(cscm_positions)
        self.tree.add_command(cscm_refresh)
        self.tree.add_command(cscm_regime)

        # Sync commands
        if self._guild:
            self.tree.copy_global_to(guild=self._guild)
            await self.tree.sync(guild=self._guild)
            logger.info(f"[Discord] Commands synced to guild {self._guild.id}")
        else:
            await self.tree.sync()
            logger.info("[Discord] Commands synced globally")

    async def on_ready(self):
        """Called when bot is ready."""
        logger.info(f"[Discord] Logged in as {self.user}")
        logger.info(f"[Discord] Connected to {len(self.guilds)} guilds")


# Global bot instance (lazy)
_bot: Optional[CSCMDiscordBot] = None


def get_bot() -> CSCMDiscordBot:
    """Get or create bot instance."""
    global _bot
    if _bot is None:
        _bot = CSCMDiscordBot()
    return _bot


# ==================== Slash Commands ====================

@app_commands.command(name="cscm-status", description="Show CSCM paper trading status")
async def cscm_status(interaction: discord.Interaction):
    """Quick portfolio status."""
    bot = get_bot()

    try:
        bot._ensure_adapter()
        status = bot._tracker.get_status()

        # Create embed
        is_bullish = status['regime'].lower() == 'bullish'
        embed = discord.Embed(
            title="CSCM Paper Trading Status",
            color=discord.Color.green() if is_bullish else discord.Color.red()
        )

        embed.add_field(
            name="Portfolio Value",
            value=f"${status['current_value']:,.2f}",
            inline=True
        )
        embed.add_field(
            name="Cash",
            value=f"${status['cash']:,.2f}",
            inline=True
        )
        embed.add_field(
            name="Positions",
            value=str(status['positions_count']),
            inline=True
        )
        embed.add_field(
            name="Regime",
            value=status['regime'].upper(),
            inline=True
        )
        embed.add_field(
            name="Equity Hours",
            value="Yes" if status['is_equity_hours'] else "No",
            inline=True
        )
        embed.add_field(
            name="Last Update",
            value=status['last_update'] or "Never",
            inline=True
        )

        await interaction.response.send_message(embed=embed)

    except Exception as e:
        logger.error(f"[Discord] Status command failed: {e}")
        await interaction.response.send_message(
            f"Error fetching status: {str(e)[:100]}",
            ephemeral=True
        )


@app_commands.command(name="cscm-positions", description="Show current CSCM positions")
async def cscm_positions(interaction: discord.Interaction):
    """Show positions with values."""
    bot = get_bot()

    try:
        bot._ensure_adapter()
        positions = bot._adapter._get_current_positions()
        prices = bot._adapter._get_current_prices()

        if not positions:
            await interaction.response.send_message("No positions currently held.")
            return

        # Build table
        lines = ["```"]
        lines.append(f"{'Symbol':<10} {'Qty':>12} {'Price':>10} {'Value':>12}")
        lines.append("-" * 46)

        total = 0
        for symbol, qty in positions.items():
            price = prices.get(symbol, 0)
            value = float(qty) * price
            total += value
            lines.append(f"{symbol:<10} {float(qty):>12.6f} ${price:>9,.2f} ${value:>11,.2f}")

        lines.append("-" * 46)
        lines.append(f"{'Total':<10} {'':<12} {'':<10} ${total:>11,.2f}")
        lines.append("```")

        await interaction.response.send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"[Discord] Positions command failed: {e}")
        await interaction.response.send_message(
            f"Error fetching positions: {str(e)[:100]}",
            ephemeral=True
        )


@app_commands.command(name="cscm-refresh", description="Force refresh portfolio value")
async def cscm_refresh(interaction: discord.Interaction):
    """On-demand portfolio value update."""
    bot = get_bot()

    await interaction.response.defer(thinking=True)

    try:
        bot._ensure_adapter()
        value = bot._tracker.update_value(force=True)

        await interaction.followup.send(f"Portfolio value refreshed: **${value:,.2f}**")

    except Exception as e:
        logger.error(f"[Discord] Refresh command failed: {e}")
        await interaction.followup.send(f"Error refreshing: {str(e)[:100]}")


@app_commands.command(name="cscm-regime", description="Show BTC regime and top momentum coins")
async def cscm_regime(interaction: discord.Interaction):
    """Show regime and top momentum coins."""
    bot = get_bot()

    await interaction.response.defer(thinking=True)

    try:
        bot._ensure_adapter()

        # Get full status from adapter
        status = bot._adapter.get_status()

        is_bullish = status['regime'].lower() == 'bullish'
        embed = discord.Embed(
            title="CSCM Regime Analysis",
            color=discord.Color.green() if is_bullish else discord.Color.red()
        )

        embed.add_field(
            name="BTC Regime",
            value=f"**{status['regime'].upper()}**",
            inline=True
        )
        embed.add_field(
            name="BTC Price",
            value=f"${status.get('btc_price', 0):,.2f}",
            inline=True
        )
        embed.add_field(
            name="BTC SMA",
            value=f"${status.get('btc_sma', 0):,.2f}",
            inline=True
        )

        # Top symbols
        top_symbols = status.get('top_symbols', [])
        if top_symbols:
            top_list = "\n".join([f"{i+1}. {sym}" for i, sym in enumerate(top_symbols[:5])])
            embed.add_field(
                name="Top Momentum Coins",
                value=top_list or "None",
                inline=False
            )

        # Trailing stop info
        embed.add_field(
            name="Trailing Stop",
            value=f"{'TRIGGERED' if status.get('trailing_stop_triggered') else 'Active'} "
                  f"(DD: {status.get('current_drawdown', 0):.1%})",
            inline=True
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"[Discord] Regime command failed: {e}")
        await interaction.followup.send(f"Error fetching regime: {str(e)[:100]}")


# ==================== Run Function ====================

def run_bot():
    """Run the Discord bot."""
    from dotenv import load_dotenv
    load_dotenv()

    token = os.getenv('DISCORD_CSCM_TOKEN')
    if not token:
        logger.error("[Discord] DISCORD_CSCM_TOKEN not set")
        print("Error: DISCORD_CSCM_TOKEN environment variable not set")
        sys.exit(1)

    logger.info("[Discord] Starting CSCM Discord bot...")
    bot = get_bot()
    bot.run(token)


if __name__ == "__main__":
    run_bot()
