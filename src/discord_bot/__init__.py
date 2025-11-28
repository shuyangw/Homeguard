"""
Homeguard Discord Bot - Read-only observability addon.

This module provides a Discord bot for monitoring the Homeguard trading system
through natural language queries. It uses Claude's tool use API to investigate
logs, status, and configurations.

CRITICAL: This is a READ-ONLY observability tool.
- No imports from src.trading, src.backtesting, src.gui, or src.strategies
- All data access is via shell commands reading files/logs
- Cannot modify, restart, or control any services
"""

__version__ = "0.1.0"
