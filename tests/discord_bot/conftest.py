"""
Pytest fixtures for Discord bot tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.discord_bot.config import DiscordBotConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = DiscordBotConfig(
        discord_token="test_token",
        anthropic_api_key="test_api_key",
        max_iterations=5,
        command_timeout=10,
        max_output_size=1000,
    )
    return config


@pytest.fixture
def mock_discord_context():
    """Create a mock Discord context."""
    ctx = MagicMock()
    ctx.channel.id = 123456789
    ctx.author.id = 987654321
    ctx.author.__str__ = MagicMock(return_value="TestUser#1234")
    ctx.send = AsyncMock()
    ctx.message.add_reaction = AsyncMock()
    return ctx


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.messages.create = AsyncMock()
    return client
