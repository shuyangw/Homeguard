"""
Discord bot configuration management.

Loads configuration from environment variables and provides defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set


@dataclass
class DiscordBotConfig:
    """Configuration for the Discord monitoring bot."""

    # API tokens (required)
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", ""))
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Channel/user restrictions (comma-separated IDs in env)
    allowed_channel_ids: Set[int] = field(default_factory=set)
    allowed_user_ids: Set[int] = field(default_factory=set)

    # Investigation limits
    max_iterations: int = 10
    command_timeout: int = 30
    max_output_size: int = 50000

    # Claude models
    model_sonnet: str = "claude-sonnet-4-5-20250929"
    model_haiku: str = "claude-3-5-haiku-20241022"

    # Homeguard-specific paths (EC2 defaults)
    log_dir: Path = field(default_factory=lambda: Path.home() / "logs")
    trading_service: str = "homeguard-trading.service"
    repo_dir: Path = field(default_factory=lambda: Path.home() / "Homeguard")

    def __post_init__(self):
        """Parse channel and user IDs from environment."""
        channels_str = os.getenv("ALLOWED_CHANNELS", "")
        users_str = os.getenv("ALLOWED_USERS", "")

        if channels_str:
            self.allowed_channel_ids = {
                int(x.strip()) for x in channels_str.split(",") if x.strip()
            }
        if users_str:
            self.allowed_user_ids = {
                int(x.strip()) for x in users_str.split(",") if x.strip()
            }

    def validate(self) -> tuple[bool, str]:
        """Validate that required configuration is present."""
        if not self.discord_token:
            return False, "DISCORD_TOKEN environment variable not set"
        if not self.anthropic_api_key:
            return False, "ANTHROPIC_API_KEY environment variable not set"
        return True, "Configuration valid"


def load_config() -> DiscordBotConfig:
    """Load configuration from environment variables."""
    return DiscordBotConfig()
