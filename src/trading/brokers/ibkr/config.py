"""
IBKR Configuration -- Pydantic settings with YAML + env override support.

Ports:
    4001 = IB Gateway live
    4002 = IB Gateway paper
    7496 = TWS live
    7497 = TWS paper
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class IBKRConfig(BaseModel):
    """
    Configuration for IBKR connection and behaviour.

    Load order (later overrides earlier):
        1. Defaults in this class
        2. config/ibkr.yaml (if present)
        3. Environment variables: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, etc.
    """

    # -- Connection --------------------------------------------------------
    host: str = Field(default="127.0.0.1", description="Gateway/TWS hostname")
    port: int = Field(default=4002, description="API socket port")
    client_id: int = Field(default=1, description="Unique client ID for this connection")
    timeout_seconds: float = Field(default=30.0, description="Connection/request timeout")
    readonly: bool = Field(default=False, description="Connect in read-only mode (no orders)")
    account: str = Field(default="", description="Account ID (empty = auto-detect first)")

    # -- Reconnection ------------------------------------------------------
    max_reconnect_attempts: int = Field(default=50)
    reconnect_base_delay: float = Field(default=2.0, description="Initial backoff delay (seconds)")
    reconnect_max_delay: float = Field(default=30.0, description="Maximum backoff delay (seconds)")

    # -- Historical data pacing (see pacing.py) ----------------------------
    historical_pacing_max_per_10min: int = Field(
        default=58,
        description="Max historical requests per 10-min window. IBKR limit is 60; leave headroom.",
    )
    identical_request_cooldown_seconds: float = Field(
        default=15.0,
        description="Minimum seconds between identical historical requests.",
    )

    # -- Streaming ---------------------------------------------------------
    max_streaming_lines: int = Field(
        default=100,
        description="Max concurrent reqMktData subscriptions. Check your IBKR account limits.",
    )

    # -- Market data type --------------------------------------------------
    market_data_type: int = Field(
        default=3,
        description="1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen. Paper accounts use 3.",
    )

    @model_validator(mode="before")
    @classmethod
    def _load_yaml_and_env(cls, values):
        """Layer YAML file, then environment variables, onto the provided values."""
        # 1. Try loading YAML
        yaml_path = Path("config/ibkr.yaml")
        if yaml_path.exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
            ibkr_section = raw.get("ibkr", raw)
            # YAML values are defaults; explicit kwargs override
            merged = {**ibkr_section, **{k: v for k, v in values.items() if v is not None}}
        else:
            merged = dict(values)

        # 2. Environment overrides
        env_map = {
            "IBKR_HOST": "host",
            "IBKR_PORT": "port",
            "IBKR_CLIENT_ID": "client_id",
            "IBKR_ACCOUNT": "account",
            "IBKR_READONLY": "readonly",
            "IBKR_TIMEOUT": "timeout_seconds",
            "IBKR_MARKET_DATA_TYPE": "market_data_type",
        }
        for env_key, field_name in env_map.items():
            env_val = os.environ.get(env_key)
            if env_val is not None:
                merged[field_name] = env_val

        return merged

    @property
    def is_paper(self) -> bool:
        """Heuristic: paper trading uses ports 4002 or 7497."""
        return self.port in (4002, 7497)

    @property
    def gateway_type(self) -> str:
        """Human-readable connection target."""
        if self.port in (4001, 4002):
            return f"IB Gateway ({'paper' if self.is_paper else 'live'})"
        return f"TWS ({'paper' if self.is_paper else 'live'})"
