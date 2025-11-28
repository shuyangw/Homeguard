"""
Security module for Discord bot command validation.

Enforces READ-ONLY access by whitelisting safe commands and blocking
any write operations, service control, or destructive patterns.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

# Allowed read-only command prefixes
ALLOWED_COMMAND_PREFIXES: List[str] = [
    # File reading
    "tail",
    "cat",
    "head",
    "grep",
    "less",
    "wc",
    "awk",
    "sed",  # sed for reading patterns only, blocked for -i
    # File system inspection
    "ls",
    "find",
    "stat",
    "file",
    "tree",
    # Process inspection
    "ps",
    "top -bn1",
    "pgrep",
    "pidof",
    # System status
    "df",
    "du",
    "free",
    "uptime",
    "hostname",
    "date",
    "who",
    "w",
    # Service status (read-only)
    "systemctl status",
    "systemctl is-active",
    "systemctl is-enabled",
    "systemctl show",
    "systemctl list-units",
    # Journal reading
    "journalctl",
    # Network inspection
    "netstat",
    "ss",
    "ip addr",
    "ip route",
    # Environment
    "env",
    "printenv",
    "echo $",  # Only for variable expansion
    # Python/pip inspection
    "python --version",
    "python3 --version",
    "pip list",
    "pip show",
    "pip3 list",
    "pip3 show",
    # JSON parsing (read-only)
    "jq",
    "python -m json.tool",
    "python3 -m json.tool",
]

# Explicitly blocked patterns (even if command starts with allowed prefix)
BLOCKED_PATTERNS: List[str] = [
    # Output redirection (writes to files)
    r"[^2]>",  # > but not 2> (stderr redirect is ok for reading)
    r">>",  # Append
    # Pipe to destructive commands
    r"\|\s*rm",
    r"\|\s*kill",
    r"\|\s*pkill",
    r"\|\s*xargs\s+rm",
    r"\|\s*tee\s",  # tee writes to files
    # Privilege escalation
    r"\bsudo\b",
    r"\bsu\b",
    r"\bdoas\b",
    # File modification
    r"\brm\b",
    r"\bmv\b",
    r"\bcp\b",
    r"\bmkdir\b",
    r"\brmdir\b",
    r"\btouch\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bchgrp\b",
    r"\bln\b",
    # In-place editing
    r"\bsed\s+-i",
    r"\bsed\s+--in-place",
    # Service control
    r"systemctl\s+(start|stop|restart|reload|enable|disable|mask|unmask)",
    r"service\s+\S+\s+(start|stop|restart|reload)",
    # Process control
    r"\bkill\b",
    r"\bpkill\b",
    r"\bkillall\b",
    # Dangerous commands
    r"\bdd\b",
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bparted\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\binit\b",
    # Fork bomb pattern
    r":\(\)\s*\{",
    # Network attacks
    r"\bnmap\b",
    r"\bnc\s+-l",  # netcat listen mode
    # Package management
    r"\bapt\b",
    r"\bapt-get\b",
    r"\byum\b",
    r"\bdnf\b",
    r"\bpip\s+install\b",
    r"\bpip3\s+install\b",
    r"\bpip\s+uninstall\b",
    # Git write operations
    r"\bgit\s+(push|commit|reset|checkout|merge|rebase|pull)",
    # Vim/editors (could modify files)
    r"\bvim?\b",
    r"\bnano\b",
    r"\bemacs\b",
    # Command substitution that could hide malicious commands
    r"\$\([^)]*rm\b",
    r"`[^`]*rm\b",
]


@dataclass
class ValidationResult:
    """Result of command validation."""

    is_allowed: bool
    reason: str
    command: str


def validate_command(command: str) -> ValidationResult:
    """
    Validate a shell command against security rules.

    Returns ValidationResult with is_allowed=True only if:
    1. Command starts with an allowed prefix
    2. Command does not match any blocked pattern

    Args:
        command: The shell command to validate

    Returns:
        ValidationResult with validation outcome and reason
    """
    command_stripped = command.strip()
    command_lower = command_stripped.lower()

    # Check for blocked patterns first (higher priority)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command_stripped, re.IGNORECASE):
            return ValidationResult(
                is_allowed=False,
                reason=f"Blocked pattern detected: {pattern}",
                command=command_stripped,
            )

    # Check if command starts with an allowed prefix
    is_allowed_prefix = False
    matched_prefix = None

    for prefix in ALLOWED_COMMAND_PREFIXES:
        if command_lower.startswith(prefix.lower()):
            is_allowed_prefix = True
            matched_prefix = prefix
            break

    if not is_allowed_prefix:
        return ValidationResult(
            is_allowed=False,
            reason=f"Command does not start with an allowed prefix. Allowed: {', '.join(ALLOWED_COMMAND_PREFIXES[:10])}...",
            command=command_stripped,
        )

    return ValidationResult(
        is_allowed=True,
        reason=f"Command allowed (prefix: {matched_prefix})",
        command=command_stripped,
    )


def sanitize_output(text: str) -> str:
    """
    Sanitize command output before sending to Discord.

    Masks potential secrets like API keys, passwords, and AWS credentials.

    Args:
        text: Raw command output

    Returns:
        Sanitized text with secrets masked
    """
    # Remove ANSI escape codes
    text = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)

    # Patterns to mask (pattern, replacement)
    secret_patterns = [
        # API keys
        (r"api[_-]?key[=:]\s*\S+", "api_key=***MASKED***"),
        (r"apikey[=:]\s*\S+", "apikey=***MASKED***"),
        # Secrets and passwords
        (r"secret[=:]\s*\S+", "secret=***MASKED***"),
        (r"password[=:]\s*\S+", "password=***MASKED***"),
        (r"passwd[=:]\s*\S+", "passwd=***MASKED***"),
        # Tokens
        (r"token[=:]\s*\S+", "token=***MASKED***"),
        (r"bearer\s+\S+", "bearer ***MASKED***"),
        # AWS credentials
        (r"AKIA[0-9A-Z]{16}", "***AWS_ACCESS_KEY***"),
        (r"aws_secret[_-]?access[_-]?key[=:]\s*\S+", "aws_secret_access_key=***MASKED***"),
        # Alpaca credentials
        (r"ALPACA[_-]?\w*[_-]?KEY[=:]\s*\S+", "ALPACA_KEY=***MASKED***"),
        (r"ALPACA[_-]?\w*[_-]?SECRET[=:]\s*\S+", "ALPACA_SECRET=***MASKED***"),
        # Discord tokens (start with specific patterns)
        (r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}", "***DISCORD_TOKEN***"),
        # Anthropic keys
        (r"sk-ant-api\d+-\S+", "***ANTHROPIC_KEY***"),
        # Generic long hex strings that might be keys (>32 chars)
        (r"\b[a-fA-F0-9]{40,}\b", "***MASKED_HEX***"),
    ]

    for pattern, replacement in secret_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def is_channel_allowed(channel_id: int, allowed_channels: set) -> bool:
    """Check if a channel is in the allowed list (empty set = all allowed)."""
    if not allowed_channels:
        return True
    return channel_id in allowed_channels


def is_user_allowed(user_id: int, allowed_users: set) -> bool:
    """Check if a user is in the allowed list (empty set = all allowed)."""
    if not allowed_users:
        return True
    return user_id in allowed_users
