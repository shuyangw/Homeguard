"""
Discord message formatting utilities.

Handles splitting long messages, creating embeds, and formatting
investigation results for Discord's character limits.
"""

from typing import List, Optional

# Discord limits
MAX_MESSAGE_LENGTH = 2000
MAX_EMBED_DESCRIPTION = 4096
MAX_EMBED_FIELD_VALUE = 1024
CODE_BLOCK_OVERHEAD = 8  # ```\n...\n```


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message into chunks that fit Discord's character limit.

    Args:
        text: The full message text
        max_length: Maximum length per chunk (default 2000)

    Returns:
        List of message chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try to split at a newline
        split_point = remaining.rfind("\n", 0, max_length)
        if split_point == -1 or split_point < max_length // 2:
            # No good newline, split at space
            split_point = remaining.rfind(" ", 0, max_length)
            if split_point == -1 or split_point < max_length // 2:
                # No good space, hard split
                split_point = max_length

        chunks.append(remaining[:split_point])
        remaining = remaining[split_point:].lstrip()

    return chunks


def format_code_block(text: str, language: str = "") -> str:
    """
    Wrap text in a Discord code block.

    Args:
        text: The text to wrap
        language: Optional syntax highlighting language

    Returns:
        Text wrapped in code block markers
    """
    return f"```{language}\n{text}\n```"


def format_investigation_result(
    answer: str,
    commands_executed: List[str],
    iterations: int,
    duration_seconds: float,
) -> List[str]:
    """
    Format an investigation result for Discord.

    Args:
        answer: The investigation answer
        commands_executed: List of commands that were run
        iterations: Number of investigation iterations
        duration_seconds: Total investigation duration

    Returns:
        List of message strings to send
    """
    messages = []

    # Format the main answer
    answer_chunks = split_message(answer, MAX_MESSAGE_LENGTH - 100)

    for i, chunk in enumerate(answer_chunks):
        if i == 0:
            messages.append(chunk)
        else:
            messages.append(chunk)

    # Add metadata footer (only if room)
    metadata = f"\n\n_Investigation: {iterations} steps, {len(commands_executed)} commands, {duration_seconds:.1f}s_"
    if messages and len(messages[-1]) + len(metadata) <= MAX_MESSAGE_LENGTH:
        messages[-1] += metadata
    else:
        messages.append(metadata)

    return messages


def format_error(error: str, context: Optional[str] = None) -> str:
    """
    Format an error message for Discord.

    Args:
        error: The error message
        context: Optional context about what failed

    Returns:
        Formatted error string
    """
    parts = ["**Investigation Failed**"]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Error: {error}")
    return "\n".join(parts)


def format_status_response(
    service_status: str,
    recent_errors: Optional[str] = None,
    last_trade: Optional[str] = None,
) -> str:
    """
    Format a quick status response.

    Args:
        service_status: Current service status line
        recent_errors: Any recent errors found
        last_trade: Most recent trade info

    Returns:
        Formatted status string
    """
    parts = ["**Trading Bot Status**", ""]

    parts.append(f"**Service:** {service_status}")

    if last_trade:
        parts.append(f"**Last Trade:** {last_trade}")

    if recent_errors:
        parts.append(f"**Recent Errors:**\n{format_code_block(recent_errors[:500])}")
    else:
        parts.append("**Errors:** None in recent logs")

    return "\n".join(parts)


def truncate_for_discord(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """
    Truncate text to fit Discord's limit with indicator.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with indicator if needed
    """
    if len(text) <= max_length:
        return text

    truncate_indicator = "\n... [truncated]"
    return text[: max_length - len(truncate_indicator)] + truncate_indicator
