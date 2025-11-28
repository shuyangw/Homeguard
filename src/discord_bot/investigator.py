"""
Claude-powered trading system investigator.

Uses Anthropic's tool use API with a ReAct pattern to investigate
the Homeguard trading system through read-only shell commands.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from .config import DiscordBotConfig
from .executor import CommandExecutor

logger = logging.getLogger(__name__)

# System prompt for Claude - emphasizes READ-ONLY access
SYSTEM_PROMPT = """You are a READ-ONLY observer for the Homeguard algorithmic trading bot on AWS EC2.

## CRITICAL: READ-ONLY MODE
You can ONLY run read-only commands. You CANNOT:
- Start, stop, or restart any services
- Modify any files or configurations
- Execute any destructive commands
- Use sudo or elevated privileges

Your purpose is OBSERVABILITY ONLY - answering questions about system status, logs, and trading activity.

## System Context
- Trading bot service: homeguard-trading.service
- Log directory: ~/logs/live_trading/paper/YYYYMMDD/
- Trading config: ~/Homeguard/config/trading/omr_trading_config.yaml
- Strategy: Overnight Mean Reversion (OMR)
- Entry time: 3:50 PM ET
- Exit time: 9:31 AM ET (next day)
- Market hours: Mon-Fri 9:00 AM - 4:30 PM ET
- Trading universe: 20 leveraged ETFs (TQQQ, SOXL, UPRO, UDOW, TNA, etc.)

## Log File Structure
Each trading day creates a directory with:
- *_session.json - Complete session data (signals, orders, checks)
- *_trades.csv - Executed trades
- *_market_checks.csv - Market status checks
- *.log - Buffered log output
- *_summary.md - End-of-day summary

## Allowed Read-Only Commands
- Service status: systemctl status homeguard-trading.service
- Recent logs: tail -100 ~/logs/live_trading/paper/$(date +%Y%m%d)/*.log
- Today's trades: cat ~/logs/live_trading/paper/$(date +%Y%m%d)/*trades.csv
- Search errors: grep -i error ~/logs/live_trading/paper/$(date +%Y%m%d)/*.log
- Session data: cat ~/logs/live_trading/paper/$(date +%Y%m%d)/*session.json
- Journal: journalctl -u homeguard-trading -n 50 --no-pager
- Process list: ps aux | grep python
- View config: cat ~/Homeguard/config/trading/omr_trading_config.yaml
- Disk usage: df -h
- Memory: free -h
- List log dirs: ls -la ~/logs/live_trading/paper/

## Investigation Guidelines
1. Start with the most relevant command for the question
2. Analyze output before running additional commands
3. Look for patterns in logs (errors, warnings, signals)
4. Check service status if bot behavior is in question
5. Provide clear, actionable summaries

If asked to modify, restart, or control anything, politely explain you are read-only and can only observe."""

# Tool definition for shell command execution
TOOLS = [
    {
        "name": "run_shell_command",
        "description": "Execute a read-only shell command to investigate the trading system. Returns stdout, stderr, and exit code. Only read-only commands are allowed (tail, cat, grep, ps, systemctl status, journalctl, etc.). Write operations, service control, and sudo are blocked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The read-only shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 120)",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    }
]


@dataclass
class InvestigationSession:
    """Tracks state during a multi-step investigation."""

    query: str
    user_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    commands_executed: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: Any):
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": content})


@dataclass
class InvestigationResult:
    """Result of a completed investigation."""

    answer: str
    commands_executed: List[str]
    iterations: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class TradingInvestigator:
    """
    Claude-powered investigator for the Homeguard trading system.

    Uses a ReAct pattern to iteratively investigate queries by:
    1. Analyzing the question
    2. Deciding what command to run
    3. Executing the command (read-only only)
    4. Analyzing the output
    5. Repeating until an answer is found or max iterations reached
    """

    def __init__(self, config: DiscordBotConfig):
        """
        Initialize the investigator.

        Args:
            config: Discord bot configuration with API keys and limits
        """
        self.config = config
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self.executor = CommandExecutor(
            timeout=config.command_timeout,
            max_output=config.max_output_size,
        )

    async def investigate(
        self, query: str, user_id: str, model: str | None = None
    ) -> InvestigationResult:
        """
        Run a multi-step investigation to answer a query.

        Args:
            query: The user's question about the trading system
            user_id: Discord user ID for logging
            model: Claude model to use (defaults to config.model_sonnet)

        Returns:
            InvestigationResult with the answer and metadata
        """
        # Use provided model or default to Sonnet
        model = model or self.config.model_sonnet

        session = InvestigationSession(query=query, user_id=user_id)
        session.add_user_message(query)

        logger.info(
            f"Starting investigation for user {user_id} using {model}: {query[:100]}..."
        )

        try:
            while session.iteration < self.config.max_iterations:
                session.iteration += 1
                logger.debug(
                    f"Investigation iteration {session.iteration}/{self.config.max_iterations}"
                )

                # Call Claude
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=session.messages,
                )

                # Check if Claude is done (natural completion)
                if response.stop_reason == "end_turn":
                    answer = self._extract_text(response.content)
                    duration = (datetime.now() - session.started_at).total_seconds()

                    logger.info(
                        f"Investigation completed in {session.iteration} iterations, "
                        f"{len(session.commands_executed)} commands"
                    )

                    return InvestigationResult(
                        answer=answer,
                        commands_executed=session.commands_executed,
                        iterations=session.iteration,
                        duration_seconds=duration,
                        success=True,
                    )

                # Claude wants to use a tool
                if response.stop_reason == "tool_use":
                    session.add_assistant_message(response.content)

                    # Process tool calls
                    tool_results = await self._process_tool_calls(
                        response.content, session
                    )
                    session.add_user_message(tool_results)

            # Hit max iterations - force conclusion
            logger.warning(
                f"Investigation hit max iterations ({self.config.max_iterations})"
            )
            session.add_user_message(
                "Maximum investigation steps reached. Please summarize your findings now based on what you've learned."
            )

            final_response = await self.client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=session.messages,
            )

            answer = self._extract_text(final_response.content)
            duration = (datetime.now() - session.started_at).total_seconds()

            return InvestigationResult(
                answer=answer,
                commands_executed=session.commands_executed,
                iterations=session.iteration,
                duration_seconds=duration,
                success=True,
            )

        except Exception as e:
            logger.error(f"Investigation failed: {e}", exc_info=True)
            duration = (datetime.now() - session.started_at).total_seconds()

            return InvestigationResult(
                answer="",
                commands_executed=session.commands_executed,
                iterations=session.iteration,
                duration_seconds=duration,
                success=False,
                error=str(e),
            )

    async def _process_tool_calls(
        self, content_blocks: List[Any], session: InvestigationSession
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls from Claude's response.

        Args:
            content_blocks: Response content blocks from Claude
            session: Current investigation session

        Returns:
            List of tool result objects for the next message
        """
        results = []

        for block in content_blocks:
            if block.type == "tool_use":
                command = block.input.get("command", "")
                timeout = min(block.input.get("timeout", 30), 120)

                logger.info(f"Executing command: {command[:100]}...")
                session.commands_executed.append(command)

                # Execute the command
                exec_result = await self.executor.execute(command, timeout=timeout)

                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": exec_result.format_output(),
                    }
                )

        return results

    def _extract_text(self, content: List[Any]) -> str:
        """Extract text from response content blocks."""
        for block in content:
            if hasattr(block, "text"):
                return block.text
        return "Investigation complete but no summary was generated."
