"""
Safe async subprocess execution for Discord bot.

Executes shell commands with security validation, timeout enforcement,
and output truncation.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .security import validate_command, sanitize_output


@dataclass
class ExecutionResult:
    """Result of command execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    blocked: bool
    block_reason: Optional[str] = None

    def format_output(self) -> str:
        """Format the execution result for display."""
        if self.blocked:
            return f"COMMAND BLOCKED: {self.block_reason}"

        if self.timed_out:
            return f"COMMAND TIMED OUT\n\nPartial output:\n{self.stdout[:1000] if self.stdout else '(none)'}"

        parts = []
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        parts.append(f"EXIT CODE: {self.exit_code}")

        return "\n\n".join(parts)


class CommandExecutor:
    """
    Safe async command executor with security validation.

    All commands are validated against the security whitelist before execution.
    Output is sanitized to remove sensitive data before returning.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 50000,
        working_dir: Optional[str] = None,
    ):
        """
        Initialize the command executor.

        Args:
            timeout: Default timeout in seconds (max 120)
            max_output: Maximum output size in characters
            working_dir: Default working directory for commands
        """
        self.timeout = min(timeout, 120)
        self.max_output = max_output
        self.working_dir = working_dir

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a shell command with security validation.

        Args:
            command: The shell command to execute
            timeout: Override default timeout (max 120 seconds)
            working_dir: Override default working directory

        Returns:
            ExecutionResult with stdout, stderr, exit code, and status flags
        """
        # Validate command against security rules
        validation = validate_command(command)
        if not validation.is_allowed:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=False,
                blocked=True,
                block_reason=validation.reason,
            )

        # Apply timeout limits
        effective_timeout = min(timeout or self.timeout, 120)
        effective_dir = working_dir or self.working_dir

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_dir,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                # Try to get partial output
                partial_stdout = ""
                try:
                    if process.stdout:
                        partial_data = await asyncio.wait_for(
                            process.stdout.read(1000), timeout=1.0
                        )
                        partial_stdout = self._decode_and_truncate(partial_data)
                except (asyncio.TimeoutError, Exception):
                    pass

                return ExecutionResult(
                    stdout=sanitize_output(partial_stdout),
                    stderr="",
                    exit_code=-1,
                    timed_out=True,
                    blocked=False,
                )

            stdout_str = self._decode_and_truncate(stdout_bytes)
            stderr_str = self._decode_and_truncate(stderr_bytes)

            return ExecutionResult(
                stdout=sanitize_output(stdout_str),
                stderr=sanitize_output(stderr_str),
                exit_code=process.returncode or 0,
                timed_out=False,
                blocked=False,
            )

        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution error: {type(e).__name__}: {str(e)}",
                exit_code=-1,
                timed_out=False,
                blocked=False,
            )

    def _decode_and_truncate(self, data: bytes) -> str:
        """Decode bytes and truncate if necessary."""
        text = data.decode("utf-8", errors="replace")
        if len(text) > self.max_output:
            truncated_amount = len(text) - self.max_output
            return text[: self.max_output] + f"\n... [TRUNCATED {truncated_amount} chars]"
        return text
