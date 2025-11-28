"""
Tests for Discord bot command executor.

Tests safe subprocess execution with security validation.
"""

import pytest
import asyncio
import sys

from src.discord_bot.executor import CommandExecutor, ExecutionResult


class TestExecutionResult:
    """Test ExecutionResult formatting."""

    def test_format_blocked_command(self):
        """Blocked commands should show block reason."""
        result = ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            timed_out=False,
            blocked=True,
            block_reason="Command not allowed",
        )
        output = result.format_output()
        assert "BLOCKED" in output
        assert "Command not allowed" in output

    def test_format_timed_out(self):
        """Timed out commands should indicate timeout."""
        result = ExecutionResult(
            stdout="partial output",
            stderr="",
            exit_code=-1,
            timed_out=True,
            blocked=False,
        )
        output = result.format_output()
        assert "TIMED OUT" in output

    def test_format_successful(self):
        """Successful commands should show stdout, stderr, exit code."""
        result = ExecutionResult(
            stdout="file contents here",
            stderr="",
            exit_code=0,
            timed_out=False,
            blocked=False,
        )
        output = result.format_output()
        assert "STDOUT" in output
        assert "file contents here" in output
        assert "EXIT CODE: 0" in output


class TestCommandExecutor:
    """Test CommandExecutor async execution."""

    @pytest.fixture
    def executor(self):
        """Create an executor instance."""
        return CommandExecutor(timeout=5, max_output=1000)

    def test_blocks_rm_command(self, executor):
        """rm commands should be blocked before execution."""
        result = asyncio.run(executor.execute("rm -rf /tmp/test"))
        assert result.blocked is True
        assert "rm" in result.block_reason.lower() or "blocked" in result.block_reason.lower()

    def test_blocks_sudo_command(self, executor):
        """sudo commands should be blocked."""
        result = asyncio.run(executor.execute("sudo cat /etc/passwd"))
        assert result.blocked is True

    def test_allows_echo_variable(self, executor):
        """echo $VAR should be allowed."""
        result = asyncio.run(executor.execute("echo $HOME"))
        assert result.blocked is False
        # On Windows, this may not expand but shouldn't error
        assert result.exit_code == 0 or result.exit_code == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_executes_ls_command(self, executor):
        """ls commands should execute successfully."""
        result = asyncio.run(executor.execute("ls /tmp"))
        assert result.blocked is False
        assert result.exit_code == 0

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_executes_date_command(self, executor):
        """date command should execute successfully."""
        result = asyncio.run(executor.execute("date"))
        assert result.blocked is False
        assert result.exit_code == 0
        assert result.stdout  # Should have some output

    def test_timeout_setting(self, executor):
        """Timeout should be configurable."""
        executor.timeout = 1  # 1 second timeout
        assert executor.timeout == 1

    def test_output_truncation(self, executor):
        """Large outputs should be truncated."""
        executor.max_output = 100

        # Test the truncation logic directly
        long_text = "x" * 200
        truncated = executor._decode_and_truncate(long_text.encode())
        assert len(truncated) <= 150  # 100 + truncation message
        assert "TRUNCATED" in truncated


class TestCommandExecutorInit:
    """Test CommandExecutor initialization."""

    def test_default_timeout(self):
        """Default timeout should be reasonable."""
        executor = CommandExecutor()
        assert executor.timeout <= 120

    def test_max_timeout_cap(self):
        """Timeout should be capped at 120 seconds."""
        executor = CommandExecutor(timeout=300)
        assert executor.timeout == 120

    def test_custom_working_dir(self):
        """Working directory can be customized."""
        executor = CommandExecutor(working_dir="/tmp")
        assert executor.working_dir == "/tmp"
