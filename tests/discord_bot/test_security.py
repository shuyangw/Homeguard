"""
Tests for Discord bot security module.

Tests the read-only command whitelist and blocked pattern detection.
"""

import pytest

from src.discord_bot.security import (
    validate_command,
    sanitize_output,
    is_channel_allowed,
    is_user_allowed,
)


class TestValidateCommand:
    """Test command validation against security rules."""

    def test_allowed_tail_command(self):
        """tail commands should be allowed."""
        result = validate_command("tail -100 /home/user/logs/test.log")
        assert result.is_allowed is True

    def test_allowed_cat_command(self):
        """cat commands should be allowed."""
        result = validate_command("cat /home/user/config.yaml")
        assert result.is_allowed is True

    def test_allowed_grep_command(self):
        """grep commands should be allowed."""
        result = validate_command("grep -i error /home/user/logs/*.log")
        assert result.is_allowed is True

    def test_allowed_systemctl_status(self):
        """systemctl status should be allowed."""
        result = validate_command("systemctl status homeguard-trading.service")
        assert result.is_allowed is True

    def test_allowed_journalctl(self):
        """journalctl should be allowed."""
        result = validate_command("journalctl -u homeguard-trading -n 50")
        assert result.is_allowed is True

    def test_allowed_ps_command(self):
        """ps commands should be allowed."""
        result = validate_command("ps aux | grep python")
        assert result.is_allowed is True

    def test_allowed_df_command(self):
        """df commands should be allowed."""
        result = validate_command("df -h")
        assert result.is_allowed is True

    def test_blocked_rm_command(self):
        """rm commands should be blocked."""
        result = validate_command("rm -rf /home/user/logs")
        assert result.is_allowed is False
        assert "rm" in result.reason.lower() or "blocked" in result.reason.lower()

    def test_blocked_sudo_command(self):
        """sudo commands should be blocked."""
        result = validate_command("sudo systemctl restart trading")
        assert result.is_allowed is False

    def test_blocked_systemctl_restart(self):
        """systemctl restart should be blocked."""
        result = validate_command("systemctl restart homeguard-trading.service")
        assert result.is_allowed is False

    def test_blocked_systemctl_stop(self):
        """systemctl stop should be blocked."""
        result = validate_command("systemctl stop homeguard-trading.service")
        assert result.is_allowed is False

    def test_blocked_output_redirect(self):
        """Output redirection should be blocked."""
        result = validate_command("cat file.txt > /tmp/output.txt")
        assert result.is_allowed is False

    def test_blocked_append_redirect(self):
        """Append redirection should be blocked."""
        result = validate_command("echo test >> /tmp/file.txt")
        assert result.is_allowed is False

    def test_blocked_pipe_to_rm(self):
        """Piping to rm should be blocked."""
        result = validate_command("find . -name '*.log' | xargs rm")
        assert result.is_allowed is False

    def test_blocked_mv_command(self):
        """mv commands should be blocked."""
        result = validate_command("mv /tmp/a /tmp/b")
        assert result.is_allowed is False

    def test_blocked_chmod_command(self):
        """chmod commands should be blocked."""
        result = validate_command("chmod 777 /home/user/file")
        assert result.is_allowed is False

    def test_blocked_unknown_command(self):
        """Unknown commands should be blocked."""
        result = validate_command("some_random_command --arg")
        assert result.is_allowed is False

    def test_blocked_kill_command(self):
        """kill commands should be blocked."""
        result = validate_command("kill -9 1234")
        assert result.is_allowed is False

    def test_blocked_pip_install(self):
        """pip install should be blocked."""
        result = validate_command("pip install requests")
        assert result.is_allowed is False

    def test_allowed_pip_list(self):
        """pip list should be allowed."""
        result = validate_command("pip list")
        assert result.is_allowed is True


class TestSanitizeOutput:
    """Test output sanitization for secrets."""

    def test_masks_api_key(self):
        """API keys should be masked."""
        text = "api_key=sk-123456789abcdef"
        result = sanitize_output(text)
        assert "sk-123456789" not in result
        assert "MASKED" in result

    def test_masks_password(self):
        """Passwords should be masked."""
        text = "password=supersecret123"
        result = sanitize_output(text)
        assert "supersecret" not in result
        assert "MASKED" in result

    def test_masks_aws_key(self):
        """AWS access keys should be masked."""
        text = "key=AKIAIOSFODNN7EXAMPLE"
        result = sanitize_output(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "AWS" in result

    def test_masks_anthropic_key(self):
        """Anthropic API keys should be masked."""
        text = "key=sk-ant-api03-abcdefghijklmnop"
        result = sanitize_output(text)
        assert "sk-ant-api03" not in result
        assert "ANTHROPIC" in result

    def test_preserves_normal_text(self):
        """Normal text should be preserved."""
        text = "Trading bot started successfully at 2024-01-15 09:00:00"
        result = sanitize_output(text)
        assert result == text

    def test_removes_ansi_codes(self):
        """ANSI escape codes should be removed."""
        text = "\x1b[32mSuccess\x1b[0m"
        result = sanitize_output(text)
        assert "\x1b" not in result
        assert "Success" in result


class TestChannelUserAllowlists:
    """Test channel and user allowlist checks."""

    def test_empty_allowed_channels_allows_all(self):
        """Empty allowed set means all channels are allowed."""
        assert is_channel_allowed(123, set()) is True
        assert is_channel_allowed(456, set()) is True

    def test_channel_in_allowlist(self):
        """Channel in allowlist should be allowed."""
        allowed = {123, 456}
        assert is_channel_allowed(123, allowed) is True
        assert is_channel_allowed(456, allowed) is True

    def test_channel_not_in_allowlist(self):
        """Channel not in allowlist should be blocked."""
        allowed = {123, 456}
        assert is_channel_allowed(789, allowed) is False

    def test_empty_allowed_users_allows_all(self):
        """Empty allowed set means all users are allowed."""
        assert is_user_allowed(123, set()) is True

    def test_user_in_allowlist(self):
        """User in allowlist should be allowed."""
        allowed = {123, 456}
        assert is_user_allowed(123, allowed) is True

    def test_user_not_in_allowlist(self):
        """User not in allowlist should be blocked."""
        allowed = {123, 456}
        assert is_user_allowed(789, allowed) is False
