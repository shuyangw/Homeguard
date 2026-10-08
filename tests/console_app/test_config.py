import pytest

from tools.console.config import Settings, settings_from_env

FULL_ENV = {
    "EC2_INSTANCE_ID": "i-0123456789abcdef0",
    "EC2_REGION": "us-east-1",
    "CONSOLE_AGENT_URL": "https://agent.example.ts.net:8443",
    "CONSOLE_SNAPSHOT_BUCKET": "bucket",
}


def test_settings_come_from_the_environment():
    settings = settings_from_env(FULL_ENV)

    assert settings == Settings("i-0123456789abcdef0", "us-east-1", "https://agent.example.ts.net:8443", "bucket")
    assert settings.aws_profile == "homeguard-console"
    assert settings.port == 8765


def test_profile_can_be_overridden():
    assert settings_from_env({**FULL_ENV, "CONSOLE_AWS_PROFILE": "other"}).aws_profile == "other"


@pytest.mark.parametrize("value", ["", "  ", "<YOUR_BUCKET_NAME>"], ids=["empty", "blank", "placeholder"])
def test_missing_or_placeholder_values_are_all_named(value):
    env = {**FULL_ENV, "CONSOLE_SNAPSHOT_BUCKET": value}
    env.pop("CONSOLE_AGENT_URL")

    with pytest.raises(ValueError, match="CONSOLE_AGENT_URL, CONSOLE_SNAPSHOT_BUCKET"):
        settings_from_env(env)
