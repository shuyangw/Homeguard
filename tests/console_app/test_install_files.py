import plistlib
from pathlib import Path

CONSOLE = Path(__file__).resolve().parents[2] / "tools" / "console"
REPO_ROOT = CONSOLE.parents[1]


def test_launchd_plist_runs_the_console_module():
    plist = plistlib.loads((CONSOLE / "com.homeguard.console.plist").read_bytes())

    assert plist["Label"] == "com.homeguard.console"
    assert plist["ProgramArguments"][1:] == ["-m", "tools.console"]
    assert plist["RunAtLoad"] is True


def test_requirements_list_the_app_dependencies():
    names = {line.split("==")[0].split(">=")[0].strip() for line in (CONSOLE / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")}

    assert {"fastapi", "uvicorn", "jinja2", "httpx", "boto3", "python-dotenv", "pandas_market_calendars"} <= names


def test_env_example_has_placeholders_for_the_console():
    text = (REPO_ROOT / ".env.example").read_text()

    assert "CONSOLE_AGENT_URL=<YOUR_" in text
    assert "CONSOLE_SNAPSHOT_BUCKET=<YOUR_" in text
