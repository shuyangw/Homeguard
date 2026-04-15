"""PreToolUse hook: block dangerous commands and enforce backtest output redirect."""
import json
import re
import sys

BLOCKED_PATTERNS = [
    (r"rm\s+-rf", "Recursive force delete"),
    (r"DROP\s+TABLE", "SQL table drop"),
    (r"git\s+reset\s+--hard", "Hard git reset"),
    (r"git\s+push.*--force", "Force push"),
    (r"del\s+/[sS]\s+/[qQ]", "Windows recursive delete"),
    (r"rmdir\s+/[sS]\s+/[qQ]", "Windows recursive rmdir"),
]

def main():
    hook_input = json.load(sys.stdin)
    tool_input = hook_input.get("tool_input", {})
    cmd = tool_input.get("command", "")

    if not cmd:
        sys.exit(0)

    # Block destructive commands
    for pattern, description in BLOCKED_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            # exit 2 = block the action; stderr message goes back to Claude
            print(f"BLOCKED: {description} detected in: {cmd}", file=sys.stderr)
            sys.exit(2)

    # Enforce backtest output redirect
    # If running a backtest script, must redirect output to file
    if re.search(r"python.*backtest", cmd, re.IGNORECASE):
        has_redirect = re.search(r">[> ]|2>&1|tee\s", cmd)
        if not has_redirect:
            print(
                "BLOCKED: Backtest commands must redirect output to a log file.\n"
                "Add '> logs/backtesting/run.log 2>&1' to the command.\n"
                "This prevents dumping verbose output into context.",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0)

if __name__ == "__main__":
    main()
