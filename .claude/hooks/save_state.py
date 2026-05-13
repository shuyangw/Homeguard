"""StopFailure hook: save recovery breadcrumb when session dies (rate limit, API error)."""
import json
import sys
import os
from datetime import datetime

def main():
    log_path = os.path.join(os.getcwd(), ".claude-recovery.log")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read hook input for failure details
    try:
        hook_input = json.load(sys.stdin)
        reason = hook_input.get("error", "unknown")
    except Exception:
        reason = "unknown"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Session interrupted: {reason}\n")
        f.write(f"[{timestamp}] Check TODO.md for current pipeline state\n")
        f.write(f"[{timestamp}] Resume with: claude --agent strategy-lead --continue\n")
        f.write("---\n")

    sys.exit(0)

if __name__ == "__main__":
    main()
