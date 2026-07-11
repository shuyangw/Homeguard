#!/usr/bin/env python3
"""PreToolUse (Bash) hook: hard-block strategy backtest/gate/verdict commands
unless the strategy-lead sentinel exists.

Rationale: CLAUDE.md requires strategy testing to be orchestrated by the
strategy-lead agent (not run ad hoc). This hook enforces that: any Bash command
matching a strategy-verdict pattern is DENIED unless .claude/.strategy-lead-active
exists (strategy-lead creates it while it owns a testing phase, removes it after).
Only actual backtest/gate/verdict commands trip it -- file edits, unit-test
pytest, and data builds pass through untouched (that is the "only if testing
strategies" scoping).
"""
import json
import os
import re
import sys

# Actual strategy-verdict entry points (runners, gates, and the sp_* smoke paths).
# Deliberately NOT matching bare pytest, edits, or data builds.
_PATTERNS = re.compile(
    r"backtest_runner"
    r"|run_futures_backtest"
    r"|walk_forward"
    r"|run_carver_walkforward"
    r"|run_fx_walkforward"
    r"|gate_return_stream"
    r"|gate_convergence"
    r"|gate_session_stream"
    r"|run_vix_rolldown"
    r"|run_vrp"
    r"|run_standard_report"
    r"|scripts/backtest_scripts/sp_"
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return  # unparseable input -> do not block

    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    if not _PATTERNS.search(cmd):
        return  # not a strategy-verdict command -> allow

    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    sentinel = os.path.join(root, ".claude", ".strategy-lead-active")
    if os.path.exists(sentinel):
        return  # strategy-lead owns a testing phase -> allow

    reason = (
        "BLOCKED: strategy backtest/gate/verdict command detected outside "
        "strategy-lead. Per CLAUDE.md, strategy testing MUST be orchestrated by "
        "the strategy-lead agent, which creates the .claude/.strategy-lead-active "
        "sentinel while it owns a testing phase. Invoke strategy-lead to run this "
        "instead of running the backtest/gate directly. (If you ARE strategy-lead "
        "or its backtest-driver, create the sentinel first: "
        "touch .claude/.strategy-lead-active)."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))


if __name__ == "__main__":
    main()
