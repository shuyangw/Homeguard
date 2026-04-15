"""PostToolUse hook: auto-lint Python files after Write/Edit/MultiEdit."""
import json
import subprocess
import sys
import os

def get_python():
    """Get the fintech conda python path for the current OS."""
    if sys.platform == "win32":
        return r"C:\Users\qwqw1\anaconda3\envs\fintech\python.exe"
    else:
        # macOS/Linux: use conda run
        return None  # signals to use conda run

def run_tool(tool, args, file_path):
    """Run a linting tool, cross-platform."""
    python = get_python()
    if python:
        # Windows: call tool module directly
        cmd = [python, "-m", tool] + args + [file_path]
    else:
        # macOS/Linux: use conda run
        cmd = ["conda", "run", "-n", "fintech", tool] + args + [file_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

def main():
    hook_input = json.load(sys.stdin)

    # Extract file path from hook JSON
    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path") or tool_input.get("path", "")

    # Only lint Python files
    if not file_path.endswith(".py"):
        sys.exit(0)

    # Skip if file doesn't exist (was deleted)
    if not os.path.exists(file_path):
        sys.exit(0)

    # Run ruff check --fix (auto-fix lint issues)
    run_tool("ruff", ["check", "--fix"], file_path)

    # Run ruff format
    run_tool("ruff", ["format"], file_path)

    # Run mypy (non-blocking, just feedback via stdout)
    result = run_tool("mypy", ["--ignore-missing-imports"], file_path)
    if result and result.returncode != 0:
        errors = [l for l in result.stdout.splitlines() if "error:" in l]
        if errors:
            # Print to stdout — Claude sees this as feedback
            print(f"mypy found {len(errors)} type error(s) in {file_path}:")
            for e in errors[:5]:
                print(f"  {e}")

    sys.exit(0)

if __name__ == "__main__":
    main()
