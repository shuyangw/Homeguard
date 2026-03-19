#!/bin/bash
#
# Update repository on EC2 instance
# Run this script ON the EC2 instance to pull latest code changes
#
# Usage:
#   ./instance_update_repo.sh           # Update code only
#   ./instance_update_repo.sh --restart # Update code and restart bot
#

set -e  # Exit on error

# Save current directory to return to it at the end
ORIGINAL_DIR="$(pwd)"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESTART_BOT=false

# Parse arguments
if [ "$1" == "--restart" ]; then
    RESTART_BOT=true
fi

echo "=========================================="
echo "Repository Update Script"
echo "=========================================="
echo ""

# Check if we're in a git repository
cd "$REPO_DIR"
if [ ! -d .git ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

echo "Repository: $REPO_DIR"
echo "Current branch: $(git branch --show-current)"
echo ""

# Check if trading bot is running
BOT_RUNNING=false
if pgrep -f "run_live_paper_trading.py" > /dev/null; then
    BOT_RUNNING=true
    echo "⚠️  Trading bot is currently RUNNING"
    echo ""
fi

# Show current status
echo "Current status:"
git status --short
echo ""

# Stash any local changes (shouldn't be any, but just in case)
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Found local changes - stashing them..."
    git stash save "Auto-stash before update $(date +'%Y-%m-%d %H:%M:%S')"
    echo ""
fi

# Get current commit hash before pulling
BEFORE_COMMIT=$(git rev-parse HEAD)

# Pull latest changes
echo "Pulling latest changes from origin..."
PULL_OUTPUT=$(git pull --ff-only origin $(git branch --show-current) 2>&1)
PULL_EXIT_CODE=$?

if [ $PULL_EXIT_CODE -ne 0 ]; then
    echo "$PULL_OUTPUT"
    echo ""
    echo "❌ Git pull failed!"
    echo "You may need to resolve conflicts manually"
    exit 1
fi

# Get commit hash after pulling
AFTER_COMMIT=$(git rev-parse HEAD)

# Check if any changes were pulled
CHANGES_PULLED=false
if [ "$BEFORE_COMMIT" != "$AFTER_COMMIT" ]; then
    CHANGES_PULLED=true
fi

echo "$PULL_OUTPUT"
echo ""

if [ "$CHANGES_PULLED" = true ]; then
    echo "✅ New changes pulled!"
    echo ""
    echo "What changed:"
    git log --oneline $BEFORE_COMMIT..$AFTER_COMMIT
    echo ""
else
    echo "✅ Already up to date (no changes)"
    echo ""
fi

# Handle bot restart if requested
if [ "$RESTART_BOT" = true ]; then
    if [ "$BOT_RUNNING" = true ]; then
        if [ "$CHANGES_PULLED" = true ]; then
            echo "Restarting trading bot with new code..."
            echo ""

            # Restart using systemd (bot is managed by systemd service)
            sudo systemctl restart homeguard-trading
            echo "✅ Bot restarted via systemd"
        else
            echo "ℹ️  No changes pulled - skipping restart"
        fi
    else
        echo "ℹ️  Bot was not running - skipping restart"
    fi
elif [ "$BOT_RUNNING" = true ] && [ "$CHANGES_PULLED" = true ]; then
    echo "⚠️  Trading bot is still running with OLD code"
    echo "   Run './instance_update_repo.sh --restart' to restart with new code"
    echo "   Or manually restart: sudo systemctl restart homeguard-trading"
fi

echo ""
echo "=========================================="
echo "Update complete!"
echo "=========================================="

# Return to original directory
cd "$ORIGINAL_DIR"
