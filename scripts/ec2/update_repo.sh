#!/bin/bash
#
# Update repository on EC2 instance
# Run this script ON the EC2 instance to pull latest code changes
#
# Usage:
#   ./update_repo.sh           # Update code only
#   ./update_repo.sh --restart # Update code and restart bot
#

set -e  # Exit on error

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

# Pull latest changes
echo "Pulling latest changes from origin..."
git pull --ff-only origin $(git branch --show-current)

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Git pull failed!"
    echo "You may need to resolve conflicts manually"
    exit 1
fi

echo ""
echo "✅ Repository updated successfully!"
echo ""

# Show what changed
echo "Recent commits:"
git log --oneline -5
echo ""

# Handle bot restart if requested
if [ "$RESTART_BOT" = true ]; then
    if [ "$BOT_RUNNING" = true ]; then
        echo "Restarting trading bot..."
        echo ""

        # Use the restart script if it exists
        if [ -f "$SCRIPT_DIR/restart_bot.sh" ]; then
            "$SCRIPT_DIR/restart_bot.sh"
        else
            echo "⚠️  restart_bot.sh not found - please restart manually"
        fi
    else
        echo "ℹ️  Bot was not running - skipping restart"
    fi
elif [ "$BOT_RUNNING" = true ]; then
    echo "⚠️  Trading bot is still running with OLD code"
    echo "   Run './update_repo.sh --restart' to restart with new code"
    echo "   Or manually restart: ./restart_bot.sh"
fi

echo ""
echo "=========================================="
echo "Update complete!"
echo "=========================================="
