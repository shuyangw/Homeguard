#!/bin/bash
# Start the Homeguard Discord bot (Linux/macOS)

cd "$(dirname "$0")/.."

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No venv found, using system Python"
fi

python -m src.discord_bot.main
