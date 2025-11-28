@echo off
REM Start the Homeguard Discord bot (Windows)

cd /d "%~dp0.."
call conda activate fintech
python -m src.discord_bot.main
