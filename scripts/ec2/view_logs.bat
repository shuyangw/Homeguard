@echo off
REM View Homeguard Trading Bot Live Logs
REM Windows Batch Script

echo ========================================
echo Viewing Live Trading Bot Logs
echo Press Ctrl+C to stop
echo ========================================
echo.

REM Use --output=cat to strip systemd formatting and colors for Windows CMD compatibility
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -f --output=cat"
