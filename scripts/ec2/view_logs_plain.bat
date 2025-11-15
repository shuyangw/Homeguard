@echo off
REM View Homeguard Trading Bot Live Logs (Plain Text - No Colors)
REM Windows Batch Script - For CMD compatibility

echo ========================================
echo Viewing Live Trading Bot Logs (Plain)
echo Press Ctrl+C to stop
echo ========================================
echo.

REM Strip ANSI color codes using sed on remote server
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -f --output=cat | sed 's/\x1b\[[0-9;]*m//g'"
