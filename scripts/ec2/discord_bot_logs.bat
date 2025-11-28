@echo off
REM View Homeguard Discord Bot Logs
REM Windows Batch Script

echo ========================================
echo Homeguard Discord Bot Logs (Live Stream)
echo ========================================
echo Press Ctrl+C to stop streaming
echo.

ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo journalctl -u homeguard-discord -f"

pause
