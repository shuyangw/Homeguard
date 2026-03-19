@echo off
REM View Homeguard Discord Bot Logs
REM Windows Batch Script

REM Load EC2 configuration from .env
call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo Homeguard Discord Bot Logs (Live Stream)
echo ========================================
echo Press Ctrl+C to stop streaming
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "sudo journalctl -u homeguard-discord -f"

pause
