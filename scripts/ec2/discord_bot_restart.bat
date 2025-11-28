@echo off
REM Restart Homeguard Discord Bot
REM Windows Batch Script

echo ========================================
echo Restarting Homeguard Discord Bot
echo ========================================
echo.

ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo systemctl restart homeguard-discord"

echo.
echo Waiting 5 seconds for service to start...
timeout /t 5 /nobreak > nul

echo.
echo ========================================
echo Current Status:
echo ========================================
echo.

ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo systemctl status homeguard-discord --no-pager"

pause
