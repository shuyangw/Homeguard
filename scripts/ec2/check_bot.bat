@echo off
REM Check Homeguard Trading Bot Status
REM Windows Batch Script

echo ========================================
echo Checking Homeguard Trading Bot Status
echo ========================================
echo.

ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo systemctl status homeguard-trading --no-pager"

echo.
echo ========================================
echo Recent Activity (last 10 lines):
echo ========================================
echo.

ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -n 10 --no-pager"

pause
