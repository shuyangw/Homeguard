@echo off
REM Stream IB Gateway logs on EC2

call "%~dp0load_env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo IB Gateway Logs (Ctrl+C to stop)
echo ========================================
echo.

ssh -i "%EC2_SSH_KEY_PATH%" %EC2_USER%@%EC2_IP% "journalctl -u homeguard-gateway -f --no-pager"
