@echo off
REM Daily Health Check for Homeguard Trading Bot
REM Quick automated health verification

echo ========================================
echo Homeguard Trading Bot Health Check
echo Date: %date% %time%
echo ========================================
echo.

echo [1/6] Instance State:
aws ec2 describe-instances --instance-ids i-02500fe2392631ff2 --query "Reservations[0].Instances[0].State.Name" --output text
echo.

echo [2/6] Bot Service Status:
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@100.30.95.146 "sudo systemctl is-active homeguard-trading"
echo.

echo [3/6] Recent Errors (last hour):
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -p err --since '1 hour ago' --no-pager | wc -l"
echo.

echo [4/6] Resource Usage:
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo systemctl status homeguard-trading --no-pager | grep -E 'Memory|CPU'"
echo.

echo [5/6] Last Activity:
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -n 3 --no-pager"
echo.

echo [6/6] Current Market Status:
ssh -i "%USERPROFILE%\.ssh\homeguard-trading.pem" -o StrictHostKeyChecking=no ec2-user@100.30.95.146 "sudo journalctl -u homeguard-trading -n 1 --no-pager | grep -oP 'Market: \K[A-Z]+'"
echo.

echo ========================================
echo Health Check Complete
echo ========================================
echo.
echo For live monitoring, run:
echo   scripts\ec2\view_logs.bat
echo.

pause
