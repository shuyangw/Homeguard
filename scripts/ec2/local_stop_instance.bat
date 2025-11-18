@echo off
REM Stop Homeguard Trading Bot EC2 Instance
REM
REM This script stops the EC2 instance from your local Windows machine using AWS CLI.
REM Use this to manually stop the instance during off-hours.
REM
REM Prerequisites:
REM   - AWS CLI installed and configured
REM   - AWS credentials with EC2 permissions (ec2:StopInstances, ec2:DescribeInstances)
REM
REM Usage:
REM   scripts\ec2\local_stop_instance.bat
REM

setlocal enabledelayedexpansion

REM Configuration
set INSTANCE_ID=i-02500fe2392631ff2
set REGION=us-east-1

echo ==========================================
echo Stop Homeguard Trading Bot EC2 Instance
echo ==========================================
echo.
echo Instance ID: %INSTANCE_ID%
echo Region: %REGION%
echo.

REM Check if AWS CLI is installed
where aws >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] AWS CLI is not installed
    echo.
    echo Please install AWS CLI:
    echo   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
    exit /b 1
)

REM Check current instance state
echo Checking instance state...
for /f "tokens=*" %%a in ('aws ec2 describe-instances --instance-ids %INSTANCE_ID% --region %REGION% --query "Reservations[0].Instances[0].State.Name" --output text 2^>nul') do set INSTANCE_STATE=%%a

if "%INSTANCE_STATE%"=="" (
    echo [ERROR] Failed to get instance state
    echo.
    echo Please check:
    echo   1. AWS CLI is configured: aws configure
    echo   2. You have permissions to access instance %INSTANCE_ID%
    echo   3. Instance exists in region %REGION%
    exit /b 1
)

echo Current state: %INSTANCE_STATE%
echo.

REM Handle different states
if "%INSTANCE_STATE%"=="stopped" (
    echo [INFO] Instance is already stopped
    echo.
    echo To start the instance:
    echo   scripts\ec2\local_start_instance.bat
    exit /b 0
)

if "%INSTANCE_STATE%"=="stopping" (
    echo [INFO] Instance is already stopping...
    echo Waiting for instance to be stopped...
    echo.

    aws ec2 wait instance-stopped --instance-ids %INSTANCE_ID% --region %REGION%

    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Timeout waiting for instance to stop
        exit /b 1
    )

    echo [SUCCESS] Instance is now stopped
    exit /b 0
)

if "%INSTANCE_STATE%"=="running" (
    echo [WARNING] This will stop the trading bot and shut down the instance.
    echo.
    set /p CONFIRM="Are you sure you want to stop the instance? (y/N): "

    if /i not "!CONFIRM!"=="y" (
        echo.
        echo Operation cancelled.
        exit /b 0
    )

    echo.
    echo Stopping instance...

    aws ec2 stop-instances --instance-ids %INSTANCE_ID% --region %REGION% >nul

    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to stop instance
        exit /b 1
    )

    echo [SUCCESS] Instance stop command sent
    echo.
    echo Waiting for instance to stop (this may take 1-2 minutes)...

    aws ec2 wait instance-stopped --instance-ids %INSTANCE_ID% --region %REGION%

    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Timeout waiting for instance to stop
        exit /b 1
    )

    echo.
    echo ==========================================
    echo Instance Stopped Successfully!
    echo ==========================================
    echo.
    echo To start the instance again:
    echo   scripts\ec2\local_start_instance.bat
    echo.

) else (
    echo [WARNING] Instance is in state: %INSTANCE_STATE%
    echo Cannot stop instance in this state
    exit /b 1
)

endlocal
