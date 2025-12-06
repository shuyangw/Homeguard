@echo off
REM ============================================================================
REM Load environment variables from .env file
REM
REM This helper script parses the root .env file and sets environment variables
REM for use in Windows batch scripts. Call this at the start of EC2 scripts.
REM
REM Usage: call "%~dp0load_env.bat"
REM ============================================================================

setlocal enabledelayedexpansion

REM Find project root (two levels up from scripts/ec2/)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."

REM Check if .env file exists
if not exist "%PROJECT_ROOT%\.env" (
    echo ERROR: .env file not found at %PROJECT_ROOT%\.env
    echo.
    echo Please create .env from the template:
    echo   copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env"
    echo.
    echo Then edit .env with your EC2 instance details:
    echo   EC2_IP=your_instance_ip
    echo   EC2_INSTANCE_ID=your_instance_id
    echo   EC2_REGION=us-east-1
    echo   EC2_SSH_KEY_PATH=path_to_your_pem_file
    echo   EC2_USER=ec2-user
    exit /b 1
)

REM Parse .env file and set variables
REM This handles KEY="value" and KEY=value formats
for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_ROOT%\.env") do (
    REM Skip empty lines and comments
    set "line=%%a"
    if defined line (
        REM Check if line starts with #
        set "first_char=!line:~0,1!"
        if not "!first_char!"=="#" (
            REM Remove quotes from value if present
            set "value=%%b"
            if defined value (
                REM Remove leading/trailing quotes
                set "value=!value:"=!"
                REM Set the environment variable (endlocal will export it)
                set "%%a=!value!"
            )
        )
    )
)

REM Validate required EC2 variables
if not defined EC2_IP (
    echo ERROR: EC2_IP not set in .env file
    exit /b 1
)
if "%EC2_IP%"=="<YOUR_EC2_IP>" (
    echo ERROR: EC2_IP is still set to placeholder value
    echo Please edit .env and set your actual EC2 IP address
    exit /b 1
)

if not defined EC2_INSTANCE_ID (
    echo ERROR: EC2_INSTANCE_ID not set in .env file
    exit /b 1
)
if "%EC2_INSTANCE_ID%"=="<YOUR_INSTANCE_ID>" (
    echo ERROR: EC2_INSTANCE_ID is still set to placeholder value
    echo Please edit .env and set your actual EC2 instance ID
    exit /b 1
)

REM Set defaults if not specified
if not defined EC2_REGION set "EC2_REGION=us-east-1"
if not defined EC2_USER set "EC2_USER=ec2-user"
if not defined EC2_SSH_KEY_PATH set "EC2_SSH_KEY_PATH=%USERPROFILE%\.ssh\homeguard-trading.pem"

REM Expand ~ to %USERPROFILE% for SSH key path
set "EC2_SSH_KEY_PATH=!EC2_SSH_KEY_PATH:~=%USERPROFILE%!"

REM Export variables by ending local scope with preservation
endlocal & (
    set "EC2_IP=%EC2_IP%"
    set "EC2_INSTANCE_ID=%EC2_INSTANCE_ID%"
    set "EC2_REGION=%EC2_REGION%"
    set "EC2_USER=%EC2_USER%"
    set "EC2_SSH_KEY_PATH=%EC2_SSH_KEY_PATH%"
)
