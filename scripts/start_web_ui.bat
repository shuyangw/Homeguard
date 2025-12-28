@echo off
setlocal

:: Get project root
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Set Python path to fintech conda environment
set "PYTHON_EXE=C:\Users\qwqw1\anaconda3\envs\fintech\python.exe"

echo ============================================
echo   Homeguard Web UI
echo ============================================
echo.

:: Start backend in background (same window)
echo [Backend] Starting on http://localhost:8000...
start /b %PYTHON_EXE% -m uvicorn src.web.backend.main:app --host 0.0.0.0 --port 8000 --reload

:: Brief pause to let backend initialize
timeout /t 2 /nobreak >nul

:: Start frontend in background (same window)
echo [Frontend] Starting on http://localhost:5173...
cd src\web\frontend
start /b npm run dev

echo.
echo ============================================
echo   Backend:  http://localhost:8000/docs
echo   Frontend: http://localhost:5173
echo ============================================
echo.
echo Press Ctrl+C to stop both servers.
echo.

:: Keep the window open and wait for Ctrl+C
cmd /k
