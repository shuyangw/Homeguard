# Start Backend
Write-Host "Starting Backend on port 8000..." -ForegroundColor Green
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m uvicorn src.web.backend.main:app --host 0.0.0.0 --port 8000 --reload"

# Start Frontend
Write-Host "Starting Frontend on port 5173..." -ForegroundColor Green
Set-Location "src/web/frontend"
Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run dev"
Set-Location "../../.."

Write-Host "Homeguard Web UI is starting!" -ForegroundColor Cyan
Write-Host "Backend: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Gray
