# Registers a logon task that runs the Homeguard Console on http://127.0.0.1:8765.
#   powershell -File tools\console\install_windows_task.ps1 -Python C:\path\to\env\pythonw.exe
param([Parameter(Mandatory = $true)][string]$Python)
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
& $Python -c "import fastapi, uvicorn, jinja2, httpx, boto3"
if ($LASTEXITCODE -ne 0) { throw "Missing dependencies; run: $Python -m pip install -r $repo\tools\console\requirements.txt" }

$action = New-ScheduledTaskAction -Execute $Python -Argument "-m tools.console" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "HomeguardConsole" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Homeguard Console (read-only) on 127.0.0.1:8765" -Force | Out-Null
Start-ScheduledTask -TaskName "HomeguardConsole"
Write-Output "[+] HomeguardConsole task registered and started; open http://127.0.0.1:8765"
