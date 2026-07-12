# Starts the whisper STT service (:8001) in a new window, waits for it to be
# ready, then starts the main API (:8000) in this window.
#
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File start_all.ps1

$ErrorActionPreference = "Stop"
$py = ".\myenv\Scripts\python.exe"

Write-Host "Starting Whisper STT service on :8001 (new window)..." -ForegroundColor Cyan
Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "whisper_service:app", "--port", "8001"

Write-Host "Waiting for the Whisper service to finish loading its model..." -ForegroundColor Cyan
do {
    Start-Sleep -Seconds 3
    try {
        $code = (Invoke-WebRequest -Uri "http://127.0.0.1:8001/health" -UseBasicParsing -TimeoutSec 3).StatusCode
        $ready = ($code -eq 200)
    } catch {
        $ready = $false
    }
} until ($ready)

Write-Host "Whisper service ready. Starting main API on :8000..." -ForegroundColor Green
& $py -m uvicorn main:app --port 8000
