# Starts the whisper STT service (:8001) in a new window, waits for it to be
# ready, then starts the main API (:8000) in this window.
#
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File start_all.ps1
#   powershell -ExecutionPolicy Bypass -File start_all.ps1 -BindHost 0.0.0.0   # share on the LAN
#
# Prefer the justfile:  `just all` (local)  /  `just share` (LAN).

param(
    # Address the MAIN api binds to. 0.0.0.0 exposes it to your network.
    # Named BindHost, not Host: $Host is a reserved PowerShell automatic variable.
    [string]$BindHost = "127.0.0.1",
    [int]$MainPort = 8000,
    [int]$SttPort = 8001
)

$ErrorActionPreference = "Stop"
# Two venvs on purpose: Leva-TTS pins transformers<5, the Cohere STT model needs
# transformers>=5.4. Each service must run on its OWN python.
$py    = ".\myenv\Scripts\python.exe"    # main API + Leva-TTS
$pyStt = ".\sttenv\Scripts\python.exe"   # STT service

function Test-SttReady {
    try {
        return (Invoke-WebRequest -Uri "http://127.0.0.1:$SttPort/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200
    } catch {
        return $false
    }
}

# The STT service always stays on loopback: only the main API calls it, and it
# holds a ~4 GB model we don't want reachable from the network.
if (Test-SttReady) {
    # Reuse the one already running (e.g. you restarted only the main API):
    # starting a second would just fail to bind :$SttPort and reload the model.
    Write-Host "Whisper STT service already running on :$SttPort - reusing it." -ForegroundColor Cyan
} else {
    Write-Host "Starting Whisper STT service on 127.0.0.1:$SttPort (new window)..." -ForegroundColor Cyan
    Start-Process -FilePath $pyStt -ArgumentList "-m", "uvicorn", "whisper_service:app", "--port", "$SttPort"

    Write-Host "Waiting for the Whisper service to finish loading its model..." -ForegroundColor Cyan
    do {
        Start-Sleep -Seconds 3
    } until (Test-SttReady)
}

Write-Host "Whisper service ready. Starting main API on ${BindHost}:$MainPort..." -ForegroundColor Green
& $py -m uvicorn main:app --host $BindHost --port $MainPort
