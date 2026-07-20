# Patient Simulator — task runner.
#
# The project needs TWO venvs: Leva-TTS pins transformers<5, the Cohere STT model
# needs transformers>=5.4. Every recipe below calls the right venv's python
# directly, so you never have to activate anything.
#
#   just              list all recipes
#   just main         main API on :8000
#   just stt          STT service on :8001
#   just all          both (STT in a new window, then main here)

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true   # LLM_PROVIDER, the API keys etc. come from .env (gitignored)

py_main := 'myenv\Scripts\python.exe'
py_stt  := 'sttenv\Scripts\python.exe'

main_port := '8000'
stt_port  := '8001'
host      := '127.0.0.1'   # `just share` overrides this with 0.0.0.0

# List available recipes
default:
    @just --list --unsorted

# --- run ---------------------------------------------------------------------

# Main API on :8000 (loads Leva-TTS onto the GPU at startup)
main:
    & '{{py_main}}' -m uvicorn main:app --host {{host}} --port {{main_port}}

# Main API with auto-reload (slow: reloads the TTS model on every edit)
main-reload:
    & '{{py_main}}' -m uvicorn main:app --reload --host {{host}} --port {{main_port}}

# BOTH services, with the main API open to the LAN so others can connect
share:
    @echo ""
    @echo "Give your friend the Wi-Fi / Ethernet address (ignore WSL, hotspot and VPN ones):"
    @Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Sort-Object InterfaceAlias | ForEach-Object { echo "   http://$($_.IPAddress):{{main_port}}/docs   [$($_.InterfaceAlias)]" }
    @echo ""
    @just host=0.0.0.0 all

# Main API only, open to the LAN (use when the STT service is already running)
share-main:
    @just host=0.0.0.0 main

# STT service on :8001 (Cohere Transcribe Arabic) — start this BEFORE main
stt:
    & '{{py_stt}}' -m uvicorn whisper_service:app --port {{stt_port}}

# Both services: STT in a new window, wait for its model, then main here
all:
    powershell -ExecutionPolicy Bypass -File start_all.ps1 -BindHost {{host}} -MainPort {{main_port}} -SttPort {{stt_port}}

# --- setup -------------------------------------------------------------------

# Create both venvs (does not install anything — run `just install` next)
venvs:
    uv venv myenv --python 3.12
    uv venv sttenv --python 3.12

# Install/update dependencies in both venvs
install: install-main install-stt

# Install the main API + Leva-TTS deps into myenv
install-main:
    uv pip install --python '{{py_main}}' -r requirements.txt

# Install the STT deps into sttenv (torch comes from the CUDA 12.1 index)
install-stt:
    uv pip install --python '{{py_stt}}' torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    uv pip install --python '{{py_stt}}' -r requirements-stt.txt

# Download the gated Cohere STT model (needs `hf auth login` + accepted license)
model-stt:
    $env:HF_HUB_DISABLE_XET = '1'; & '{{py_stt}}' -m huggingface_hub.commands.huggingface_cli download CohereLabs/cohere-transcribe-arabic-07-2026

# --- checks ------------------------------------------------------------------

# Ping both services' /health
health:
    @try { echo "main :{{main_port}} -> $(Invoke-RestMethod http://127.0.0.1:{{main_port}}/health -TimeoutSec 3 | ConvertTo-Json -Compress)" } catch { echo "main :{{main_port}} -> DOWN" }
    @try { echo "stt  :{{stt_port}} -> $(Invoke-RestMethod http://127.0.0.1:{{stt_port}}/health -TimeoutSec 3 | ConvertTo-Json -Compress)" } catch { echo "stt  :{{stt_port}} -> DOWN" }

# List the scenarios and investigation categories the API serves
catalog:
    @try { Invoke-RestMethod http://127.0.0.1:{{main_port}}/scenarios -TimeoutSec 3 | ConvertTo-Json -Compress } catch { echo "main API not running on :{{main_port}}" }
    @try { Invoke-RestMethod http://127.0.0.1:{{main_port}}/test-categories -TimeoutSec 3 | ConvertTo-Json -Compress } catch { echo "" }

# Check that main.py imports and its JSON data files parse (no server, no GPU)
check:
    & '{{py_main}}' -c "import json; [json.load(open(f, encoding='utf-8')) for f in ('scenarios.json','test_categories.json','tests.json')]; print('json ok')"
    & '{{py_main}}' -m compileall -q main.py whisper_service.py app; if ($?) { echo "syntax ok" }

# Show which python each venv uses and whether the API key is set
info:
    @echo "main venv: {{py_main}}"
    @& '{{py_main}}' --version
    @echo "stt venv:  {{py_stt}}"
    @& '{{py_stt}}' --version
    @& '{{py_main}}' -c "from app import config as c; print('llm chain: ' + (' -> '.join(str(p) for p in c.CHAIN) or 'EMPTY - no keys set, see .env.example')); print('no key for: ' + (', '.join(c.MISSING_KEYS) or '-'))"

# List the models each provider in the chain can actually reach
models:
    @& '{{py_main}}' -c "import requests; from app import config as c; [print(f'--- {p.name} ---') or [print(' ', m['id']) for m in sorted(requests.get(p.url.replace('/chat/completions','/models'), headers={'Authorization': f'Bearer {p.key}'}, timeout=30).json().get('data', []), key=lambda m: m['id'])] for p in c.CHAIN]"

# Send one prompt through the chain (no GPU, no TTS); warnings show any failover
llm-ping:
    @& '{{py_main}}' -c "import logging; logging.basicConfig(level=logging.WARNING); from app import config as c, llm; print('chain: ' + ' -> '.join(str(p) for p in c.CHAIN)); print('reply: ' + llm.call_llm([{'role':'user','content':'قل مرحبا بكلمة واحدة'}], 32, 0.0))"

# --- housekeeping ------------------------------------------------------------

# Remove __pycache__ and generated wav files
clean:
    -Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
    -Remove-Item -Force patient_reply.wav -ErrorAction SilentlyContinue
