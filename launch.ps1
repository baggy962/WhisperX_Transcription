# ============================================================================
# Whisper Real-Time Transcriber - Launch Script (PowerShell)
# For Windows 11 with Miniconda
# ============================================================================

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Whisper Real-Time Transcriber" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCommand) {
    Write-Host "[ERROR] Miniconda/Anaconda not found in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run setup.bat or setup.ps1 first to configure the environment."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Environment name
$ENV_NAME = "whisper-transcriber"

Write-Host "[INFO] Checking environment '$ENV_NAME'..." -ForegroundColor Cyan
$envExists = conda env list | Select-String -Pattern $ENV_NAME -Quiet

if (-not $envExists) {
    Write-Host "[ERROR] Environment '$ENV_NAME' not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run setup.bat or setup.ps1 first to create the environment."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Activating environment..." -ForegroundColor Cyan
conda activate $ENV_NAME
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate environment!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Starting Whisper Real-Time Transcriber..." -ForegroundColor Green
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Controls:" -ForegroundColor White
Write-Host "  - Hotkey: Ctrl+F9 to toggle recording" -ForegroundColor Cyan
Write-Host "  - Model selection: Choose between General and Medical modes" -ForegroundColor Cyan
Write-Host "  - Device: Auto-detects GPU/CPU" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if the Python script exists
if (-not (Test-Path "realtime_transcriber_cross_platform.py")) {
    Write-Host "[ERROR] realtime_transcriber_cross_platform.py not found!" -ForegroundColor Red
    Write-Host "Please ensure this script is run from the project directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Launch the application
python realtime_transcriber_cross_platform.py

# Check if the application exited with an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Application exited with error code $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[INFO] Application closed successfully." -ForegroundColor Green
# Don't pause on normal exit to allow clean shutdown
