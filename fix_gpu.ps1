# ============================================================================
# Fix PyTorch GPU Support - Install CUDA-enabled PyTorch
# For CUDA 12.8 (compatible with cu121 build)
# ============================================================================

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  PyTorch GPU Support Installer" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCommand) {
    Write-Host "[ERROR] Miniconda/Anaconda not found in PATH!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$ENV_NAME = "whisper-transcriber"

Write-Host "[INFO] Checking environment..." -ForegroundColor Cyan
$envExists = conda env list | Select-String -Pattern $ENV_NAME -Quiet

if (-not $envExists) {
    Write-Host "[ERROR] Environment '$ENV_NAME' not found!" -ForegroundColor Red
    Write-Host "Please run setup.bat or setup.ps1 first." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Step 1/4] Checking current PyTorch installation..." -ForegroundColor Green
conda run -n $ENV_NAME python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Write-Host ""

Write-Host "[Step 2/4] Uninstalling CPU-only PyTorch..." -ForegroundColor Green
conda run -n $ENV_NAME pip uninstall torch torchvision torchaudio -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to uninstall PyTorch!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Old PyTorch removed" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 3/4] Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Green
Write-Host "[INFO] CUDA 12.8 is compatible with PyTorch cu121 build" -ForegroundColor Cyan
Write-Host "[INFO] This will download ~2.5GB..." -ForegroundColor Cyan
conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PyTorch with CUDA 12.1!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Trying CUDA 11.8 build instead..." -ForegroundColor Yellow
    conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Both CUDA builds failed!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "[OK] PyTorch with CUDA installed" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 4/4] Verifying GPU support..." -ForegroundColor Green
conda run -n $ENV_NAME python -c "import torch; print('='*60); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); cuda_available = torch.cuda.is_available(); print('='*60); print('GPU SUPPORT: ENABLED' if cuda_available else 'GPU SUPPORT: FAILED'); print('='*60); exit(0 if cuda_available else 1)"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] GPU support verification failed!" -ForegroundColor Yellow
    Write-Host "This might still work, but GPU may not be detected." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "[SUCCESS] GPU support is now enabled!" -ForegroundColor Green
}
Write-Host ""

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Installation Complete" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Close the app if it's running" -ForegroundColor Cyan
Write-Host "2. Restart the app: python realtime_transcriber_cross_platform.py" -ForegroundColor Cyan
Write-Host "3. You should now see GPU option in the device selection" -ForegroundColor Cyan
Write-Host ""
Write-Host "Expected output on startup:" -ForegroundColor White
Write-Host "  PyTorch version: 2.x.x" -ForegroundColor Cyan
Write-Host "  CUDA available: True" -ForegroundColor Green
Write-Host "  CUDA version: 12.1" -ForegroundColor Cyan
Write-Host "  GPU: [Your NVIDIA GPU Name]" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
