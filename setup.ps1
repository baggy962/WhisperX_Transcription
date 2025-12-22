# ============================================================================
# Whisper Real-Time Transcriber - Environment Setup Script (PowerShell)
# For Windows 11 with Miniconda
# Python 3.11
# ============================================================================

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Whisper Real-Time Transcriber - Environment Setup" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCommand) {
    Write-Host "[ERROR] Miniconda/Anaconda not found in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure Miniconda is installed and added to PATH."
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Step 1/7] Checking conda installation..." -ForegroundColor Green
conda --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Conda command failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Conda found" -ForegroundColor Green
Write-Host ""

# Environment name
$ENV_NAME = "whisper-transcriber"

Write-Host "[Step 2/7] Checking if environment '$ENV_NAME' exists..." -ForegroundColor Green
$envExists = conda env list | Select-String -Pattern $ENV_NAME -Quiet

if ($envExists) {
    Write-Host "[WARNING] Environment '$ENV_NAME' already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove and recreate it? (Y/N)"
    if ($response -ne 'Y' -and $response -ne 'y') {
        Write-Host "Setup cancelled."
        Read-Host "Press Enter to exit"
        exit 0
    }
    Write-Host "Removing existing environment..."
    conda deactivate 2>$null
    conda env remove -n $ENV_NAME -y
}
Write-Host ""

Write-Host "[Step 3/7] Creating conda environment with Python 3.11..." -ForegroundColor Green
conda create -n $ENV_NAME python=3.11 -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create conda environment!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Environment created" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 4/7] Activating environment..." -ForegroundColor Green
# Note: In PowerShell, we need to use conda activate differently
conda activate $ENV_NAME
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate environment!" -ForegroundColor Red
    Write-Host "[INFO] You may need to initialize conda for PowerShell:" -ForegroundColor Yellow
    Write-Host "       conda init powershell" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Environment activated" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 5/7] Checking for CUDA availability..." -ForegroundColor Green
$nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvccCommand) {
    Write-Host "[OK] CUDA detected - Will install PyTorch with GPU support" -ForegroundColor Green
    $CUDA_AVAILABLE = $true
    nvcc --version | Select-String "release"
} else {
    Write-Host "[INFO] CUDA not detected - Will install CPU-only PyTorch" -ForegroundColor Yellow
    Write-Host "[INFO] For GPU support, install CUDA Toolkit first:" -ForegroundColor Yellow
    Write-Host "       https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    $CUDA_AVAILABLE = $false
}
Write-Host ""

Write-Host "[Step 6/7] Installing PyTorch..." -ForegroundColor Green
if ($CUDA_AVAILABLE) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing PyTorch (CPU only)..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PyTorch!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] PyTorch installed" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 7/7] Installing application dependencies..." -ForegroundColor Green
if (-not (Test-Path "requirements.txt")) {
    Write-Host "[ERROR] requirements.txt not found!" -ForegroundColor Red
    Write-Host "Please ensure this script is run from the project directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] All dependencies installed" -ForegroundColor Green
Write-Host ""

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Environment name: $ENV_NAME" -ForegroundColor White
Write-Host "Python version: 3.11" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT NOTES:" -ForegroundColor Yellow
Write-Host "---------------" -ForegroundColor Yellow
Write-Host "1. CUDA Support:" -ForegroundColor White
if ($CUDA_AVAILABLE) {
    Write-Host "   [OK] GPU support enabled" -ForegroundColor Green
} else {
    Write-Host "   [!] CPU-only mode - For GPU support, install CUDA Toolkit" -ForegroundColor Yellow
    Write-Host "       Download: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
    Write-Host "       Then rerun: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "2. Running the Application:" -ForegroundColor White
Write-Host "   - Use launch.bat or launch.ps1 to start the transcriber" -ForegroundColor Cyan
Write-Host "   - Or manually: conda activate $ENV_NAME ; python realtime_transcriber_cross_platform.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Medical Transcription:" -ForegroundColor White
Write-Host "   - The app includes Crystalcareai/Whisper-Medicalv1 model support" -ForegroundColor Cyan
Write-Host "   - Medical model will be downloaded on first use" -ForegroundColor Cyan
Write-Host "   - Edit medical_vocabulary.txt to add custom medical terms" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Hotkey: Ctrl+F9 to toggle recording" -ForegroundColor White
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
