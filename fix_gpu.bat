@echo off
REM ============================================================================
REM Fix PyTorch GPU Support - Install CUDA-enabled PyTorch
REM For CUDA 12.8 (compatible with cu121 build)
REM ============================================================================

echo.
echo ========================================================================
echo  PyTorch GPU Support Installer
echo ========================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Miniconda/Anaconda not found in PATH!
    pause
    exit /b 1
)

set ENV_NAME=whisper-transcriber

echo [INFO] Checking environment...
conda env list | findstr /C:"%ENV_NAME%" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Environment '%ENV_NAME%' not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo [Step 1/4] Checking current PyTorch installation...
call conda run -n %ENV_NAME% python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo.

echo [Step 2/4] Uninstalling CPU-only PyTorch...
call conda run -n %ENV_NAME% pip uninstall torch torchvision torchaudio -y
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to uninstall PyTorch!
    pause
    exit /b 1
)
echo [OK] Old PyTorch removed
echo.

echo [Step 3/4] Installing PyTorch with CUDA 12.1 support...
echo [INFO] CUDA 12.8 is compatible with PyTorch cu121 build
echo [INFO] This will download ~2.5GB...
call conda run -n %ENV_NAME% pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install PyTorch with CUDA!
    echo.
    echo Trying CUDA 11.8 build instead...
    call conda run -n %ENV_NAME% pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Both CUDA builds failed!
        pause
        exit /b 1
    )
)
echo [OK] PyTorch with CUDA installed
echo.

echo [Step 4/4] Verifying GPU support...
call conda run -n %ENV_NAME% python -c "import torch; print('='*60); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); cuda_available = torch.cuda.is_available(); print('='*60); print('GPU SUPPORT: ENABLED' if cuda_available else 'GPU SUPPORT: FAILED'); print('='*60); exit(0 if cuda_available else 1)"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] GPU support verification failed!
    echo This might still work, but GPU may not be detected.
) else (
    echo.
    echo [SUCCESS] GPU support is now enabled!
)
echo.

echo ========================================================================
echo  Installation Complete
echo ========================================================================
echo.
echo Next steps:
echo 1. Close the app if it's running
echo 2. Restart the app: python realtime_transcriber_cross_platform.py
echo 3. You should now see GPU option in the device selection
echo.
echo Expected output on startup:
echo   PyTorch version: 2.x.x
echo   CUDA available: True
echo   CUDA version: 12.1
echo   GPU: [Your NVIDIA GPU Name]
echo.
pause
