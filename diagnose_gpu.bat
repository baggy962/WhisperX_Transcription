@echo off
REM ============================================================================
REM GPU Diagnostics and PyTorch Reinstallation
REM Fixes CUDA kernel image compatibility issues
REM ============================================================================

echo.
echo ========================================================================
echo  GPU Diagnostics and PyTorch Fix
echo ========================================================================
echo.

set ENV_NAME=whisper-transcriber

echo [Step 1/5] Checking your GPU compute capability...
echo.
call conda run -n %ENV_NAME% python -c "import torch; print('='*60); print('CHECKING GPU INFORMATION'); print('='*60); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version in PyTorch: {torch.version.cuda}'); print(''); print('GPU Details:'); if torch.cuda.is_available(): print(f'  GPU Name: {torch.cuda.get_device_name(0)}'); props = torch.cuda.get_device_properties(0); print(f'  Compute Capability: {props.major}.{props.minor}'); print(f'  Total Memory: {props.total_memory / 1024**3:.1f} GB'); print(f'  Multi Processor Count: {props.multi_processor_count}'); print(''); print('='*60); print('DIAGNOSIS:'); print('='*60); cc = f'{props.major}.{props.minor}'; cc_float = float(cc); print(f'Your GPU Compute Capability: {cc}'); print(''); if cc_float >= 10.0: print('EXCELLENT: RTX 50-series (Blackwell) detected!'); print('Recommendation: Use CUDA 12.4 (cu124) build'); print('Note: CUDA 12.8 on system is compatible'); need_cu124 = True; need_cu118 = False; elif cc_float >= 8.9: print('EXCELLENT: RTX 40-series (Ada Lovelace) detected!'); print('Recommendation: Use CUDA 12.1 build'); need_cu124 = False; need_cu118 = False; elif cc_float >= 8.0: print('GOOD: RTX 30-series or newer'); print('Recommendation: Use CUDA 12.1 build'); need_cu124 = False; need_cu118 = False; elif cc_float >= 7.5: print('GOOD: RTX 20-series detected'); print('Recommendation: Use CUDA 11.8 build'); need_cu124 = False; need_cu118 = True; elif cc_float >= 6.1: print('OK: GTX 10-series detected'); print('Recommendation: Use CUDA 11.8 build'); need_cu124 = False; need_cu118 = True; elif cc_float >= 5.0: print('OK: Older GPU detected'); print('Recommendation: Use CUDA 11.8 build'); need_cu124 = False; need_cu118 = True; else: print('WARNING: Very old GPU'); print('Recommendation: Use CUDA 11.8 build'); need_cu124 = False; need_cu118 = True; print(''); print('='*60); exit(2 if need_cu124 else (1 if need_cu118 else 0))" 2>nul

set GPU_STATUS=%ERRORLEVEL%

echo.
echo [Step 2/5] Checking CUDA Toolkit version...
nvcc --version 2>nul | findstr "release"
echo.

echo [Step 3/5] Current PyTorch installation...
call conda run -n %ENV_NAME% python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Built with CUDA: {torch.version.cuda}')"
echo.

echo [Step 4/5] Determining best PyTorch build...
if %GPU_STATUS%==2 (
    echo Recommendation: CUDA 12.4 build for RTX 50-series
    set PYTORCH_URL=https://download.pytorch.org/whl/cu124
    set CUDA_BUILD=12.4
) else if %GPU_STATUS%==1 (
    echo Recommendation: CUDA 11.8 build
    set PYTORCH_URL=https://download.pytorch.org/whl/cu118
    set CUDA_BUILD=11.8
) else (
    echo Recommendation: CUDA 12.1 build
    set PYTORCH_URL=https://download.pytorch.org/whl/cu121
    set CUDA_BUILD=12.1
)
echo.

echo ========================================================================
echo RECOMMENDED ACTION
echo ========================================================================
echo.
echo Your GPU requires PyTorch built with CUDA %CUDA_BUILD%
echo.
choice /C YN /M "Do you want to reinstall PyTorch with CUDA %CUDA_BUILD%?"
if errorlevel 2 (
    echo Setup cancelled.
    pause
    exit /b 0
)

echo.
echo [Step 5/5] Reinstalling PyTorch with CUDA %CUDA_BUILD%...
echo This will download ~2.5GB...
echo.

echo Uninstalling current PyTorch...
call conda run -n %ENV_NAME% pip uninstall torch torchvision torchaudio -y

echo.
echo Installing PyTorch with CUDA %CUDA_BUILD%...
call conda run -n %ENV_NAME% pip install torch torchvision torchaudio --index-url %PYTORCH_URL%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Installation failed!
    echo Trying alternative installation method...
    call conda run -n %ENV_NAME% conda install pytorch torchvision torchaudio pytorch-cuda=%CUDA_BUILD% -c pytorch -c nvidia -y
)

echo.
echo ========================================================================
echo Verifying installation...
echo ========================================================================
call conda run -n %ENV_NAME% python -c "import torch; print('='*60); print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); if torch.cuda.is_available(): print(f'GPU Name: {torch.cuda.get_device_name(0)}'); props = torch.cuda.get_device_properties(0); print(f'Compute Capability: {props.major}.{props.minor}'); print('='*60); print('Testing CUDA operations...'); try: x = torch.randn(100, 100).cuda(); y = torch.randn(100, 100).cuda(); z = torch.matmul(x, y); print('CUDA operations: SUCCESS'); except Exception as e: print(f'CUDA operations: FAILED - {e}'); print('='*60)"

echo.
echo ========================================================================
echo Installation Complete
echo ========================================================================
echo.
echo Next steps:
echo 1. Close the app if running
echo 2. Restart the app
echo 3. Medical Mode should now work on GPU
echo.
pause
