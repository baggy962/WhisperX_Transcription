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
call conda run -n %ENV_NAME% python -c "import torch; print('='*60); print('CHECKING GPU INFORMATION'); print('='*60); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version in PyTorch: {torch.version.cuda}'); print(''); print('GPU Details:'); if torch.cuda.is_available(): print(f'  GPU Name: {torch.cuda.get_device_name(0)}'); props = torch.cuda.get_device_properties(0); print(f'  Compute Capability: {props.major}.{props.minor}'); print(f'  Total Memory: {props.total_memory / 1024**3:.1f} GB'); print(f'  Multi Processor Count: {props.multi_processor_count}'); print(''); print('='*60); print('DIAGNOSIS:'); print('='*60); cc = f'{props.major}.{props.minor}'; print(f'Your GPU Compute Capability: {cc}'); print(''); if float(cc) < 5.0: print('ERROR: Your GPU is too old for CUDA 12.x'); print('Recommendation: Use CUDA 11.8 build'); need_cu118 = True; elif float(cc) >= 8.0: print('GOOD: Your GPU supports modern CUDA'); print('Recommendation: Use CUDA 12.1 build'); need_cu118 = False; else: print('OK: Your GPU supports CUDA 11.8/12.1'); print('Recommendation: Use CUDA 11.8 build (safer)'); need_cu118 = True; print(''); print('='*60); exit(0 if not need_cu118 else 1)" 2>nul

set NEED_CU118=%ERRORLEVEL%

echo.
echo [Step 2/5] Checking CUDA Toolkit version...
nvcc --version 2>nul | findstr "release"
echo.

echo [Step 3/5] Current PyTorch installation...
call conda run -n %ENV_NAME% python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Built with CUDA: {torch.version.cuda}')"
echo.

echo [Step 4/5] Determining best PyTorch build...
if %NEED_CU118%==1 (
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
