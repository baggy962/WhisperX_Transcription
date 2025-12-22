@echo off
REM ============================================================================
REM Whisper Real-Time Transcriber - Environment Setup Script
REM For Windows 11 with Miniconda
REM Python 3.11
REM ============================================================================

echo.
echo ========================================================================
echo  Whisper Real-Time Transcriber - Environment Setup
echo ========================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Miniconda/Anaconda not found in PATH!
    echo.
    echo Please ensure Miniconda is installed and added to PATH.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

echo [Step 1/7] Checking conda installation...
conda --version
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Conda command failed!
    pause
    exit /b 1
)
echo [OK] Conda found
echo.
echo [DEBUG] Moving to Step 2...
echo.

REM Environment name
set ENV_NAME=whisper-transcriber

echo [Step 2/7] Checking if environment '%ENV_NAME%' exists...
conda env list | findstr /C:"%ENV_NAME%" >nul
if %ERRORLEVEL% equ 0 (
    echo [WARNING] Environment '%ENV_NAME%' already exists.
    choice /C YN /M "Do you want to remove and recreate it?"
    if errorlevel 2 (
        echo Setup cancelled.
        pause
        exit /b 0
    )
    echo Removing existing environment...
    call conda deactivate 2>nul
    call conda env remove -n %ENV_NAME% -y
)
echo.

echo [Step 3/7] Creating conda environment with Python 3.11...
call conda create -n %ENV_NAME% python=3.11 -y
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create conda environment!
    pause
    exit /b 1
)
echo [OK] Environment created
echo.

echo [Step 4/7] Checking for CUDA availability...
where nvcc >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] CUDA detected - Will install PyTorch with GPU support
    set CUDA_AVAILABLE=1
    nvcc --version | findstr "release"
) else (
    echo [INFO] CUDA not detected - Will install CPU-only PyTorch
    echo [INFO] For GPU support, install CUDA Toolkit first:
    echo        https://developer.nvidia.com/cuda-downloads
    set CUDA_AVAILABLE=0
)
echo.

echo [Step 5/7] Installing PyTorch in environment...
if "%CUDA_AVAILABLE%"=="1" (
    echo Installing PyTorch with CUDA support...
    call conda run -n %ENV_NAME% pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing PyTorch CPU-only...
    call conda run -n %ENV_NAME% pip install torch torchvision torchaudio
)
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install PyTorch!
    pause
    exit /b 1
)
echo [OK] PyTorch installed
echo.

echo [Step 6/7] Installing application dependencies...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    echo Please ensure this script is run from the project directory.
    pause
    exit /b 1
)

echo Installing dependencies to environment...
call conda run -n %ENV_NAME% pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo [OK] All dependencies installed
echo.

echo [Step 7/7] Verifying installation...
call conda run -n %ENV_NAME% python --version
echo [OK] Installation verified
echo.

echo ========================================================================
echo  Setup Complete!
echo ========================================================================
echo.
echo Environment name: %ENV_NAME%
echo Python version: 3.11
echo.
echo IMPORTANT NOTES:
echo ---------------
echo 1. CUDA Support: 
if "%CUDA_AVAILABLE%"=="1" (
    echo    [OK] GPU support enabled
) else (
    echo    [!] CPU-only mode - For GPU support, install CUDA Toolkit
    echo        Download: https://developer.nvidia.com/cuda-downloads
    echo        Then rerun: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
echo.
echo 2. Running the Application:
echo    - Use launch.bat to start the transcriber
echo    - Or manually: conda activate %ENV_NAME% ^&^& python realtime_transcriber_cross_platform.py
echo.
echo 3. Medical Transcription:
echo    - The app includes Crystalcareai/Whisper-Medicalv1 model support
echo    - Medical model will be downloaded on first use
echo    - Edit medical_vocabulary.txt to add custom medical terms
echo.
echo 4. Hotkey: Ctrl+F9 to toggle recording
echo.
echo ========================================================================
echo.
pause
