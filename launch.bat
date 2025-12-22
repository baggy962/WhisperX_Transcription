@echo off
REM ============================================================================
REM Whisper Real-Time Transcriber - Launch Script
REM For Windows 11 with Miniconda
REM ============================================================================

echo.
echo ========================================================================
echo  Whisper Real-Time Transcriber
echo ========================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Miniconda/Anaconda not found in PATH!
    echo.
    echo Please run setup.bat first to configure the environment.
    echo.
    pause
    exit /b 1
)

REM Environment name
set ENV_NAME=whisper-transcriber

echo [INFO] Checking environment '%ENV_NAME%'...
conda env list | findstr /C:"%ENV_NAME%" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Environment '%ENV_NAME%' not found!
    echo.
    echo Please run setup.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating environment...
call conda activate %ENV_NAME%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate environment!
    pause
    exit /b 1
)

echo [INFO] Starting Whisper Real-Time Transcriber...
echo.
echo ========================================================================
echo  Controls:
echo  - Hotkey: Ctrl+F9 to toggle recording
echo  - Model selection: Choose between General and Medical modes
echo  - Device: Auto-detects GPU/CPU
echo ========================================================================
echo.

REM Check if the Python script exists
if not exist "realtime_transcriber_cross_platform.py" (
    echo [ERROR] realtime_transcriber_cross_platform.py not found!
    echo Please ensure this script is run from the project directory.
    pause
    exit /b 1
)

REM Launch the application
python realtime_transcriber_cross_platform.py

REM Check if the application exited with an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Application closed successfully.
REM Don't pause on normal exit to allow clean shutdown
