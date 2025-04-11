@echo off
echo =============================
echo Testing SOVL Environment
echo =============================

:: Check Python version
echo Checking Python version...
python --version
IF ERRORLEVEL 1 (
    echo Python is not installed or not added to PATH. Please install Python 3.8+.
    pause
    exit /b 1
)

:: Check if pip is installed
echo Checking pip...
pip --version
IF ERRORLEVEL 1 (
    echo pip is not installed. Please install pip.
    pause
    exit /b 1
)

:: Check for required libraries
echo Checking required Python libraries...

pip show torch >nul 2>&1
IF ERRORLEVEL 1 (
    echo Missing library: torch. Installing...
    pip install torch
)

pip show transformers >nul 2>&1
IF ERRORLEVEL 1 (
    echo Missing library: transformers. Installing...
    pip install transformers
)

pip show peft >nul 2>&1
IF ERRORLEVEL 1 (
    echo Missing library: peft. Installing...
    pip install peft
)

pip show bitsandbytes >nul 2>&1
IF ERRORLEVEL 1 (
    echo Missing library: bitsandbytes. Installing...
    pip install bitsandbytes
)

:: Check for config.json
echo Checking for config.json...
IF NOT EXIST config.json (
    echo config.json not found. Please ensure it is in the directory.
    pause
    exit /b 1
)

:: Check for sample_log.jsonl
echo Checking for sample_log.jsonl...
IF NOT EXIST sample_log.jsonl (
    echo sample_log.jsonl not found. Please ensure it is in the directory.
    pause
    exit /b 1
)

:: Run the main script as a test
echo Running SOVL system script to test...
python sovl_system/sovl_main.py --dry-run
IF ERRORLEVEL 1 (
    echo SOVL system test failed. Please check the output for errors.
    pause
    exit /b 1
)

echo =============================
echo Environment Test Passed
echo =============================
pause
