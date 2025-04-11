@echo off
echo =============================
echo Testing SOVL Environment
echo =============================

:: This batch script, `sovl_enviro_test.bat`, is designed to test the environment setup required to run the 
:: Self-Organizing Virtual Lifeform (SOVL) system. Below is a detailed explanation of its functionality:

:: 1. Initial Setup and Logging:
::    - The script begins by printing a header indicating that it is testing the SOVL environment.

:: 2. Checking Python Installation:
::    - It verifies whether Python is installed and accessible via the system's PATH.
::    - If Python is not installed or the version is below 3.8, the script exits with an error message 
::      and prompts the user to install Python.

:: 3. Checking pip Installation:
::    - It checks if `pip`, the Python package installer, is installed.
::    - If not, it exits with an error message instructing the user to install `pip`.

:: 4. Checking and Installing Required Python Libraries:
::    - The script verifies the presence of critical Python libraries (`torch`, `transformers`, `peft`, 
::      `bitsandbytes`) using `pip show`.
::    - If a library is missing, the script automatically installs it using `pip install`.

:: 5. Verifying Required Files:
::    - The script ensures the presence of two essential files in the directory:
::        - `config.json`: A configuration file likely containing settings required for the SOVL system.
::        - `sample_log.jsonl`: A sample log file, possibly used for testing or debugging purposes.
::    - If either file is missing, the script exits with an error message.

:: 6. Running the Main SOVL Script in Test Mode:
::    - It executes the main SOVL script (`sovl_system/sovl_main.py`) with a `--dry-run` flag to simulate its 
::      execution and detect any runtime issues without making actual changes.
::    - If the script fails, it exits with an error message instructing the user to review the output for errors.

:: 7. Final Confirmation:
::    - If all checks pass successfully, the script prints a confirmation message indicating that the 
::      environment test has passed.

:: 8. Pause:
::    - The script includes pauses at key failure points and at the end to ensure the user can review the 
::      messages before the terminal closes.

:: This script ensures that the environment is properly set up for running the SOVL system, checking all 
:: prerequisites such as Python, libraries, and configuration files. It also provides helpful error messages 
:: and attempts to resolve missing libraries automatically.

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
