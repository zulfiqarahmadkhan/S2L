@echo off
setlocal enabledelayedexpansion
:: S2L setup + launcher for Windows. Works on a new PC: installs Miniforge (conda)
:: if no conda is found, creates the S2L environment, installs PyTorch for the
:: GPU driver, all packages and the pretrained models, then starts S2L.
:: Later runs only check that everything is in place (a few seconds).
::
:: Optional settings (environment variables, mainly for testing):
::   S2L_ENV_NAME    conda environment name           (default S2L)
::   S2L_CONDA_HOME  use / install conda in this folder only
::   S2L_NO_LAUNCH   set to 1 to set up without starting the app

cd /d "%~dp0"
set "ENV_NAME=S2L"
if not "%S2L_ENV_NAME%"=="" set "ENV_NAME=%S2L_ENV_NAME%"
set "PYTHON_VER=3.11"
set "NEW_CONDA=%USERPROFILE%\miniforge3"
if not "%S2L_CONDA_HOME%"=="" set "NEW_CONDA=%S2L_CONDA_HOME%"
set "MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"

echo.
echo  ========================================
echo              S2L setup and launcher
echo  ========================================
echo.

:: --- Find conda (Anaconda, Miniconda or Miniforge) -------------------------
set "CONDA_CMD="
if not "%S2L_CONDA_HOME%"=="" (
    if exist "%S2L_CONDA_HOME%\condabin\conda.bat" set "CONDA_CMD=%S2L_CONDA_HOME%\condabin\conda.bat"
) else (
    for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%C"
    for %%P in (
        "%USERPROFILE%\miniforge3\condabin\conda.bat"
        "%LOCALAPPDATA%\miniforge3\condabin\conda.bat"
        "%USERPROFILE%\anaconda3\condabin\conda.bat"
        "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
        "%USERPROFILE%\miniconda3\condabin\conda.bat"
        "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
        "C:\ProgramData\miniforge3\condabin\conda.bat"
        "C:\ProgramData\anaconda3\condabin\conda.bat"
        "C:\ProgramData\miniconda3\condabin\conda.bat"
        "C:\anaconda3\condabin\conda.bat"
    ) do (
        if not defined CONDA_CMD if exist %%P set "CONDA_CMD=%%~P"
    )
)

:: --- No conda: install Miniforge for this user (no admin rights needed) ----
if not defined CONDA_CMD (
    echo [*] No conda found. Installing Miniforge to "!NEW_CONDA!" ^(about 100 MB, one time^)...
    set "MF_EXE=%TEMP%\Miniforge3-Windows-x86_64.exe"
    curl.exe -L --fail --silent --show-error -o "!MF_EXE!" "%MINIFORGE_URL%"
    if errorlevel 1 (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri '%MINIFORGE_URL%' -OutFile '!MF_EXE!'"
    )
    if not exist "!MF_EXE!" (
        echo [ERROR] Could not download Miniforge. Check the internet connection and try again.
        pause
        exit /b 1
    )
    start /wait "" "!MF_EXE!" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=!NEW_CONDA!
    del "!MF_EXE!" >nul 2>&1
    if not exist "!NEW_CONDA!\condabin\conda.bat" (
        echo [ERROR] Miniforge installation failed.
        pause
        exit /b 1
    )
    set "CONDA_CMD=!NEW_CONDA!\condabin\conda.bat"
    echo [*] Miniforge installed.
)
echo [*] Using conda: %CONDA_CMD%

:: --- S2L environment -------------------------------------------------------
call "%CONDA_CMD%" env list | findstr /b /c:"%ENV_NAME% " >nul 2>&1
if errorlevel 1 (
    echo [*] Creating the '%ENV_NAME%' environment with Python %PYTHON_VER% ^(one time^)...
    rem conda-forge only: works with every conda and needs no Anaconda terms-of-service acceptance
    call "%CONDA_CMD%" create -y -n %ENV_NAME% -c conda-forge --override-channels python=%PYTHON_VER%
    if errorlevel 1 (
        echo [ERROR] Could not create the conda environment '%ENV_NAME%'.
        pause
        exit /b 1
    )
    if exist ".s2l_req_done" del ".s2l_req_done"
)
set "RUN=call "%CONDA_CMD%" run --no-capture-output -n %ENV_NAME%"

:: --- PyTorch matching the NVIDIA driver (before requirements.txt) ----------
echo [*] Checking PyTorch (GPU build if an NVIDIA driver supports it)...
%RUN% python install_torch.py
if errorlevel 1 echo [WARN] PyTorch setup reported a problem; S2L may run on the CPU.

:: --- Packages (skipped when unchanged since the last successful install) ---
copy /b /y requirements.txt+requirements-optional.txt ".s2l_req_now" >nul
set "NEED_PIP=1"
if exist ".s2l_req_done" (
    fc /b ".s2l_req_now" ".s2l_req_done" >nul 2>&1 && set "NEED_PIP=0"
)
if "!NEED_PIP!"=="1" (
    echo [*] Installing required packages ^(first run: several minutes^)...
    %RUN% python -m pip install --quiet --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo [*] Retrying once ^(downloads can fail on a slow or busy connection^)...
        %RUN% python -m pip install --quiet --disable-pip-version-check -r requirements.txt
    )
    if errorlevel 1 (
        echo [ERROR] Required packages could not be installed. Check the messages above.
        pause
        exit /b 1
    )
    echo [*] Installing optional packages ^(StarDist, TensorFlow, Cell-ACDC^)...
    %RUN% python -m pip install --quiet --disable-pip-version-check -r requirements-optional.txt
    if errorlevel 1 echo [WARN] Some optional packages failed; S2L works without them ^(StarDist may be unavailable^).
)
%RUN% python -c "import PyQt6.QtWidgets, cellpose.models, torch, sklearn, skimage, pandas"
if errorlevel 1 (
    echo [ERROR] S2L's packages do not import. Delete the file .s2l_req_done in this folder and run again.
    if exist ".s2l_req_done" del ".s2l_req_done"
    pause
    exit /b 1
)
move /y ".s2l_req_now" ".s2l_req_done" >nul

:: --- Pretrained models (Cellpose-SAM + StarDist; skipped when present) -----
echo [*] Checking pretrained models ^(the first run downloads about 1.2 GB^)...
%RUN% python -m s2l.download_models
if errorlevel 1 echo [WARN] Some models could not be downloaded; they are downloaded on first use.

if "%S2L_NO_LAUNCH%"=="1" (
    echo [*] Setup complete ^(not launching: S2L_NO_LAUNCH=1^).
    exit /b 0
)

:: --- Launch ----------------------------------------------------------------
echo [*] Starting S2L... ^(keep this window open while S2L runs^)
echo.
%RUN% python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] S2L exited with an error. Details are in s2l_crash.log in this folder.
    pause
)
