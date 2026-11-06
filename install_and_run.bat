@echo off
setlocal
:: Download (or update) S2L from GitHub into %USERPROFILE%\S2L, then run its
:: setup + launcher (install_and_run_local.bat), which installs conda if needed.
set "REPO_URL=https://github.com/we-02/S2L.git"
set "INSTALL_DIR=%USERPROFILE%\S2L"

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed. Install it from https://git-scm.com/download/win
    echo         or use S2L_Setup.exe, which needs no Git.
    pause
    exit /b 1
)
if exist "%INSTALL_DIR%\.git" (
    echo [*] Updating S2L...
    git -C "%INSTALL_DIR%" pull --ff-only || echo [WARN] Could not pull updates, using the existing code.
) else (
    echo [*] Downloading S2L...
    git clone "%REPO_URL%" "%INSTALL_DIR%"
    if errorlevel 1 (
        echo [ERROR] Could not clone %REPO_URL%
        pause
        exit /b 1
    )
)
call "%INSTALL_DIR%\install_and_run_local.bat"
