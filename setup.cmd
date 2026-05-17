@echo off
REM ============================================================================
REM Greek TTS - Setup script (English-only to avoid cmd.exe encoding issues)
REM ----------------------------------------------------------------------------
REM What this script does, in order:
REM   1. Verifies winget is available.
REM   2. Installs ffmpeg via winget (skipped if already installed).
REM   3. Ensures Python is available (installs via winget if missing).
REM   4. Installs the huggingface_hub Python package.
REM   5. Downloads the three model components to the user's HF cache.
REM
REM Designed to be re-runnable: each step checks whether it's already done
REM and skips if so. Safe to run multiple times - downloads resume from
REM where they stopped.
REM
REM Usage: double-click setup.cmd, or run from cmd:  setup.cmd
REM ============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   Greek TTS - Setup
echo   Installing dependencies and downloading models
echo ============================================================
echo.

REM Track whether anything failed so we can summarize at the end
set "SETUP_FAILED=0"

REM ----------------------------------------------------------------------------
REM Step 1: Check for winget (Windows Package Manager)
REM ----------------------------------------------------------------------------
echo [1/5] Checking for winget...
where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ERROR: winget is not available on this machine.
    echo   Solutions:
    echo     - On Windows 11 and recent Windows 10, install winget from the
    echo       Microsoft Store: search for "App Installer".
    echo     - Or install ffmpeg and Python manually, then re-run this script.
    echo.
    pause
    exit /b 1
)
echo   winget found.
echo.

REM ----------------------------------------------------------------------------
REM Step 2: Install ffmpeg (if not already on PATH)
REM ----------------------------------------------------------------------------
echo [2/5] Checking for ffmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo   ffmpeg not found. Installing via winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements --silent
    if errorlevel 1 (
        echo   WARNING: winget install of ffmpeg failed.
        echo   Try installing manually from https://ffmpeg.org/download.html
        set "SETUP_FAILED=1"
    ) else (
        echo   ffmpeg installed.
        echo   NOTE: a fresh Command Prompt may be needed for ffmpeg to be
        echo         on PATH. If GreekTTS.exe later complains about ffmpeg,
        echo         close all command prompts and try again.
    )
) else (
    echo   ffmpeg already installed.
)
echo.

REM ----------------------------------------------------------------------------
REM Step 3: Install Python 3.11 (if not already present)
REM ----------------------------------------------------------------------------
echo [3/5] Checking for Python...
where python >nul 2>&1
if errorlevel 1 (
    echo   Python not found. Installing Python 3.11 via winget...
    winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements --silent
    if errorlevel 1 (
        echo   ERROR: failed to install Python.
        set "SETUP_FAILED=1"
        goto :summary
    )
    REM winget puts Python on PATH for new shells, but not for this one.
    REM Try to refresh PATH from the registry so subsequent commands work.
    for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
    set "PATH=%PATH%;%USER_PATH%"
    where python >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   Python was installed but is not on PATH in this session.
        echo   Please close this window and re-run setup.cmd.
        echo.
        pause
        exit /b 1
    )
) else (
    echo   Python already installed.
)
python --version
echo.

REM ----------------------------------------------------------------------------
REM Step 4: Install huggingface_hub
REM ----------------------------------------------------------------------------
echo [4/5] Installing huggingface_hub...
REM Note: don't use the [cli] extra - in huggingface_hub 1.0+ the CLI is part
REM of the base package and the [cli] extra was removed. The --no-warn-script-location
REM flag suppresses the noisy PATH warnings on Microsoft Store Python installs;
REM we don't need the CLI on PATH because we call snapshot_download directly.
echo   (this may take a minute on first install)
python -m pip install --upgrade --quiet --no-warn-script-location huggingface_hub
if errorlevel 1 (
    echo   ERROR: failed to install huggingface_hub.
    set "SETUP_FAILED=1"
    goto :summary
)
echo   huggingface_hub ready.
echo.

REM ----------------------------------------------------------------------------
REM Step 5: Download model components (~2.5 GB total, resumable)
REM ----------------------------------------------------------------------------
echo [5/5] Downloading model components...
echo   This will take 5-15 minutes depending on your network speed.
echo   Models are cached in: %USERPROFILE%\.cache\huggingface\hub
echo   Downloads resume if interrupted - safe to re-run this script.
echo.

echo   Downloading Orpheus base model (~2 GB)...
python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/orpheus-3b-0.1-ft')"
if errorlevel 1 (
    echo   ERROR: download of Orpheus base failed.
    set "SETUP_FAILED=1"
)
echo.

echo   Downloading Greek LoRA adapter (~389 MB)...
python -c "from huggingface_hub import snapshot_download; snapshot_download('moiralabs/GreekTTS-1.5')"
if errorlevel 1 (
    echo   ERROR: download of Greek LoRA failed.
    set "SETUP_FAILED=1"
)
echo.

echo   Downloading SNAC vocoder (~80 MB)...
python -c "from huggingface_hub import snapshot_download; snapshot_download('hubertsiuzdak/snac_24khz')"
if errorlevel 1 (
    echo   ERROR: download of SNAC vocoder failed.
    set "SETUP_FAILED=1"
)
echo.

REM ----------------------------------------------------------------------------
REM Summary
REM ----------------------------------------------------------------------------
:summary
echo ============================================================
if "%SETUP_FAILED%"=="0" (
    echo   Setup complete!
    echo   You can now launch GreekTTS.exe.
    echo.
    echo   The first launch will skip the model-download dialog
    echo   since everything is already cached.
) else (
    echo   Setup finished with WARNINGS or ERRORS.
    echo   Re-run setup.cmd to retry any failed steps - downloads
    echo   resume from where they stopped.
    echo.
    echo   If problems persist, see INSTALL-WINDOWS.md
    echo   for manual installation steps.
)
echo ============================================================
echo.
pause
endlocal
