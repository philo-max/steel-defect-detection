# Steel Surface Defect Detection System - PowerShell Startup Script
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Steel Surface Defect Detection System" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[1/4] Python Environment: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[Error] Python not found. Please ensure Python is installed and added to PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "[2/4] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[Info] No .venv found, using system Python" -ForegroundColor Yellow
}

Write-Host ""

# Check dependencies
Write-Host "[3/4] Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import torch, gradio, ultralytics, cv2, yaml"
    Write-Host "[3/4] Dependencies check passed" -ForegroundColor Green
} catch {
    Write-Host "[Info] Installing dependencies..." -ForegroundColor Yellow
    try {
        pip install -r requirements.txt
        Write-Host "[3/4] Dependencies installation completed" -ForegroundColor Green
    } catch {
        Write-Host "[Error] Dependencies installation failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# Start application
Write-Host "[4/4] Starting Gradio Web Workbench..." -ForegroundColor Green
Write-Host ""
Write-Host "Access URL: http://localhost:7860" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop service" -ForegroundColor Gray
Write-Host ""

python main.py --mode app

Read-Host "Press Enter to exit"
