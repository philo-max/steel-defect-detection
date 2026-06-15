@echo off
chcp 65001 >nul
echo ==========================================
# 钢铁表面缺陷检测系统 - 一键打包脚本
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pack.ps1"
echo.
pause
