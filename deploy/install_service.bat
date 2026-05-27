@echo off
REM 钢铁表面缺陷检测系统 - Windows 开机自启配置 (NSSM)
REM 需要先下载 NSSM: https://nssm.cc/download

set SERVICE_NAME=SteelDefectDetection
set APP_DIR=%~dp0
set PYTHON=%APP_DIR%.venv\Scripts\python.exe
set MAIN=%APP_DIR%main.py

echo === 钢铁表面缺陷检测系统 - 服务安装 ===

REM 检查 NSSM
where nssm >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] NSSM 未安装，请先下载: https://nssm.cc/download
    echo        将 nssm.exe 放入 PATH 目录后重试
    pause
    exit /b 1
)

REM 安装服务
nssm install %SERVICE_NAME% "%PYTHON%" "%MAIN%" --mode app
nssm set %SERVICE_NAME% AppDirectory "%APP_DIR%"
nssm set %SERVICE_NAME% DisplayName "Steel Defect Detection System"
nssm set %SERVICE_NAME% Description "钢铁表面缺陷检测系统自动启动服务"
nssm set %SERVICE_NAME% Start SERVICE_AUTO_START
nssm set %SERVICE_NAME% AppStdout "%APP_DIR%logs\service_stdout.log"
nssm set %SERVICE_NAME% AppStderr "%APP_DIR%logs\service_stderr.log"
nssm set %SERVICE_NAME% AppRotateFiles 1
nssm set %SERVICE_NAME% AppRotateBytes 10485760

echo [OK] 服务已安装: %SERVICE_NAME%
echo [INFO] 启动服务: nssm start %SERVICE_NAME%
echo [INFO] 停止服务: nssm stop %SERVICE_NAME%
echo [INFO] 卸载服务: nssm remove %SERVICE_NAME% confirm
pause
