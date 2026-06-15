@echo off
chcp 65001 >nul
echo ==========================================
echo   钢铁表面缺陷检测系统 - 启动脚本
echo ==========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保 Python 已安装并添加到 PATH
    pause
    exit /b 1
)

echo [1/4] Python 环境检查通过
echo.

REM 激活虚拟环境
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo [2/4] 虚拟环境已激活
) else (
    echo [信息] 未找到 .venv 虚拟环境，使用系统 Python
)
echo.

REM 检查依赖
echo [3/4] 检查依赖...
python -c "import torch, gradio, ultralytics, cv2, yaml" >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装依赖 (使用清华镜像源加速)...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败
        pause
        exit /b 1
    )
)
echo [3/4] 依赖检查通过
echo.

REM 启动应用
echo [4/4] 启动 Gradio Web 工作台...
echo.
echo 访问地址: http://localhost:7860
echo 按 Ctrl+C 停止服务
echo.

python main.py --mode app

pause
