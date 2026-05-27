@echo off
REM ============================================================
REM 钢铁表面缺陷检测系统 - 一键部署脚本 (Windows)
REM ============================================================

echo.
echo ============================================
echo   钢铁表面缺陷检测系统 - 环境部署
echo ============================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 未安装，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/4] 创建虚拟环境...
if not exist .venv (
    python -m venv .venv
    echo 虚拟环境已创建
) else (
    echo 虚拟环境已存在，跳过
)

echo [2/4] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [3/4] 安装依赖...
pip install --upgrade pip -q
pip install -r requirements.txt

echo [4/4] 创建必要目录...
if not exist models\weights mkdir models\weights
if not exist data\images mkdir data\images
if not exist data\exports mkdir data\exports
if not exist logs mkdir logs

echo.
echo ============================================
echo   部署完成！
echo.
echo   下一步:
echo   1. 将 YOLO 模型权重放入 models\weights\
echo   2. 复制 .env.example 为 .env 并填入 API Key
echo   3. 运行: .venv\Scripts\python main.py
echo ============================================
echo.
pause
