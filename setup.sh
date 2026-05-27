#!/bin/bash
# 钢铁表面缺陷检测系统 - Linux 一键部署脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  钢铁表面缺陷检测系统 - 环境部署"
echo "======================================"

# 1. 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[FAIL] Python3 未安装，请先安装 Python 3.8+"
    exit 1
fi
echo "[OK] Python: $(python3 --version)"

# 2. 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "[WARN] nvidia-smi 未找到，将使用 CPU 推理"
fi

# 3. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "[INFO] 创建虚拟环境..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "[OK] 虚拟环境已激活"

# 4. 安装依赖
echo "[INFO] 安装依赖..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "[OK] 依赖安装完成"

# 5. 创建目录
mkdir -p data/images data/exports/csv data/exports/reports data/exports/badcase logs models/weights
echo "[OK] 目录结构已创建"

# 6. 环境变量
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[INFO] 已创建 .env 文件，请填写 API Key"
fi

# 7. 环境验证
echo ""
echo "======================================"
echo "  环境验证"
echo "======================================"
python cli.py verify || true

echo ""
echo "======================================"
echo "  部署完成!"
echo "======================================"
echo ""
echo "启动方式:"
echo "  启动 Web 工作台:  python main.py --mode app"
echo "  命令行检测:       python main.py --mode detect --image <path>"
echo "  导出数据:         python main.py --mode export"
echo "  查看状态:         python cli.py status"
echo "  验证环境:         python cli.py verify"
