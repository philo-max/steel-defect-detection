#!/bin/bash
# ============================================================
# 钢铁表面缺陷检测系统 - 服务器环境部署脚本 (Ubuntu CPU环境)
# ============================================================

set -e

echo "=== 开始配置服务器部署环境 ==="

# 1. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "[1/4] 创建 Python 虚拟环境..."
    python3 -m venv .venv
    echo "虚拟环境已创建"
else
    echo "[1/4] 虚拟环境已存在，跳过创建"
fi

# 2. 激活虚拟环境
echo "[2/4] 激活虚拟环境..."
source .venv/bin/activate

# 3. 安装依赖包 (使用腾讯云 Pypi 镜像源并配置 CPU-only PyTorch 以节省流量和空间)
echo "[3/4] 安装依赖 (使用腾讯云镜像源 & CPU-only PyTorch)..."
pip install --upgrade pip -i https://mirrors.cloud.tencent.com/pypi/simple

# 安装依赖。由于是 CPU 运行环境，特别加入 PyTorch CPU 专属索引
pip install -r requirements.txt \
    -i https://mirrors.cloud.tencent.com/pypi/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 4. 创建必要的运行文件夹
echo "[4/4] 创建项目所需的数据和日志目录..."
mkdir -p data/images data/uploads data/exports logs models/weights

# 5. 初始化配置文件
if [ ! -f ".env" ]; then
    echo "创建默认环境变量配置文件 .env ..."
    cp .env.example .env
fi

echo "================================================="
echo " 服务器环境部署成功！"
echo "================================================="
echo " 启动指令:"
echo "   source .venv/bin/activate"
echo "   python3 main.py --mode app"
echo "================================================="
