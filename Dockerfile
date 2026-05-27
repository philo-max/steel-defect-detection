FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# 系统依赖 (OpenCV 需要 libgl/libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# 创建数据目录
RUN mkdir -p data/images data/exports/csv data/exports/reports data/exports/badcase logs

# 暴露 Gradio 端口
EXPOSE 7860

# 环境变量 (覆盖配置)
ENV PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["python", "main.py", "--mode", "app"]
