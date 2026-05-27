# 运维文档

> 钢铁表面缺陷检测系统 V2.0 | 2026-05-27

## 部署

### 环境要求

- Windows 11 / Ubuntu 20.04+
- Python 3.12
- NVIDIA GPU (RTX 5060 8GB 推荐)
- 16GB+ RAM, 50GB 存储

### 一键部署

```bash
git clone https://github.com/philo-max/steel-defect-detection.git
cd steel-defect-detection
setup.bat          # Windows
# bash setup.sh    # Linux
```

### 启动

```bash
python app.py                           # Web 工作台 → :7860
python app.py --config custom.yaml      # 自定义配置
```

### Docker

```bash
docker-compose up -d
```

## 配置

核心文件：`config.yaml`、`.env`

| 文件 | 用途 |
|------|------|
| `config.yaml` | 系统参数（相机/YOLO/VLM/PLC） |
| `.env` | 密钥（API Key）不提交 Git |

### 最小配置

```yaml
yolo:
  model_path: "models/weights/steel_defect.pt"
  device: "auto"

vlm:
  enabled: true      # 留空自动探测 Gemini/Qwen
```

## 模型管理

### 模型文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `steel_defect.pt` | 22.5MB | 默认推理 |
| `steel_defect.onnx` | 42.7MB | ONNX 部署 |
| `steel_defect.pt.bak` | 22.5MB | 上一版本备份 |

### 切换模型

```bash
# 命令行
python cli.py switch-model models/weights/new_model.pt

# 或手动修改 config.yaml
yolo:
  model_path: "models/weights/new_model.pt"
```

### 模型更新

```bash
# 全量训练
python scripts/train_yolo.py --model yolov8s.pt --epochs 200

# 快速验证
python scripts/train_yolo.py --quick

# 训练后自动覆盖 steel_defect.pt
```

## 监控

### 健康检查

```bash
curl http://127.0.0.1:7861/health
```

返回:

```json
{
  "status": "healthy",
  "gpu": {"available": true, "memory_usage": 78.5},
  "cpu": {"usage": 35.2},
  "camera": {"connected": true, "fps": 30}
}
```

### 性能基准

```bash
python scripts/benchmark.py --model models/weights/steel_defect.pt
```

基准数据 (RTX 5060):

| 指标 | 值 |
|------|-----|
| P50 延迟 | 46ms |
| P99 延迟 | 76ms |
| 单张 FPS | 21.5 |
| 批量 FPS | 23.1 |

## 日志

| 位置 | 内容 |
|------|------|
| 终端 stdout | 实时运行日志 |
| `logs/` | 文件日志（如配置） |
| `runs/train/` | 训练历史 |

## 备份

### 需要备份的内容

- `models/weights/steel_defect.pt` — 模型权重
- `data/inspection.db` — 检测记录
- `config.yaml` — 系统配置
- `.env` — API Key

### 备份命令

```bash
# 手动备份
copy models\weights\steel_defect.pt D:\backup\steel_defect_20260527.pt

# 数据库备份
copy data\inspection.db D:\backup\inspection_20260527.db
```

## 故障处理

| 问题 | 检查 | 解决 |
|------|------|------|
| 启动失败 | `python -c "import torch; print(torch.cuda.is_available())"` | 重装 CUDA PyTorch |
| 模型加载失败 | `models/weights/steel_defect.pt` 是否存在 | 重新训练或下载 |
| VLM 不可用 | 网络连接、API Key | 降级为纯 YOLO |
| GPU OOM | `nvidia-smi` | 降 batch size / ONNX |
| 端口冲突 | `netstat -an \| findstr 7860` | 修改 `app.py` 端口 |

## 升级

```bash
git pull origin master
pip install -r requirements.txt --upgrade
# 如有新模型，替换 models/weights/steel_defect.pt
python app.py
```

## 当前模型指标

| 类别 | mAP50 |
|------|:-----:|
| crazing | 0.993 |
| inclusion | 0.749 |
| patches | 0.995 |
| pitted_surface | 0.995 |
| rolled-in-scale | 0.609 |
| scratches | 0.978 |
| **总体** | **0.906** |
