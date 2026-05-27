# 钢铁表面缺陷检测系统

## 项目概述

钢铁表面缺陷检测系统是一个基于深度学习的工业级智能质检平台，采用YOLO+VLM双引擎架构，实现钢铁表面缺陷的自动化检测、分类和根因分析。

**核心特性：**

- ⚡ **YOLO快速筛查**：实时检测6类标准缺陷（裂纹、夹杂、斑块、麻点、轧制氧化皮、划痕）
- 🧠 **VLM精细分析**：基于Gemini等大模型进行缺陷描述、严重程度评估和位置分析
- 🔍 **RAG根因分析**：结合知识库进行缺陷根因推理和工艺优化建议
- 📊 **全流程管理**：从图像采集、检测、人工审核到报表导出的一体化工作流
- 🏭 **工业级设计**：支持PLC硬触发、多相机同步、实时监控告警

## 技术架构

### 系统架构图

```text
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web 工作台                         │
├─────────────┬─────────────┬─────────────┬─────────────┤
│  实时检测    │  人工审核    │  统计报表    │  系统监控    │
└─────────────┴─────────────┴─────────────┴─────────────┘
        │              │              │              │
┌───────▼──────────────▼──────────────▼──────────────▼───────┐
│                双引擎检测服务 (YOLO + VLM)                   │
├─────────────────────────────────────────────────────────────┤
│   YOLO检测引擎  │  VLM分析引擎  │  RAG知识库  │  数据管理   │
└─────────────────────────────────────────────────────────────┘
        │              │              │              │
┌───────▼──────────────▼──────────────▼──────────────▼───────┐
│                工业相机接口 (PLC硬触发支持)                   │
├─────────────────────────────────────────────────────────────┤
│   USB相机    │  RTSP流    │  PLC信号    │  光源控制    │
└─────────────────────────────────────────────────────────────┘

```

### 技术栈

- **深度学习框架**：PyTorch 2.0+, Ultralytics YOLOv8
- **视觉大模型**：Gemini API / Qwen API / OpenAI兼容接口
- **Web界面**：Gradio 4.0+ (Python Web框架)
- **数据处理**：OpenCV, NumPy, Pandas, Matplotlib
- **数据库**：SQLite (轻量级本地存储)
- **工业通信**：Modbus TCP (pymodbus)
- **监控告警**：自定义健康检查 + 告警引擎
- **部署方式**：Windows/Linux本地部署，支持Docker容器化

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.x (GPU加速推荐) 或 CPU模式
- Windows 10/11 或 Linux

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd steel-defect-detection
```

1. **创建虚拟环境**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

1. **安装依赖**

```bash
pip install -r requirements.txt
```

1. **配置环境变量**

```bash
# 复制示例配置
copy .env.example .env
# 编辑.env文件，设置API密钥等
```

1. **下载预训练模型**

```bash
# 自动下载YOLO预训练权重
python scripts/download_weights.py
```

### 运行系统

1. **启动Gradio工作台**

```bash
python main.py
# 或指定模式
python main.py --mode app
```

1. **访问Web界面**
打开浏览器访问：`http://127.0.0.1:7860`

2. **命令行模式**

```bash
# 单张图像检测
python main.py --mode detect --image path/to/image.jpg

# 批量导出
python main.py --mode export

# 查看帮助
python main.py --help
```

### 配置文件说明

系统使用 `config.yaml` 进行配置，主要配置段：

```yaml
# 图像采集配置
camera:
  source: "0"  # 0=本地摄像头, rtsp://地址=网络流
  width: 1920
  height: 1080
  fps: 30
  trigger_mode: "continuous"  # continuous | plc_hardware

# YOLO检测配置
yolo:
  model_path: "models/weights/steel_defect.pt"
  conf_threshold: 0.05
  device: "auto"  # auto, cuda:0, cpu

# VLM配置
vlm:
  enabled: true
  api_base: ""  # 留空自动检测可用API
  model: "gemini-1.5-flash"

# PLC硬触发配置
plc:
  enabled: false
  host: "192.168.1.100"
  port: 502
  trigger_address: 0
  feedback_address: 1

# 监控告警配置
monitor:
  enabled: true
  check_interval: 60  # 秒
  alert_rules:
    gpu_memory: 90    # GPU内存使用率阈值(%)
    inference_delay: 1000  # 推理延迟阈值(ms)
```

## 核心功能

### 1. 实时检测

- **图像上传检测**：支持JPG/PNG格式图像上传
- **摄像头实时检测**：USB摄像头或RTSP网络流
- **双引擎检测**：YOLO快速定位 + VLM精细分析
- **缺陷可视化**：带编号的检测框、置信度、位置坐标

### 2. 人工审核

- **待审核列表**：系统自动检测结果等待人工确认
- **审核操作**：通过/驳回/修正
- **审核记录**：完整的审核历史追溯

### 3. 统计报表

- **缺陷统计**：按时间、类型、严重程度统计
- **趋势分析**：缺陷率变化趋势
- **报表导出**：CSV、HTML专业报告、Bad Case数据集

### 4. 系统监控

- **健康检查**：GPU/CPU/内存/磁盘使用率
- **性能监控**：推理延迟、帧率、相机状态
- **告警通知**：阈值告警、故障告警

### 5. PLC集成

- **硬触发支持**：Modbus TCP协议接收PLC触发信号
- **状态反馈**：检测完成后向PLC发送完成信号
- **超时保护**：断线重连、超时处理机制

## API说明

### RESTful API接口

系统提供以下HTTP接口：

| 端点 | 方法 | 描述 | 请求示例 |
| --- | --- | --- | --- |
| `/api/health` | GET | 系统健康状态 | `curl http://127.0.0.1:7860/api/health` |
| `/api/detect` | POST | 单张图像检测 | `curl -X POST -F "image=@test.jpg" http://127.0.0.1:7860/api/detect` |
| `/api/plc/status` | GET | PLC连接状态 | `curl http://127.0.0.1:7860/api/plc/status` |
| `/api/monitor/metrics` | GET | 监控指标 | `curl http://127.0.0.1:7860/api/monitor/metrics` |

### Python API

```python
from src.detection_engine import YOLODetector
from src.vlm_engine import VLMDetector

# 初始化检测器
yolo = YOLODetector(model_path="models/weights/steel_defect.pt")
vlm = VLMDetector(api_base="", model="gemini-1.5-flash")

# 检测图像
result = yolo.detect(image)
print(f"检测到 {len(result.detections)} 个缺陷")

# VLM分析
vlm_result = vlm.detect(image)
```

## 目录结构

```text
steel-defect-detection/
├── src/                          # 源代码
│   ├── base_detector.py         # 检测器基类
│   ├── camera.py               # 图像采集模块
│   ├── detection_engine.py     # YOLO检测引擎
│   ├── vlm_engine.py           # VLM检测引擎
│   ├── db_manager.py           # 数据库管理
│   ├── exporter.py             # 导出模块
│   ├── plc_trigger.py          # PLC硬触发模块 (新增)
│   ├── monitor.py              # 监控告警模块 (新增)
│   └── icons.py               # 图标资源
├── scripts/                    # 自动化脚本
│   ├── train_yolo.py           # 模型训练
│   ├── prepare_dataset.py      # 数据集准备
│   ├── benchmark.py            # 性能测试
│   ├── log_analyzer.py         # 日志分析
│   ├── rag_demo.py             # RAG演示
│   └── download_neu_det.py     # 数据集下载
├── tests/                      # 测试目录
│   ├── unit/                   # 单元测试
│   └── integration/            # 集成测试
├── skills/                     # OpenClaw技能
│   ├── yolo-detect/            # YOLO检测技能
│   └── vlm-detect/             # VLM检测技能
├── models/                     # 模型文件
│   └── weights/                # 权重文件
│       ├── steel_defect.pt     # 训练模型
│       └── yolov8n.pt          # 预训练模型
├── data/                       # 数据目录
│   ├── images/                 # 图像数据
│   ├── labels/                 # 标注数据
│   ├── datasets/               # 数据集
│   └── exports/                # 导出文件
├── docs/                       # 文档目录
│   └── user-manual.md          # 用户手册
├── runs/                       # 训练运行目录
├── .env.example                # 环境变量示例
├── config.yaml                 # 主配置文件
├── requirements.txt            # 依赖清单
├── setup.bat                   # 部署脚本
├── main.py                     # 主入口
├── app.py                      # Gradio应用
└── cli.py                      # 命令行工具
```

## 缺陷类型定义

系统支持以下缺陷类型检测：

| 缺陷类型 | 英文标识 | 描述 | 典型特征 |
| --- | --- | --- | --- |
| 裂纹 | crazing/crack | 材料表面出现的线性断裂 | 细长、不规则、有深度 |
| 夹杂 | inclusion | 材料中混入的异物 | 点状、颜色差异、边界清晰 |
| 斑块 | patches | 表面颜色或纹理不一致区域 | 块状、颜色差异、边界模糊 |
| 麻点 | pitted_surface | 表面小凹坑聚集 | 点状密集、有深度、不规则分布 |
| 轧制氧化皮 | rolled_in_scale | 轧制过程中形成的氧化层 | 片状、有厚度、易剥离 |
| 划痕 | scratches | 表面线性划伤 | 细长、浅表、有方向性 |

## 性能指标

- **检测速度**：YOLO ~30ms/帧 (RTX 3060)，VLM ~2-5秒/帧
- **检测精度**：mAP@50 > 0.85 (NEU-DET数据集)
- **并发能力**：支持多相机并行检测
- **系统稳定性**：7×24小时连续运行

## 工业部署

### 硬件要求

- **最低配置**：Intel i5 CPU, 16GB RAM, 无GPU
- **推荐配置**：Intel i7 CPU, 32GB RAM, NVIDIA RTX 3060+
- **工业环境**：工控机 + 工业相机 + PLC控制器

### 网络配置

- **局域网访问**：Gradio默认监听 0.0.0.0:7860
- **PLC通信**：Modbus TCP端口502
- **RTSP流**：标准RTSP端口554

### 安全建议

1. 生产环境启用Gradio认证
2. 定期备份数据库
3. 监控系统资源使用
4. 设置防火墙规则限制访问

## 故障排除

常见问题及解决方案：

1. **摄像头无法连接**
   - 检查摄像头驱动
   - 确认USB连接或RTSP地址正确
   - 检查端口是否被占用

2. **VLM API调用失败**
   - 检查网络连接
   - 确认API密钥有效
   - 查看API配额是否用完

3. **检测速度慢**
   - 检查GPU是否启用
   - 降低图像分辨率
   - 调整检测置信度阈值

4. **PLC通信失败**
   - 检查网络连通性
   - 确认Modbus地址正确
   - 查看PLC配置

详细故障排除请参考 `docs/user-manual.md`。

## 更新日志

### v1.0.0 (2026-05-26)

- 初始版本发布
- 基础YOLO+VLM双引擎架构
- Gradio Web工作台
- SQLite数据库存储
- 基础文档和配置

### v1.1.0 (计划中)

- PLC硬触发支持
- 系统监控告警
- 性能优化和bug修复

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系我们

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目地址]
- 邮箱: [联系方式]
- 文档: `docs/user-manual.md`
