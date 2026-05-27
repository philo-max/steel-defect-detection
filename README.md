# 钢铁表面缺陷检测系统 V2.1

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8s-orange)](https://ultralytics.com)
[![Tests](https://img.shields.io/badge/tests-106%20passed-green)](tests/)
[![mAP50](https://img.shields.io/badge/mAP50-0.906-brightgreen)](runs/train/)

基于 **YOLOv8s + VLM + RAG 三引擎架构** 的工业级钢铁表面缺陷智能检测平台。

## 核心数据

| 指标 | 数值 |
|------|------|
| 检测模型 | YOLOv8s (11M 参数) |
| mAP50 | **0.906** (NEU-DET) |
| 推理延迟 | P50=46ms, ~143 FPS |
| 支持缺陷 | 6 类 (裂纹/夹杂/斑块/麻点/氧化皮/划痕) |
| 测试覆盖 | 106 passed, 0 failed |

## 快速开始

```bash
git clone https://github.com/philo-max/steel-defect-detection.git
cd steel-defect-detection
setup.bat          # Windows 一键部署
python app.py      # 启动工作台 → http://127.0.0.1:7860
```

## 功能特性

- ⚡ **YOLO 筛查** — mAP50=0.906, 7ms推理
- 🧠 **VLM 复核** — Gemini/Qwen 自动探测
- 📚 **RAG 根因** — 缺陷→原因→工艺建议
- 🎤 **语音控制** — 20+ 条语音指令
- 🎨 **工业UI** — 深浅双主题, 钢铁之眼Logo
- 📦 **多格式导出** — CSV / Bad Case / HTML
- 📊 **系统监控** — GPU/CPU/相机健康检查
- 🐳 **Docker** — 容器化部署支持

## 文档

- [用户手册](docs/user-manual.md)
- [运维文档](docs/operations.md)
- [需求规格说明书](docs/需求规格说明书.md)
- [GitHub](https://github.com/philo-max/steel-defect-detection)

## API

### 健康检查

```bash
curl http://127.0.0.1:7861/health
# {"status":"healthy","gpu":{"available":true,"memory_usage":78.5},...}
```

### MJPEG 视频流

```bash
http://127.0.0.1:7861/camera
```

### 逐类性能

| 类别 | mAP50 |
|------|:-----:|
| crazing 裂纹 | 0.993 |
| inclusion 夹杂 | 0.749 |
| patches 斑块 | 0.995 |
| pitted_surface 麻点 | 0.995 |
| rolled-in-scale 氧化皮 | 0.609 |
| scratches 划痕 | 0.978 |
| **总体** | **0.906** |

```yaml
# PLC 触发配置
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
