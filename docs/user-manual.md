# 钢铁表面缺陷检测系统 — 用户手册 V2.0

> 版本：V2.0 | 日期：2026-05-27 | 模型：yolov8s (mAP50=0.886)

## 目录
- [系统概述](#系统概述)
- [快速入门](#快速入门)
- [界面导览](#界面导览)
- [检测操作](#检测操作)
- [语音命令](#语音命令)
- [人工审核](#人工审核)
- [统计报表](#统计报表)
- [模型训练](#模型训练)
- [系统监控](#系统监控)
- [常见问题](#常见问题)

## 系统概述

钢铁表面缺陷检测系统 V2.0 基于 **YOLO + VLM + RAG 三引擎架构**，实现钢铁表面缺陷的自动化检测、分类与根因分析。

### 核心特性

| 特性 | 说明 |
|------|------|
| ⚡ YOLOv8s 检测 | mAP50=0.886, 推理 7ms/张, 143 FPS |
| 🧠 VLM 复核 | Gemini/Qwen 自动探测，精细缺陷分析 |
| 📚 RAG 根因 | 缺陷→原因→工艺建议，知识库检索 |
| 🎤 语音控制 | 20+ 语音指令，按住右下角麦克风说话 |
| 🎨 工业主题 | 深浅双模式，钢铁之眼 Logo，LED 状态灯 |
| 📦 多格式导出 | CSV / Bad Case / HTML 报告 |

### 技术栈

| 组件 | 选型 |
|------|------|
| 检测模型 | YOLOv8s (11M 参数) |
| 推理加速 | ONNX 导出, TensorRT 可选 |
| Web 框架 | Gradio 6.x |
| 数据库 | SQLite (WAL 模式) |
| GPU | NVIDIA RTX 5060 8GB |

## 快速入门

### 系统要求
- **OS**: Windows 11 / Ubuntu 20.04+
- **Python**: 3.12
- **GPU**: NVIDIA RTX 5060 8GB（推荐）/ RTX 3060+
- **内存**: 16GB+
- **存储**: 50GB

### 一键安装

```bash
# 克隆仓库
git clone https://github.com/philo-max/steel-defect-detection.git
cd steel-defect-detection

# 一键部署 (Windows)
setup.bat

# 或 Linux
bash setup.sh
```

### 启动系统

```bash
# Web 工作台
python app.py

# 命令行模式
python cli.py --help
```

浏览器打开 `http://127.0.0.1:7860`

## 界面导览

### 顶部导航栏
系统顶部显示：
- **钢铁之眼 Logo** — 六边形晶格 + 检测准星
- **STEEL VISION PRO** — 系统名称
- **状态指示灯** — 绿色(正常) / 蓝色(GPU) / 橙色(VLM)
- **主题切换按钮** — 🌙 深色 / ☀️ 浅色

### 四页标签
| 页面 | 功能 |
|------|------|
| 📷 实时采集 | 工业相机连接、RTSP 流预览、快照 |
| 🔍 实时检测 | 上传图像、YOLO/VLM/全流程检测 |
| 📋 人工审核 | 待审核记录、通过/驳回 |
| 📊 统计报表 | 数据统计、多格式导出 |

### 统计卡片
页面顶部四张卡片实时显示：今日检测数、检出缺陷数、待审核数、模型准确率。

## 检测操作

### 单图检测
1. 切换到「🔍 实时检测」页
2. 上传钢板表面图像（支持拖拽/剪贴板）
3. 调节**检测灵敏度**滑块（0.01=敏感, 0.50=精准）
4. 点击检测按钮：

| 按钮 | 说明 |
|------|------|
| ⚡ YOLO 快速筛查 | 本地推理，~7ms，定位缺陷位置 |
| 🧠 VLM 精细分析 | API 调用，确认缺陷类型和严重程度 |
| 🚀 一键全流程 | YOLO→VLM→RAG 三步串联，含进度条+容错 |

### 一键全流程详解
点击「🚀 一键全流程」后：
1. **阶段 1/3**: YOLO 快速筛查（本地 GPU）
2. **阶段 2/3**: VLM 精细复核（Gemini API）
3. **阶段 3/3**: RAG 根因分析（知识库检索）

任一阶段失败不影响后续。最终报告显示：
```
📊 综合检测报告
| 状态 | 总耗时 | 阶段数 | 通过 |
| 2/3 通过 | 3240 ms | 3 | 2 |
```

### 摄像头实时采集
1. 切换到「📷 实时采集」页
2. 选择信号源：USB 摄像头 / RTSP 网络流
3. 点击「▶ 连接摄像头」开始 MJPEG 推流
4. 「📸 截取快照」获取当前帧
5. 切换到检测页分析

## 语音命令

页面**右下角**有 🎤 浮动麦克风按钮。

### 使用方式
1. **按住**麦克风按钮
2. 说出指令，**松开**自动执行
3. 气泡显示识别结果和执行状态

### 支持指令

| 类别 | 可以这样说 |
|------|-----------|
| 🔍 检测 | "YOLO筛查" / "VLM分析" / "一键全流程" / "根因分析" |
| ✅ 审核 | "审核通过" / "驳回修正" / "刷新审核" |
| 📥 导出 | "导出CSV" / "导出HTML" / "导出BadCase" / "生成报告" |
| 📷 相机 | "连接摄像头" / "断开摄像头" / "截取快照" |
| 🎨 主题 | "深色模式" / "浅色模式" |
| 🧭 导航 | "打开检测页" / "去审核页" / "打开报表" |

> 支持模糊匹配："帮我做一下YOLO检测" 也能识别。
> 需要 Chrome/Edge 浏览器（Web Speech API）。

## 人工审核

### 审核流程
1. 切换到「📋 人工审核」页
2. 点击「🔄 刷新待审核列表」
3. 逐条审核：输入记录 ID → 审核人 → 备注
4. 点击「✅ 审核通过」或「❌ 驳回修正」

### 快捷键
- **刷新列表**: 语音命令 "刷新审核"

## 统计报表

### 生成报告
1. 切换到「📊 统计报表」页
2. 设置日期范围（默认全年）
3. 点击「📊 生成统计报告」

### 导出格式

| 格式 | 用途 |
|------|------|
| 📥 CSV | Excel 数据分析 |
| 🌐 HTML 报告 | 专业质检报告（含图像+坐标） |
| 📦 Bad Case | 误检/漏检样本数据集，用于模型迭代 |

## 模型训练

### 全量训练

```bash
# yolov8s 全量训练 (200 epochs, ~2h)
python scripts/train_yolo.py --model yolov8s.pt --epochs 200 --imgsz 640 --batch 16

# 快速验证 (3 epochs)
python scripts/train_yolo.py --quick
```

### 弱项专项微调

```bash
# 针对 inclusion + rolled-in_scale 高分辨率微调
python scripts/finetune_weak.py --epochs 80 --imgsz 960 --batch 8 --lr 0.0005
```

### 模型文件

| 文件 | 说明 |
|------|------|
| `models/weights/steel_defect.pt` | 当前部署模型 (22.5MB) |
| `models/weights/yolov8s_steel.pt` | yolov8s 命名副本 |
| `models/weights/steel_defect.onnx` | ONNX 导出 (42.7MB) |

## 系统监控

### 健康检查 API

```bash
curl http://127.0.0.1:7861/health
```

返回 JSON:
```json
{
  "status": "healthy",
  "gpu": {"available": true, "memory_usage": 78.5},
  "cpu": {"usage": 35.2},
  "camera": {"connected": true, "fps": 30}
}
```

### 告警规则
系统内置 10 条告警规则：GPU 内存/温度、CPU/内存/磁盘使用率、推理延迟、相机帧率等。

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| 模型加载失败 | 确认 `models/weights/steel_defect.pt` 存在 |
| VLM API 超时 | 检查网络，系统自动降级为纯 YOLO |
| 摄像头连不上 | 检查 RTSP 地址、防火墙、相机电源 |
| 语音识别无反应 | 需 Chrome/Edge 浏览器，允许麦克风权限 |
| GPU 内存不足 | 降低 batch size 或使用 ONNX 推理 |

### 审核记录要求
1. **必填项**：审核结果（通过/驳回）
2. **建议项**：审核意见（说明审核依据）
3. **可选项**：修正建议（如驳回时提供）

## 报表导出功能

### 1. CSV数据导出
**文件格式：** `inspection_export_YYYYMMDD_HHMMSS.csv`

**包含字段：**
- 检测ID
- 检测时间
- 图像路径
- 缺陷类型
- 缺陷数量
- 最大置信度
- 审核状态
- 审核人
- 审核时间

**使用场景：**
- 数据统计分析
- 质量报告生成
- 生产数据追溯

### 2. HTML专业报告
**报告内容：**
- 执行摘要
- 检测统计图表
- 缺陷分布热力图
- 典型缺陷示例
- 趋势分析
- 建议措施

**报告特点：**
- 响应式设计，支持手机查看
- 交互式图表
- 可直接打印
- 支持中英文切换

### 3. Bad Case数据集
**数据集结构：**
```
badcase_export_YYYYMMDD/
├── images/              # 原始图像
├── labels/              # 标注文件
├── metadata.json        # 元数据
└── README.txt           # 使用说明
```

**使用场景：**
- 模型优化训练
- 算法性能评估
- 误检分析
- 新缺陷类型收集

## PLC硬触发配置

### 1. 硬件连接
**所需设备：**
- PLC控制器（支持Modbus TCP）
- 工业相机
- 工控机
- 网络交换机

**连接拓扑：**
```
PLC → 以太网 → 工控机
相机 → USB/网线 → 工控机
```

### 2. 软件配置
**修改config.yaml：**
```yaml
plc:
  enabled: true
  host: "192.168.1.100"    # PLC IP地址
  port: 502                # Modbus TCP端口
  trigger_address: 0       # 触发信号寄存器地址
  feedback_address: 1      # 反馈信号寄存器地址
  timeout: 5.0             # 通信超时(秒)
  retry_interval: 1.0      # 重试间隔(秒)
  max_retries: 3           # 最大重试次数
```

### 3. 触发流程
```
PLC发送触发信号 → 系统接收信号 → 相机拍照 → 图像检测 → 结果存储 → 反馈信号给PLC
```

**时序要求：**
- 触发信号到拍照延迟：<100ms
- 检测到反馈延迟：<2秒
- 整体流程时间：<3秒

### 4. 故障处理
**常见问题：**
1. **PLC连接失败**
   - 检查网络连通性
   - 确认IP地址和端口正确
   - 检查防火墙设置

2. **触发信号丢失**
   - 检查PLC程序
   - 确认寄存器地址正确
   - 检查信号电平

3. **反馈信号未发送**
   - 检查检测流程是否完成
   - 确认反馈寄存器地址正确
   - 检查网络延迟

## 系统监控与告警

### 1. 监控指标
#### 硬件监控
- **GPU监控**：使用率、温度、显存、功率
- **CPU监控**：使用率、温度、频率、负载
- **内存监控**：使用率、可用内存、交换空间
- **磁盘监控**：使用率、IOPS、读写速度
- **网络监控**：带宽、延迟、丢包率

#### 软件监控
- **进程状态**：主要服务进程是否运行
- **服务健康**：API服务响应状态
- **数据库状态**：连接状态、查询性能
- **日志监控**：错误日志、异常告警

#### 业务监控
- **检测性能**：推理延迟、帧率、准确率
- **数据质量**：图像质量、标注质量
- **系统负载**：并发任务数、队列长度

### 2. 告警配置
**修改config.yaml：**
```yaml
monitor:
  enabled: true
  check_interval: 60  # 检查间隔(秒)
  
  alert_rules:
    # GPU告警
    gpu_memory: 90    # GPU内存使用率阈值(%)
    gpu_temperature: 85  # GPU温度阈值(℃)
    
    # CPU告警
    cpu_usage: 90     # CPU使用率阈值(%)
    cpu_temperature: 80  # CPU温度阈值(℃)
    
    # 内存告警
    memory_usage: 90  # 内存使用率阈值(%)
    
    # 磁盘告警
    disk_usage: 90    # 磁盘使用率阈值(%)
    
    # 性能告警
    inference_delay: 1000  # 推理延迟阈值(ms)
    camera_fps: 10    # 相机帧率阈值(FPS)
    
    # 业务告警
    defect_rate: 10   # 缺陷率阈值(%)
    false_positive_rate: 5  # 误检率阈值(%)
```

### 3. 告警通知
**支持的通知方式：**
- **日志告警**：写入系统日志文件
- **邮件告警**：发送到指定邮箱
- **钉钉告警**：发送到钉钉群
- **微信告警**：发送到企业微信
- **短信告警**：发送到手机（需要短信网关）

**配置示例：**
```yaml
monitor:
  notifications:
    email:
      enabled: true
      smtp_server: "smtp.example.com"
      smtp_port: 587
      username: "alert@example.com"
      password: "your_password"
      receivers: ["admin@example.com", "operator@example.com"]
    
    dingtalk:
      enabled: true
      webhook: "https://oapi.dingtalk.com/robot/send?access_token=xxx"
      secret: "your_secret"
    
    wechat:
      enabled: false  # 待实现
```

### 4. 健康检查接口
**访问地址：** `http://127.0.0.1:7860/api/health`

**返回示例：**
```json
{
  "status": "healthy",
  "timestamp": "2026-05-26T10:30:00Z",
  "components": {
    "gpu": {"status": "healthy", "usage": 45, "temperature": 65},
    "cpu": {"status": "healthy", "usage": 30, "temperature": 55},
    "memory": {"status": "healthy", "usage": 60},
    "disk": {"status": "healthy", "usage": 40},
    "camera": {"status": "healthy", "fps": 25},
    "plc": {"status": "healthy", "connected": true},
    "database": {"status": "healthy", "connections": 2}
  },
  "performance": {
    "inference_delay": 35,
    "fps": 28,
    "queue_length": 0
  }
}
```

## 常见故障排除

### 1. 系统启动失败

#### 问题：Python依赖安装失败
**解决方案：**
```bash
# 1. 升级pip
python -m pip install --upgrade pip

# 2. 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 单独安装失败包
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 问题：GPU不可用
**解决方案：**
1. 检查CUDA安装
```bash
nvcc --version  # 应该显示CUDA版本
python -c "import torch; print(torch.cuda.is_available())"  # 应该返回True
```

2. 安装对应CUDA版本的PyTorch
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. 如果GPU确实不可用，使用CPU模式
```yaml
# 修改config.yaml
yolo:
  device: "cpu"
```

### 2. 摄像头无法连接

#### 问题：USB摄像头无响应
**解决方案：**
1. 检查物理连接
2. 查看设备管理器中的摄像头状态
3. 尝试其他摄像头编号（0,1,2...）
4. 关闭其他占用摄像头的程序

#### 问题：RTSP流无法连接
**解决方案：**
1. 测试RTSP地址
```bash
# 使用VLC测试
vlc rtsp://192.168.1.100:554/stream
```

2. 检查网络连通性
```bash
ping 192.168.1.100
telnet 192.168.1.100 554
```

3. 确认摄像头支持RTSP协议

### 3. 检测结果异常

#### 问题：检测不到缺陷
**解决方案：**
1. 降低检测灵敏度
2. 检查图像质量（亮度、对比度）
3. 确认模型权重文件存在
4. 检查缺陷类型是否在支持范围内

#### 问题：误检过多
**解决方案：**
1. 提高检测灵敏度
2. 优化光照条件
3. 更新模型权重
4. 调整图像预处理参数

### 4. VLM API调用失败

#### 问题：API配额用完
**解决方案：**
1. 等待配额重置（通常24小时）
2. 更换API密钥
3. 使用本地VLM模型（如有）

#### 问题：网络超时
**解决方案：**
1. 检查网络连接
2. 增加超时时间
```yaml
vlm:
  timeout: 120  # 增加到120秒
```

3. 使用代理（如需要）
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### 5. 数据库错误

#### 问题：数据库文件损坏
**解决方案：**
1. 从备份恢复
2. 重建数据库
```bash
python scripts/init_database.py
```

3. 检查磁盘空间

#### 问题：数据库锁死
**解决方案：**
1. 重启系统
2. 删除锁文件
```bash
rm data/inspection.db-wal
rm data/inspection.db-shm
```

### 6. PLC通信故障

#### 问题：无法连接PLC
**解决方案：**
1. 检查网络配置
2. 确认PLC IP地址和端口
3. 检查防火墙设置
4. 测试Modbus通信
```bash
# 使用modbus-cli测试
modbus read --host 192.168.1.100 --port 502 0 1
```

#### 问题：触发信号不响应
**解决方案：**
1. 检查PLC程序
2. 确认寄存器地址
3. 检查信号电平
4. 查看PLC日志

## 高级配置

### 1. 多相机配置
```yaml
cameras:
  - id: "camera1"
    source: "0"
    width: 1920
    height: 1080
    trigger_mode: "plc_hardware"
    plc_trigger_id: 1
    
  - id: "camera2"
    source: "rtsp://192.168.1.101:554/stream"
    width: 1280
    height: 720
    trigger_mode: "software"
    
  - id: "camera3"
    source: "1"
    width: 640
    height: 480
    trigger_mode: "continuous"
```

### 2. 自定义缺陷类型
```yaml
defect_types:
  - name: "crazing"
    display_name: "裂纹"
    severity: "critical"
    color: "#FF0000"
    threshold: 0.3
    
  - name: "scratch"
    display_name: "划痕"
    severity: "major"
    color: "#FFA500"
    threshold: 0.2
    
  - name: "inclusion"
    display_name: "夹杂"
    severity: "minor"
    color: "#FFFF00"
    threshold: 0.1
```

### 3. 性能优化配置
```yaml
performance:
  # 图像处理
  image_quality: 85          # JPEG质量(1-100)
  resize_method: "bilinear"  # 缩放算法
  
  # 推理优化
  batch_size: 1              # 批处理大小
  half_precision: true       # FP16推理
  tensorrt: false            # TensorRT加速
  
  # 内存优化
  cache_size: 100           # 图像缓存数量
  cleanup_interval: 3600    # 清理间隔(秒)
```

### 4. 日志配置
```yaml
logging:
  level: "INFO"             # DEBUG, INFO, WARNING, ERROR
  file: "logs/system.log"
  max_size: 104857600       # 100MB
  backup_count: 10          # 保留10个备份
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # 组件日志
  components:
    camera: "INFO"
    detection: "DEBUG"
    database: "INFO"
    plc: "DEBUG"
    monitor: "INFO"
```

## 维护与备份

### 1. 日常维护
**每日检查：**
- 系统健康状态
- 磁盘空间使用情况
- 日志文件大小
- 数据库备份状态

**每周维护：**
- 清理临时文件
- 优化数据库
- 更新模型权重（如有）
- 检查系统更新

**每月维护：**
- 全面系统检查
- 性能测试
- 安全审计
- 备份验证

### 2. 数据备份
**备份策略：**
```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"      # 每天凌晨2点
  retention_days: 30         # 保留30天
  
  # 备份内容
  include:
    - "data/inspection.db"   # 数据库
    - "data/images/"         # 图像数据
    - "config.yaml"          # 配置文件
    - "models/weights/"      # 模型权重
    
  # 备份目标
  target:
    type: "local"            # local, network, cloud
    path: "D:/backups/"      # 本地路径
    # network_path: "\\\\nas\\backup\\"  # 网络路径
    # cloud_provider: "aws_s3"           # 云存储
```

**手动备份：**
```bash
python scripts/backup.py --full --compress
```

### 3. 系统更新
**更新步骤：**
1. 备份当前系统
2. 停止所有服务
3. 更新代码
```bash
git pull origin main
```
4. 更新依赖
```bash
pip install -r requirements.txt --upgrade
```
5. 迁移数据库（如有需要）
```bash
python scripts/migrate_database.py
```
6. 启动系统测试

### 4. 故障恢复
**恢复流程：**
1. 识别故障原因
2. 查看相关日志
3. 执行恢复操作
4. 验证恢复结果
5. 记录故障报告

**恢复工具：**
- `scripts/repair_database.py` - 数据库修复
- `scripts/restore_backup.py` - 备份恢复
- `scripts/reset_system.py` - 系统重置

---

## 附录

### A. 快捷键列表
- `Ctrl+U`：上传图像
- `Ctrl+D`：开始检测
- `Ctrl+R`：刷新页面
- `Ctrl+E`：导出报告
- `Ctrl+S`：保存记录
- `Ctrl+Q`：退出系统

### B. 错误代码说明
| 错误代码 | 含义 | 解决方案 |
|----------|------|----------|
| ERR-001 | 摄像头连接失败 | 检查摄像头连接和配置 |
| ERR-002 | 模型加载失败 | 检查模型文件路径和权限 |
| ERR-003 | 数据库错误 | 检查数据库连接和文件完整性 |
| ERR-004 | PLC通信超时 | 检查网络连接和PLC状态 |
| ERR-005 | 内存不足 | 关闭其他程序或增加内存 |
| ERR-006 | GPU错误 | 检查GPU驱动和CUDA安装 |
| ERR-007 | 文件权限错误 | 检查文件读写权限 |
| ERR-008 | 网络错误 | 检查网络连接和代理设置 |

### C. 技术支持
如遇无法解决的问题，请提供以下信息：
1. 系统版本信息
2. 错误日志文件
3. 复现步骤
4. 相关配置文件（脱敏后）

**联系方式：**
- 技术支持邮箱：support@example.com
- 紧急联系电话：+86-xxx-xxxx-xxxx
- 在线文档：https://docs.example.com

---

*最后更新：2026年5月26日*
*版本：v1.0.0*