# 钢铁表面缺陷检测系统 - 用户手册

## 目录
- [系统概述](#系统概述)
- [快速入门](#快速入门)
- [Gradio工作台使用指南](#gradio工作台使用指南)
  - [实时检测页](#实时检测页)
  - [人工审核页](#人工审核页)
  - [统计报表页](#统计报表页)
  - [系统监控页](#系统监控页)
- [检测流程详解](#检测流程详解)
- [审核流程规范](#审核流程规范)
- [报表导出功能](#报表导出功能)
- [PLC硬触发配置](#plc硬触发配置)
- [系统监控与告警](#系统监控与告警)
- [常见故障排除](#常见故障排除)
- [高级配置](#高级配置)
- [维护与备份](#维护与备份)

## 系统概述

钢铁表面缺陷检测系统是一个基于深度学习的工业级智能质检平台，专为钢铁制造企业设计。系统通过YOLO目标检测算法和视觉大模型（VLM）实现钢铁表面缺陷的自动化检测、分类和分析。

**核心价值：**
- ✅ **替代人工目检**：7×24小时不间断检测，提高检测效率
- ✅ **降低漏检率**：双引擎检测，缺陷检出率>95%
- ✅ **数据追溯**：完整的检测记录和审核历史
- ✅ **工艺优化**：缺陷根因分析，指导生产工艺改进

## 快速入门

### 系统要求
- **操作系统**：Windows 10/11 或 Linux (Ubuntu 18.04+)
- **Python版本**：3.8 - 3.11
- **硬件要求**：
  - CPU：Intel i5 8代以上 或 AMD Ryzen 5以上
  - 内存：16GB以上
  - 存储：50GB可用空间
  - GPU（推荐）：NVIDIA RTX 3060 8GB以上

### 安装步骤

1. **下载安装包**
```bash
# 从GitHub下载
git clone https://github.com/your-repo/steel-defect-detection.git
cd steel-defect-detection
```

2. **安装依赖**
```bash
# 创建虚拟环境
python -m venv venv

# Windows激活
venv\Scripts\activate

# Linux/macOS激活
source venv/bin/activate

# 安装依赖包
pip install -r requirements.txt
```

3. **配置环境**
```bash
# 复制环境变量模板
copy .env.example .env

# 编辑.env文件，设置以下关键参数
# VLM_API_KEY=your_gemini_api_key_here
# PLC_HOST=192.168.1.100
# PLC_PORT=502
```

4. **下载模型权重**
```bash
# 自动下载预训练模型
python scripts/download_weights.py
```

5. **启动系统**
```bash
# 启动Gradio工作台
python main.py

# 或使用命令行模式
python main.py --mode cli
```

6. **访问Web界面**
打开浏览器，访问：`http://127.0.0.1:7860`

## Gradio工作台使用指南

### 实时检测页

#### 1. 图像上传检测
**操作步骤：**
1. 点击"选择文件"按钮或拖拽图像到上传区域
2. 选择检测模式：
   - **YOLO快速筛查**：快速定位缺陷位置
   - **VLM精细分析**：详细分析缺陷类型和严重程度
   - **一键全流程**：完整执行YOLO→VLM→RAG分析流程
3. 查看检测结果：
   - 左侧：带标注框的检测图像
   - 右侧：详细的检测报告，包括缺陷类型、位置、置信度

**参数说明：**
- **检测灵敏度**：控制YOLO检测的置信度阈值（0.01-0.50）
  - 值越低：检出更多缺陷，可能增加误报
  - 值越高：检出更精准，可能漏检轻微缺陷
- **推荐设置**：0.05（平衡精度和召回率）

#### 2. 摄像头实时检测
**操作步骤：**
1. 选择信号源类型：
   - **USB摄像头**：连接本地工业相机
   - **RTSP网络流**：连接网络摄像头
2. 配置参数：
   - 摄像头编号（USB摄像头）
   - RTSP地址（网络摄像头）
   - 分辨率（推荐1280×720）
3. 点击"连接摄像头"开始预览
4. 使用"截取快照"获取当前帧
5. 切换到"实时检测"页分析快照

**注意事项：**
- RTSP地址格式：`rtsp://username:password@ip:port/stream`
- 确保网络摄像头支持RTSP协议
- 首次连接可能需要等待几秒钟

### 人工审核页

#### 1. 审核流程
1. 系统自动检测后，结果进入"待审核"列表
2. 审核员查看检测结果和原始图像
3. 做出审核决定：
   - **审核通过**：确认系统检测结果正确
   - **驳回修正**：系统检测有误，需要修正
4. 填写审核意见（可选）

#### 2. 批量审核
- **全部通过**：一键通过所有待审核记录
- **按时间筛选**：只审核指定时间段的记录
- **按缺陷类型筛选**：只审核特定缺陷类型的记录

#### 3. 审核记录查看
- 查看历史审核记录
- 导出审核报告
- 统计审核准确率

### 统计报表页

#### 1. 数据统计
1. 选择时间范围（开始日期和结束日期）
2. 点击"生成统计报告"
3. 查看以下统计指标：
   - 检测总数
   - 缺陷检出率
   - 各缺陷类型分布
   - 缺陷严重程度分布
   - 检测时间趋势

#### 2. 报表导出
**支持导出格式：**
- **CSV文件**：原始数据，用于Excel分析
- **HTML报告**：专业格式报告，包含图表和图像
- **Bad Case数据集**：误检/漏检样本，用于模型优化

**导出步骤：**
1. 选择时间范围
2. 点击对应导出按钮
3. 下载生成的文件

### 系统监控页

#### 1. 健康状态监控
- **GPU状态**：使用率、显存、温度
- **CPU状态**：使用率、温度、频率
- **内存状态**：使用率、可用内存
- **磁盘状态**：使用率、IO速度
- **网络状态**：连接状态、延迟

#### 2. 性能监控
- **推理延迟**：YOLO和VML检测时间
- **帧率**：摄像头采集帧率
- **系统负载**：当前并发任务数

#### 3. 告警管理
- **告警规则配置**：设置各项指标的告警阈值
- **告警历史**：查看历史告警记录
- **告警通知**：配置邮件、钉钉、微信通知

## 检测流程详解

### 标准检测流程
```
图像输入 → 预处理 → YOLO快速筛查 → 缺陷判断 → VLM精细分析 → RAG根因分析 → 人工审核 → 结果存储
```

### 1. 图像预处理
- **尺寸调整**：统一调整为640×640
- **归一化**：像素值归一化到0-1范围
- **增强处理**：可选的数据增强（对比度、亮度调整）

### 2. YOLO快速筛查
- **检测速度**：~30ms/帧（RTX 3060）
- **检测精度**：mAP@50 > 0.85
- **输出结果**：缺陷位置、类型、置信度

### 3. VLM精细分析
**触发条件：**
- YOLO检测到缺陷
- 用户手动触发VLM分析

**分析内容：**
- 缺陷详细描述
- 严重程度评估（minor/moderate/severe）
- 位置精确描述
- 工艺影响分析

### 4. RAG根因分析
**分析内容：**
- 缺陷产生原因
- 工艺参数建议
- 预防措施
- 相关标准参考

## 审核流程规范

### 审核员职责
1. **准确性审核**：确认系统检测结果是否正确
2. **完整性审核**：检查是否漏检重要缺陷
3. **一致性审核**：确保审核标准统一
4. **记录保存**：完整记录审核过程和结果

### 审核标准
#### 通过标准（满足以下所有条件）
- 系统检测到的缺陷真实存在
- 缺陷类型分类正确
- 缺陷位置标注准确
- 置信度>0.3

#### 驳回标准（满足以下任一条件）
- 系统误检（将正常区域标记为缺陷）
- 缺陷类型分类错误
- 严重漏检（重要缺陷未检出）
- 位置标注严重偏差

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