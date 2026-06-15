# 钢铁表面缺陷智能检测系统 — 部署与运维手册

本手册旨在指导系统管理员和现场实施工程师（IMPL）在生产环境（如现场工控机）中安装、配置、优化及托管“钢铁表面缺陷智能检测系统”。

---

## 一、 系统环境要求

### 1. 硬件配置推荐

| 硬件 | 最低配置 (CPU模式) | 推荐配置 (GPU加速模式) | 备注 |
| :--- | :--- | :--- | :--- |
| **处理器 (CPU)** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / Xeon 或同等处理器 | 保证并发处理与多线程图像拉取 |
| **内存 (RAM)** | 16 GB | 32 GB | 频繁读写与图像队列缓存 |
| **显卡 (GPU)** | 无 (核显) | NVIDIA GeForce RTX 3060 / 4060 或以上 | 必须支持 CUDA 核心，显存 $\ge$ 8GB |
| **存储 (SSD)** | 256 GB SATA SSD | 1 TB NVMe M.2 SSD | 用于存储历史缺陷大图及 SQLite WAL 日志 |
| **相机接入** | 1 路 USB 相机 / 模拟Preset | 1~2 路 千兆以太网工业相机 (RTSP流) | 支持硬件光电触发信号接入 |

### 2. 软件运行环境

* **操作系统**：Windows 10 / Windows 11 / Windows Server 2019 或以上 (64位)
* **Python 版本**：Python 3.12.x (推荐 3.12.0)
* **C++ 推理支持**：Microsoft Visual C++ Redistributable (2015-2022)
* **网络连接**：封闭工业局域网，需在边缘网关配置临时外网出方向规则，允许访问：
  * 阿里云百炼 API 域名 (`dashscope.aliyuncs.com`)
  * Google AI Studio / Gemini API 域名 (如使用)

---

## 二、 GPU加速环境配置 (CUDA & cuDNN)

如需启用 GPU 推理以达到最优吞吐量（FPS），请按以下步骤配置显卡驱动及加速库：

1. **安装 NVIDIA 显卡驱动**：
   * 前往 NVIDIA 官网下载并安装对应显卡的最新版 Studio/Game Ready 驱动。
2. **安装 CUDA Toolkit 12.1**：
   * 下载地址：[CUDA Toolkit 12.1 Archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)
   * 安装类型选择“自定义安装”，勾选全部组件，并确认环境变量 `CUDA_PATH` 已自动添加到系统中。
3. **配置 cuDNN 8.9**：
   * 下载对应 CUDA 12.x 的 cuDNN 压缩包。
   * 解压后，将 `bin\`, `include\`, `lib\` 目录下的所有文件复制到 CUDA 安装目录（默认路径：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\`）的对应子目录下。
4. **验证安装**：
   * 打开 PowerShell 运行：
     ```powershell
     nvcc -V
     nvidia-smi
     ```
     确保输出正确的 CUDA 版本号与 GPU 设备状态。

---

## 三、 系统部署与安装步骤

### 1. 克隆/解压项目源码
将项目代码解压至工控机的固定工作目录，例如 `D:\steel-defect-detection\`。

### 2. 初始化 Python 虚拟环境
在项目根目录下，使用 PowerShell 创建并激活虚拟环境：
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\Activate.ps1
```

### 3. 安装依赖包
在激活的虚拟环境中，通过 `pip` 安装项目所需的依赖。国内环境推荐使用清华源加速：
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 验证依赖与测试
运行单元测试，确认基础模块（数据库、推理引擎、导出模块等）正常：
```powershell
$env:PYTHONPATH="."
.venv\Scripts\pytest
```

---

## 四、 关键配置文件详解

系统运行依赖于两个核心配置文件：根目录下的 `.env`（敏感环境变量）与 `config.yaml`（系统运行参数）。

### 1. 环境变量配置 (`.env`)
在项目根目录下创建 `.env` 文件。参考配置模板如下：
```env
# 基础服务配置
PORT=8000
HOST=0.0.0.0
DEBUG=false

# 数据库配置
DB_PATH=data/inspection.db

# 大模型 API 密钥 (按需配置)
DASHSCOPE_API_KEY=sk-your-aliyun-dashscope-key-here
GEMINI_API_KEY=AIzaSy-your-gemini-key-here

# 系统会话密钥
SECRET_KEY=steel_defect_secure_secret_key_2026
```

### 2. 参数配置文件 (`config.yaml`)
此文件控制 YOLO 阈值、VLM 降级开关、相机流地址等。模板如下：
```yaml
system:
  name: "SteelEye 钢铁表面缺陷检测系统"
  version: "4.2.0-Pro"
  log_level: "INFO"

detector:
  yolo_model_path: "models/weights/yolo26n.pt" # 推荐使用轻量化高性能 yolo26n.pt 模型
  confidence_threshold: 0.25 # YOLO 起步过滤阈值
  iou_threshold: 0.45 # NMS 重叠过滤阈值
  device: "cuda:0" # 显卡设备，如无显卡则配置为 "cpu"

vlm:
  enable: true # 是否开启双引擎协同复核
  provider: "qwen" # 大模型提供商: qwen 或 gemini
  model_name: "qwen-vl-max"
  timeout_seconds: 8.0 # 超时降级阈值（秒）
  concurrency_limit: 3 # 并发请求上限

camera:
  mode: "simulate" # 采集模式: simulate (模拟标样) 或 rtsp (实时相机)
  rtsp_url: "rtsp://admin:admin12345@192.168.1.64:554/h264/ch1/main/av_stream"
  fps_limit: 30
```

---

## 五、 数据库调优 (SQLite WAL 模式)

为保证产线高速检测写入与质检大屏、审核端高频读取之间不发生锁库冲突，系统底层已默认启用 SQLite **WAL (Write-Ahead Logging) 写入预写日志模式**。

如果手动初始化或维护数据库，可执行以下命令手动调优性能：
```sql
-- 启用 WAL 模式
PRAGMA journal_mode = WAL;

-- 提高读写同步性能
PRAGMA synchronous = NORMAL;

-- 设置缓存大小（缓存 10000 页）
PRAGMA cache_size = -10000;
```
*注：在 WAL 模式下，系统会在 `data/` 目录下生成 `inspection.db-shm` 和 `inspection.db-wal` 临时文件，此为正常行为，请勿手动删除。*

---

## 六、 Windows 系统服务托管与开机自启动

为了在无人看守的工控机上保障服务 24 小时稳定运行，推荐使用 **NSSM (Non-Sucking Service Manager)** 将后台服务包装为 Windows 系统服务。

### NSSM 安装与配置步骤：

1. 下载 NSSM 软件并解压至系统目录（如 `C:\Windows\System32\`）。
2. 使用管理员权限打开 PowerShell，运行 NSSM 注册服务：
   ```powershell
   nssm install SteelEyeService
   ```
3. 在弹出的图形界面中进行如下配置：
   * **Path** (可执行程序)：`D:\steel-defect-detection\.venv\Scripts\python.exe` (指向虚拟环境中的 python)
   * **Startup directory** (工作目录)：`D:\steel-defect-detection`
   * **Arguments** (启动参数)：`server.py`
4. 切换到 **Details** 标签页：
   * **Display name**：`SteelEye Detector Backend`
   * **Startup type**：`Automatic` (自动启动)
5. 切换到 **Environment** 标签页，填入运行环境变量：
   ```env
   PYTHONPATH=.
   ```
6. 点击 **Install service** 按钮。
7. 在 Windows“服务”管理器（`services.msc`）中找到 `SteelEyeService`，启动该服务，并确保其随系统开机自启动。

---

## 七、 异常监控与日志排查

系统运行日志默认输出至终端并持久化至 `logs/` 目录下。

### 常用运维排查指令：

* **检查进程是否存活**：
  ```powershell
  Get-Process -Name python | Format-Table Id, CPU, WorkingSet
  ```
* **检查网络端口占用 (8000端口)**：
  ```powershell
  Get-NetTCPConnection -LocalPort 8000
  ```
* **查看最新运行日志**：
  在 PowerShell 中实时跟踪后台日志输出：
  ```powershell
  Get-Content -Path "logs/app.log" -Tail 50 -Wait
  ```
