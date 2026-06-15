# 钢铁表面缺陷检测系统 — 增量学习飞轮与深度 RAG 融合方案

本方案旨在实现双引擎检测系统的数据闭环（Data Flywheel）与冶金知识库（RAG）深度融合，打通“推理-审核-归档-重训-部署”的全链路，并提升缺陷成因分析的专业水平。

## User Review Required

> [!IMPORTANT]
> **关于后台增量训练的资源消耗与锁定问题：**
> 在工控机（如单张 RTX 4060 GPU）环境下，启动后台 YOLO 训练（`scripts/train_yolo.py`）会占用大量的显存与 GPU 算力。如果在训练期间产线工业相机仍在持续发送检测图像，可能会导致 YOLO 实时推理延迟骤增（甚至 OOM 崩溃）。
> **应对方案：** 系统在检测到有后台训练任务运行时，会自动将实时检测降级为“CPU 推理”模式，以保证产线实时图像流不中断；训练完成后再切换回 GPU 模式并加载新模型。

## Open Questions

1. **增量训练的数据集形式**：是直接使用本次收集的少量 Bad Case 进行轻量级 Epoch 微调（容易导致灾难性遗忘），还是将 Bad Case 合并进 NEU-DET 原始 1800 张图片的底库中进行混合全量训练（耗时约 10~30 分钟）？
   * *(推荐)* 默认采用合并底库的混合全量训练，以保证模型对经典缺陷的识别精度不会衰退。

---

## Proposed Changes

### 后端 API 与算法模块

#### [MODIFY] [server.py](file:///f:/steel-defect-detection/server.py)
- **RAG 融合**：在 `/api/detect` 路由中，当 YOLO 检测出缺陷后，系统自动从 `KNOWLEDGE_BASE` 中匹配相关缺陷类型的成因与对策，并将参考知识嵌入到给 VLM (Gemini 3.5 Flash / Qwen-VL) 的 Prompt 中，生成标准格式的物理成因和停机处置意见。
- **后台训练接口**：新增 `/api/train/start` 和 `/api/train/status` 两个 API 端点。
  - `/api/train/start`：将数据库中所有经人工修改确权（`review_status = 'corrected'`）的 Bad Case 图像和 XML/YOLO 标注导出，合并至 NEU-DET 原始数据目录，使用子进程（`subprocess.Popen`）在后台执行 `scripts/train_yolo.py`。
  - `/api/train/status`：实时读取训练输出的日志，解析并返回当前的 Epoch 进度、mAP 指标及训练状态。

#### [MODIFY] [src/db_manager.py](file:///f:/steel-defect-detection/src/db_manager.py)
- 新增 `get_audited_dataset()` 方法，用于筛选未导出的、状态为已审核修正的缺陷记录，提取其图像文件与调整后的标注框，将其格式化输出为 YOLO 标准训练格式（`txt` 标注）。

---

### 前端大屏交互模块

#### [MODIFY] [App.tsx](file:///f:/steel-defect-detection/frontend/src/App.tsx)
- **AI 工程师专用 Tab**：在 `userRole === 'ai_engineer'` 时，大屏增加“数据飞轮与模型迭代”控制面板：
  - 显示当前数据库中“待重训的 Bad Case 样本数量”。
  - 提供“一键启动模型重训”按钮。
  - 渲染一个实时进度条，展示后台 YOLOv8 训练的 Epoch 进度与 Loss 曲线变化。
- **OpenClaw 浮动助手机制**：在大屏右下角引入一个半透明玻璃态的“AI 工艺助理”聊天框：
  - 支持收起与展开。
  - 允许质检员或工程师输入诸如：“氧化皮是怎么产生的？”或“今天上午有严重的裂纹缺陷吗？”等自然语言指令。
  - 调用后端的 `/api/chat`，渲染结构化多段回复。

---

## Verification Plan

### Automated Tests
- 运行 FastAPI 单元测试，确保训练 API 正确响应：
  `pytest tests/unit/test_train_endpoints.py`
- 运行集成测试验证 RAG + VLM 双引擎协同检测：
  `pytest tests/integration/test_pipeline.py`

### Manual Verification
1. 登录为 `admin` 或 `inspector`，上传一张缺陷图片并进行检测。
2. 进入“审核”弹窗，人工修正缺陷类型（例如将 Pitting 修正为 Cracks），并提交审核。
3. 登录为 `ai_engineer`，进入“数据飞轮”页面，确认刚才修正的 Bad Case 已被统计。
4. 点击“开始增量训练”，验证进度条是否正常随着后台训练日志更新。
5. 训练结束后，上传一张同类型图片，验证新模型权重是否生效。
