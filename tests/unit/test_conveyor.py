"""
单元测试 - 验证数字孪生虚拟流水线传送带与 AppState 新增状态方法。
"""

import pytest
import numpy as np
from app import AppState, render_conveyor_belt

def test_conveyor_history_init():
    """验证 AppState 初始化时包含默认的跑马灯卡片记录"""
    state_instance = AppState()
    assert len(state_instance.conveyor_history) == 4
    assert state_instance.conveyor_counter == 104
    
    # 验证卡片的基本结构
    card = state_instance.conveyor_history[0]
    assert "id" in card
    assert "status" in card
    assert "details" in card
    assert "time" in card

def test_add_conveyor_sheet():
    """验证 add_conveyor_sheet 能成功追加并保留最多 8 条记录"""
    state_instance = AppState()
    
    # 连续追加 6 条记录（原本 4 条，总共将有 10 条，但由于上限 8 条，最旧的会被抛出）
    for i in range(6):
        sheet_id = state_instance.add_conveyor_sheet("pass", f"测试合格-{i}")
        assert sheet_id == f"Plate-{105+i:03d}"
        
    assert len(state_instance.conveyor_history) == 8
    
    # 验证最旧的 Plate-101 和 Plate-102 已被踢出，最新的包含 Plate-110
    ids = [item["id"] for item in state_instance.conveyor_history]
    assert "Plate-101" not in ids
    assert "Plate-102" not in ids
    assert "Plate-110" in ids

def test_render_conveyor_belt():
    """验证 render_conveyor_belt 生成合法的 HTML 且包含指示器状态"""
    # 这里需要临时访问全局 state
    import app
    # 备份原有 state
    old_state = app.state
    
    try:
        app.state = AppState()
        
        # 1. 默认渲染 (done 状态)
        html_done = render_conveyor_belt("done")
        assert "conveyor-panel" in html_done
        assert "steel-conveyor-belt" in html_done
        assert "Plate-104" in html_done
        assert "decision-node completed" in html_done # YOLO、VLM、RAG 均完成
        
        # 2. 会诊渲染 (yolo 状态)
        html_yolo = render_conveyor_belt("yolo")
        assert "decision-node active" in html_yolo # YOLO 活跃中
        
        # 3. 待处理队列渲染
        app.state.defect_task_queue.put({"task": "dummy"})
        html_queue = render_conveyor_belt("done")
        assert "待会诊排队: 1 帧" in html_queue
        
    finally:
        # 还原 state
        app.state = old_state
