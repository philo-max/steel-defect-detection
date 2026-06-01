"""
单元测试 - 验证 RAG 工业化知识库与检索生成模块。
"""

import os
import pytest
import sqlite3
from pathlib import Path

from scripts.rag_demo import query_knowledge_base, rag_analyze

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "inspection.db"

def test_database_populated():
    """验证知识库已成功导入 SQLite 数据库且非空"""
    assert DB_PATH.exists(), f"数据库文件不存在: {DB_PATH}"
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count >= 10, f"知识库条目过少: {count}"

def test_query_knowledge_base_exact():
    """验证能够精准查询特定缺陷的标准"""
    results = query_knowledge_base("crazing")
    assert len(results) > 0, "精准匹配 crazing 失败"
    
    # 验证返回结构包含所需字段
    item = results[0]
    assert "title" in item
    assert "standard_code" in item
    assert "content" in item
    assert "GB/T 1499.2" in item["standard_code"] or "crazing" in item["title"].lower()

def test_query_knowledge_base_fuzzy():
    """验证能够模糊/关键字匹配相关缺陷"""
    # 传入 VLM 描述中包含“裂纹”
    results = query_knowledge_base("unknown", vlm_desc="左上角微小的红棕色裂纹")
    assert len(results) > 0, "通过 VLM 描述模糊匹配裂纹失败"
    
    # 验证含有相关的国标
    has_crack_standard = any("crazing" in r["defect_type"] or "crack" in r["defect_type"] for r in results)
    assert has_crack_standard

def test_rag_analyze_offline_fallback():
    """验证 RAG 在离线（或 API 未加载）时的模板拼接生成结果"""
    report = rag_analyze("scratches", "连续的划伤")
    
    assert "scratches" in report or "划痕" in report
    assert "GB/T 3280" in report
    assert "📑 工业标准比对报告" in report
    assert "🔬 物理根因分析" in report
    assert "🛠️ 车间工艺纠偏动作" in report
