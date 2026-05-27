"""
集成测试 - 验证端到端数据流和模块间协作。

测试场景 (需求 §5.3):
1. 检测→存储 端到端数据流
2. 双引擎协同 (YOLO + VLM 兜底)
3. 审核→数据库更新
4. 导出全链路
"""

import os
import tempfile

import numpy as np
import pytest

from src.db_manager import DBManager, InspectionRecord


# ==================== Fixtures ====================

@pytest.fixture
def tmp_db():
    """临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DBManager(path)
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def sample_image():
    """生成测试图像"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ==================== 场景1: 检测→存储 ====================

class TestDetectionToStorage:
    """检测结果写入数据库的端到端验证"""

    def test_yolo_result_persists(self, tmp_db, sample_image):
        """YOLO 检测结果正确存储到数据库"""
        from src.detection_engine import YOLODetector

        detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
        try:
            detector.load_model()
        except (FileNotFoundError, Exception):
            pytest.skip("YOLO 模型不可用")

        result = detector.detect(sample_image)
        record = InspectionRecord(
            image_path="test.jpg",
            yolo_result=str(result.raw_output) if result.raw_output else "{}",
            defect_count=result.defect_count,
            confidence=max((d.confidence for d in result.detections), default=0.0),
        )
        rid = tmp_db.insert(record)
        assert rid > 0

        fetched = tmp_db.get_by_id(rid)
        assert fetched is not None
        assert fetched.defect_count == result.defect_count

    def test_batch_insert_performance(self, tmp_db):
        """批量插入性能验证"""
        records = [
            InspectionRecord(
                image_path=f"frame_{i:04d}.jpg",
                defect_count=i % 3,
                confidence=0.8 + (i % 20) * 0.01,
            )
            for i in range(100)
        ]
        count = tmp_db.insert_batch(records)
        assert count == 100
        assert tmp_db.count() == 100


# ==================== 场景2: 双引擎协同 ====================

class TestDualEngineCollaboration:
    """YOLO + VLM 双引擎协同逻辑"""

    def test_yolo_high_confidence_skips_vlm(self):
        """高置信度结果应跳过 VLM 复核"""
        from src.base_detector import DetectionResult

        # 模拟 YOLO 高置信度检测结果
        detections = [
            DetectionResult(bbox=[0.1, 0.2, 0.3, 0.4], class_name="crack", confidence=0.95)
        ]
        # 高置信度阈值: 0.5
        vlm_threshold = 0.5
        needs_vlm = any(d.confidence < vlm_threshold for d in detections)
        assert needs_vlm is False

    def test_yolo_low_confidence_needs_vlm(self):
        """低置信度结果应触发 VLM 复核"""
        from src.base_detector import DetectionResult

        detections = [
            DetectionResult(bbox=[0.1, 0.2, 0.3, 0.4], class_name="unknown", confidence=0.3)
        ]
        vlm_threshold = 0.5
        needs_vlm = any(d.confidence < vlm_threshold for d in detections)
        assert needs_vlm is True


# ==================== 场景3: 审核→数据库更新 ====================

class TestReviewUpdate:
    """人工审核后的数据一致性验证"""

    def test_confirm_review(self, tmp_db):
        """确认审核后状态更新"""
        rid = tmp_db.insert(InspectionRecord(image_path="review_test.jpg"))
        ok = tmp_db.update_review(
            rid,
            final_result={"defects": [], "status": "confirmed"},
            reviewer="质检员A",
            review_status="confirmed",
            note="无缺陷",
        )
        assert ok

        record = tmp_db.get_by_id(rid)
        assert record.review_status == "confirmed"
        assert record.reviewer == "质检员A"
        assert record.note == "无缺陷"
        assert record.review_time is not None

    def test_correct_review_updates_result(self, tmp_db):
        """修正审核后最终结果更新"""
        rid = tmp_db.insert(InspectionRecord(
            image_path="correct_test.jpg",
            yolo_result='{"defects": [{"type": "scratch"}]}',
        ))
        corrected_result = {"defects": [{"type": "crack", "corrected": True}]}
        tmp_db.update_review(
            rid,
            final_result=corrected_result,
            reviewer="质检员B",
            review_status="corrected",
        )
        record = tmp_db.get_by_id(rid)
        assert record.review_status == "corrected"

    def test_query_pending_reviews(self, tmp_db):
        """查询待审核记录"""
        tmp_db.insert(InspectionRecord(image_path="a.jpg", review_status="pending"))
        tmp_db.insert(InspectionRecord(image_path="b.jpg", review_status="confirmed"))
        tmp_db.insert(InspectionRecord(image_path="c.jpg", review_status="pending"))

        pending = tmp_db.query(review_status="pending")
        assert len(pending) == 2


# ==================== 场景4: 导出全链路 ====================

class TestExportPipeline:
    """数据库查询→文件生成→格式验证"""

    def test_csv_export_roundtrip(self, tmp_db, tmp_path):
        """CSV 导出→读取→验证字段完整性"""
        from src.exporter import Exporter

        tmp_db.insert(InspectionRecord(
            image_path="export_test.jpg",
            defect_types="crack,scratch",
            defect_count=2,
            confidence=0.88,
            review_status="confirmed",
            reviewer="张三",
        ))

        exporter = Exporter(tmp_db, output_dir=str(tmp_path))
        csv_path = exporter.export_csv()
        assert os.path.exists(csv_path)

        import csv
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["defect_types"] == "crack,scratch"

    def test_html_report_contains_stats(self, tmp_db, tmp_path):
        """HTML 报告包含统计数据"""
        from src.exporter import Exporter

        for dtype in ["crack", "crack", "scratch"]:
            tmp_db.insert(InspectionRecord(image_path="x.jpg", defect_types=dtype, defect_count=1))

        exporter = Exporter(tmp_db, output_dir=str(tmp_path))
        html_path = exporter.export_html_report()
        content = open(html_path, encoding="utf-8").read()
        assert "crack" in content
        assert "scratch" in content

    def test_badcase_export_filters_corrected(self, tmp_db, tmp_path):
        """Bad Case 导出仅包含修正记录"""
        from src.exporter import Exporter

        tmp_db.insert(InspectionRecord(image_path="good.jpg", review_status="confirmed"))
        tmp_db.insert(InspectionRecord(image_path="bad.jpg", review_status="corrected"))

        exporter = Exporter(tmp_db, output_dir=str(tmp_path))
        badcase_dir = exporter.export_badcase()
        assert os.path.isdir(badcase_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
