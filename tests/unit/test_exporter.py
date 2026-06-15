"""
导出模块单元测试。
"""

import os
import tempfile
import pytest

from src.db_manager import DBManager, InspectionRecord
from src.exporter import Exporter


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_mgr = DBManager(path)
    # 插入测试数据
    for i in range(3):
        db_mgr.insert(InspectionRecord(
            image_path=f"test_{i}.jpg",
            defect_types="crack" if i == 0 else "scratch",
            defect_count=1,
            confidence=0.5 + i * 0.2,
            review_status="confirmed" if i < 2 else "pending",
        ))
    yield db_mgr
    db_mgr.close()
    os.unlink(path)


@pytest.fixture
def exporter(db):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Exporter(db, output_dir=tmpdir)


class TestExporter:
    def test_export_csv(self, exporter):
        path = exporter.export_csv()
        assert os.path.exists(path)
        with open(path, encoding="utf-8-sig") as f:
            content = f.read()
            assert "crack" in content or "scratch" in content

    def test_export_html_report(self, exporter):
        path = exporter.export_html_report()
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
            assert "<html" in content.lower()
            assert "缺陷" in content

    def test_export_badcase(self, exporter):
        path = exporter.export_badcase()
        assert os.path.exists(path)
        assert path.endswith(".zip")

    def test_export_csv_with_time_range(self, exporter):
        path = exporter.export_csv(
            start_time="2020-01-01",
            end_time="2030-12-31",
        )
        assert os.path.exists(path)

    def test_empty_db_export(self, db):
        """空数据库导出不会崩溃"""
        fd, empty_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        empty_db = DBManager(empty_path)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                exp = Exporter(empty_db, output_dir=tmpdir)
                csv_path = exp.export_csv()
                assert os.path.exists(csv_path)
                html_path = exp.export_html_report()
                assert os.path.exists(html_path)
        finally:
            empty_db.close()
            os.unlink(empty_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
