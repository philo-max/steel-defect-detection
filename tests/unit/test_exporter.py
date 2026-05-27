"""
导出模块单元测试。
"""

import csv
import os
import tempfile

import pytest

from src.db_manager import DBManager, InspectionRecord
from src.exporter import Exporter


@pytest.fixture
def db_with_data():
    """创建含测试数据的临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DBManager(path)
    db.insert(InspectionRecord(
        image_path="test1.jpg",
        defect_types="crack",
        defect_count=1,
        confidence=0.92,
        review_status="confirmed",
    ))
    db.insert(InspectionRecord(
        image_path="test2.jpg",
        defect_types="scratch,scale",
        defect_count=2,
        confidence=0.85,
        review_status="pending",
    ))
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def exporter(db_with_data, tmp_path):
    return Exporter(db_with_data, output_dir=str(tmp_path))


class TestExporterCSV:
    def test_export_creates_file(self, exporter, tmp_path):
        path = exporter.export_csv()
        assert os.path.exists(path)
        assert path.endswith(".csv")

    def test_csv_content(self, exporter):
        path = exporter.export_csv()
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        defect_types = {r["defect_types"] for r in rows}
        assert "crack" in defect_types
        assert "scratch,scale" in defect_types

    def test_csv_custom_path(self, exporter, tmp_path):
        custom = str(tmp_path / "custom.csv")
        path = exporter.export_csv(output_path=custom)
        assert path == custom
        assert os.path.exists(custom)


class TestExporterHTML:
    def test_export_html_creates_file(self, exporter):
        path = exporter.export_html_report()
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_html_contains_defect_info(self, exporter):
        path = exporter.export_html_report()
        content = open(path, encoding="utf-8").read()
        assert "crack" in content
        assert "scratch" in content


class TestExporterBadCase:
    def test_export_badcase_creates_dir(self, exporter, tmp_path):
        path = exporter.export_badcase()
        assert os.path.isdir(path)

    def test_export_badcase_with_data(self, exporter):
        path = exporter.export_badcase()
        files = os.listdir(path)
        assert len(files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
