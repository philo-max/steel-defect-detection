"""
工具函数单元测试。
"""

import pytest
from src.utils.bbox import bbox_iou
from src.utils.severity import (
    compute_severity,
    compute_overall_status,
    compute_severity_index,
    compute_defect_density,
)


class TestBboxIOU:
    def test_perfect_overlap(self):
        box = [0.1, 0.2, 0.5, 0.6]
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box1 = [0.0, 0.0, 0.1, 0.1]
        box2 = [0.9, 0.9, 1.0, 1.0]
        assert bbox_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = [0.0, 0.0, 0.5, 0.5]
        box2 = [0.25, 0.25, 0.75, 0.75]
        iou = bbox_iou(box1, box2)
        assert 0.1 < iou < 0.5

    def test_contained_box(self):
        box1 = [0.0, 0.0, 1.0, 1.0]
        box2 = [0.2, 0.2, 0.4, 0.4]
        iou = bbox_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_empty_box(self):
        box1 = [0.5, 0.5, 0.5, 0.5]
        box2 = [0.0, 0.0, 1.0, 1.0]
        assert bbox_iou(box1, box2) == 0.0


class TestSeverity:
    def test_grade_a_large_area(self):
        assert compute_severity([0.0, 0.0, 0.9, 0.9], 0.8) == "A"

    def test_grade_a_high_conf(self):
        assert compute_severity([0.0, 0.0, 0.2, 0.2], 0.95) == "A"

    def test_grade_d_small(self):
        # 极小面积 + 极低置信度 = D
        assert compute_severity([0.0, 0.0, 0.02, 0.02], 0.15) == "D"

    def test_grade_b_medium(self):
        assert compute_severity([0.0, 0.0, 0.5, 0.3], 0.75) == "B"

    def test_grade_c_low_med(self):
        # 小面积 + 中等置信度 = C
        assert compute_severity([0.0, 0.0, 0.1, 0.1], 0.45) == "C"


class TestOverallStatus:
    def test_pass(self):
        assert compute_overall_status(10) == "Pass"

    def test_marginal(self):
        assert compute_overall_status(50) == "Marginal"

    def test_fail(self):
        assert compute_overall_status(80) == "Fail"

    def test_boundary_marginal(self):
        assert compute_overall_status(30) == "Pass"
        assert compute_overall_status(31) == "Marginal"

    def test_boundary_fail(self):
        assert compute_overall_status(75) == "Marginal"
        assert compute_overall_status(76) == "Fail"


class TestSeverityIndex:
    def test_no_defects(self):
        assert compute_severity_index([], 0.0) == 4

    def test_all_a_defects(self):
        dets = [{"severity": "A"}, {"severity": "A"}]
        assert compute_severity_index(dets, 10.0) > 80

    def test_all_d_defects(self):
        dets = [{"severity": "D"}]
        assert compute_severity_index(dets, 0.0) < 20


class TestDefectDensity:
    def test_no_defects(self):
        assert compute_defect_density([]) == 0.0

    def test_single_defect(self):
        dets = [{"bbox": [0, 0, 20, 20]}]  # 400/10000 * 100 = 4%
        assert compute_defect_density(dets) == pytest.approx(4.0)

    def test_capped_at_100(self):
        dets = [{"bbox": [0, 0, 100, 100]}] * 10
        assert compute_defect_density(dets) == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
