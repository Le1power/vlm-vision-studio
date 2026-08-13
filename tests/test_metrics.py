"""评估指标单元测试：纯 Python 逻辑，不涉及模型下载。"""

import pytest

from src.evaluation.metrics import (
    caption_metrics,
    clip_score_placeholder,
    detection_metrics,
)
from src.models.detector import Detection


def _det(label: str, score: float, box=None) -> Detection:
    return Detection(label=label, score=score, box=box or [0.0, 0.0, 10.0, 10.0])


class TestDetectionMetrics:
    def test_empty(self) -> None:
        m = detection_metrics([])
        assert m["num_detections"] == 0
        assert m["mean_score"] == 0.0
        assert m["category_counts"] == {}
        assert m["mean_box_area_ratio"] is None

    def test_stats(self) -> None:
        dets = [_det("dog", 0.9), _det("dog", 0.7), _det("cat", 0.8)]
        m = detection_metrics(dets)
        assert m["num_detections"] == 3
        assert m["mean_score"] == pytest.approx(0.8, abs=1e-4)
        assert m["min_score"] == pytest.approx(0.7, abs=1e-4)
        assert m["max_score"] == pytest.approx(0.9, abs=1e-4)
        assert m["category_counts"] == {"dog": 2, "cat": 1}

    def test_box_area_ratio(self) -> None:
        # 图像 200x200=40000，框 10x10=100 -> 占比 0.0025
        m = detection_metrics([_det("dog", 0.9)], image_area=40000.0)
        assert m["mean_box_area_ratio"] == pytest.approx(0.0025, abs=1e-6)


class TestCaptionMetrics:
    def test_empty_caption(self) -> None:
        m = caption_metrics("")
        assert m["num_words"] == 0
        assert m["ttr"] == 0.0
        assert m["quality_pass"] is False

    def test_word_and_ttr(self) -> None:
        m = caption_metrics("a dog runs in the green park")
        assert m["num_words"] == 7
        assert m["num_unique_words"] == 7
        assert m["ttr"] == 1.0
        assert m["quality_pass"] is True

    def test_repeated_words_lower_ttr(self) -> None:
        m = caption_metrics("dog dog dog dog dog")
        assert m["ttr"] == pytest.approx(0.2)
        assert m["passed_ttr"] is False
        assert m["passed_length"] is True
        assert m["quality_pass"] is False

    def test_thresholds_configurable(self) -> None:
        m = caption_metrics("one two three", min_words=3, min_ttr=0.9)
        assert m["quality_pass"] is True
        m2 = caption_metrics("one two three", min_words=4)
        assert m2["quality_pass"] is False


class TestClipPlaceholder:
    def test_returns_none(self) -> None:
        assert clip_score_placeholder("a caption", "x.png") is None
