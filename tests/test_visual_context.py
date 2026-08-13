"""颜色、目标尺寸和空间关系描述测试。"""

import numpy as np

from src.models.detector import Detection
from src.prompt_engineering.visual_context import describe_visual_context, dominant_colors


def test_dominant_colors_recognizes_simple_palette() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :30] = (240, 240, 240)
    image[:, 30:] = (220, 50, 50)

    colors = dominant_colors(image, count=2)

    assert colors[0] == "white"
    assert "red" in colors


def test_context_describes_size_position_and_relation() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    detections = [
        Detection("cat", 0.9, [0, 0, 20, 20]),
        Detection("dog", 0.8, [60, 60, 100, 100]),
    ]

    context = describe_visual_context(image, detections)

    assert any("small white cat in the upper left" in item for item in context["objects"])
    assert any("dog" in item and "lower right" in item for item in context["objects"])
    assert context["relations"] == ["cat left of dog"]
