"""从图像与检测框提取适合 Prompt 的颜色、尺寸和空间关系。"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from ..models.detector import Detection

_COLOR_PALETTE = {
    "black": np.array([20, 20, 20]),
    "white": np.array([235, 235, 235]),
    "gray": np.array([128, 128, 128]),
    "red": np.array([210, 55, 55]),
    "orange": np.array([230, 135, 45]),
    "yellow": np.array([225, 205, 55]),
    "green": np.array([65, 155, 75]),
    "blue": np.array([65, 105, 205]),
    "purple": np.array([145, 75, 175]),
    "pink": np.array([225, 135, 170]),
    "brown": np.array([125, 85, 55]),
}


def _nearest_color(rgb: np.ndarray) -> str:
    return min(_COLOR_PALETTE, key=lambda name: float(np.linalg.norm(rgb - _COLOR_PALETTE[name])))


def dominant_colors(image: np.ndarray, count: int = 3) -> List[str]:
    """用量化后的像素采样估算图像主色，避免引入额外聚类依赖。"""
    if image.ndim != 3 or image.shape[2] != 3:
        return []
    sampled = image[:: max(1, image.shape[0] // 80), :: max(1, image.shape[1] // 80)]
    names = (_nearest_color(pixel.astype(float)) for pixel in sampled.reshape(-1, 3))
    return [name for name, _ in Counter(names).most_common(max(1, count))]


def _position(cx: float, cy: float, width: float, height: float) -> str:
    horizontal = "left" if cx < width / 3 else "right" if cx > width * 2 / 3 else "center"
    vertical = "upper" if cy < height / 3 else "lower" if cy > height * 2 / 3 else "middle"
    return f"{vertical} {horizontal}" if horizontal != "center" or vertical != "middle" else "center"


def describe_visual_context(
    image: np.ndarray,
    detections: Sequence[Detection],
) -> Dict[str, object]:
    """返回主色、逐目标布局短语和简单的目标间空间关系。"""
    height, width = image.shape[:2]
    image_area = max(1.0, float(height * width))
    objects: List[str] = []
    centers: List[tuple[str, float, float]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        size = "large" if ratio >= 0.25 else "small" if ratio < 0.06 else "medium-sized"
        crop = image[
            max(0, int(y1)) : min(height, int(y2)),
            max(0, int(x1)) : min(width, int(x2)),
        ]
        color = dominant_colors(crop, 1)
        color_text = f"{color[0]} " if color else ""
        objects.append(f"{size} {color_text}{detection.label} in the {_position(cx, cy, width, height)}")
        centers.append((detection.label, cx, cy))

    relations: List[str] = []
    for index, (label_a, ax, ay) in enumerate(centers[:5]):
        for label_b, bx, by in centers[index + 1 : 5]:
            if abs(ax - bx) >= abs(ay - by):
                relation = "left of" if ax < bx else "right of"
            else:
                relation = "above" if ay < by else "below"
            relations.append(f"{label_a} {relation} {label_b}")

    colors = dominant_colors(image)
    tags = [f"dominant colors: {', '.join(colors)}"] if colors else []
    tags.extend(objects)
    tags.extend(relations[:4])
    return {"dominant_colors": colors, "objects": objects, "relations": relations, "prompt_tags": tags}
