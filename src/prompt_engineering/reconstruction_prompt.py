"""生成面向原图复刻的结构化 Prompt。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..models.detector import Detection
from .visual_context import dominant_colors


@dataclass
class VisualElement:
    shape: str
    color: str
    position: str
    size: str
    bounds_percent: Tuple[int, int, int, int]


def _position(cx: float, cy: float, width: int, height: int) -> str:
    horizontal = "left" if cx < width / 3 else "right" if cx > width * 2 / 3 else "center"
    vertical = "top" if cy < height / 3 else "bottom" if cy > height * 2 / 3 else "middle"
    return "center" if horizontal == "center" and vertical == "middle" else f"{vertical} {horizontal}"


def _shape_name(contour: np.ndarray) -> str:
    import cv2

    perimeter = cv2.arcLength(contour, True)
    vertices = len(cv2.approxPolyDP(contour, 0.025 * perimeter, True))
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        x, y, width, height = cv2.boundingRect(contour)
        return "square" if 0.85 <= width / max(height, 1) <= 1.15 else "rectangle"
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
    return "circle" if circularity >= 0.72 else "organic shape"


def _element_color(image: np.ndarray, contour: np.ndarray) -> str:
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    pixels = image[mask > 0]
    return dominant_colors(pixels.reshape(-1, 1, 3), 1)[0] if pixels.size else "unknown"


def extract_visual_elements(image: np.ndarray, max_elements: int = 12) -> List[VisualElement]:
    """用轮廓提取显著色块及其几何、位置和相对尺寸。"""
    try:
        import cv2
    except ImportError:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = max(1, height * width)
    elements: List[VisualElement] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area / image_area < 0.002 or area / image_area > 0.92:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = area / image_area
        size = "large" if ratio >= 0.18 else "small" if ratio < 0.035 else "medium"
        bounds = (
            round(x / width * 100),
            round(y / height * 100),
            round((x + w) / width * 100),
            round((y + h) / height * 100),
        )
        elements.append(
            VisualElement(
                shape=_shape_name(contour),
                color=_element_color(image, contour),
                position=_position(x + w / 2, y + h / 2, width, height),
                size=size,
                bounds_percent=bounds,
            )
        )
        if len(elements) >= max_elements:
            break
    return elements


def build_reconstruction_prompt(
    image: np.ndarray,
    caption: str,
    detections: Sequence[Detection],
) -> Dict[str, object]:
    """构建不主动添加摄影风格、强调几何和布局的复刻说明。"""
    height, width = image.shape[:2]
    colors = dominant_colors(image, 5)
    elements = extract_visual_elements(image)
    luminance = image.astype(np.float32).mean(axis=2)
    contrast = float(luminance.std())
    saturation = float((image.max(axis=2).astype(float) - image.min(axis=2)).mean())
    quantized = (image // 32).reshape(-1, 3)
    color_bins = len(np.unique(quantized, axis=0))
    try:
        import cv2

        edge_ratio = float(np.count_nonzero(cv2.Canny(image, 80, 160)) / max(1, width * height))
    except ImportError:
        edge_ratio = 0.0
    visual_medium = (
        "minimal flat vector graphic" if color_bins < 45 and edge_ratio < 0.10
        else "clean digital illustration" if color_bins < 120
        else "natural photographic image"
    )
    style_tags = [
        visual_medium,
        "low contrast" if contrast < 45 else "medium contrast" if contrast < 85 else "high contrast",
        "muted colors" if saturation < 35 else "moderately saturated colors" if saturation < 85 else "highly saturated colors",
        "even diffuse lighting" if contrast < 60 else "directional lighting",
        "clean sharp edges" if edge_ratio < 0.12 else "fine detailed edges",
    ]
    background = colors[0] if colors else "neutral"
    orientation = "landscape" if width > height else "portrait" if height > width else "square"
    element_phrases = [
        f"a {item.size} {item.color} {item.shape} at the {item.position} "
        f"occupying bounds {item.bounds_percent[0]}%-{item.bounds_percent[2]}% horizontally "
        f"and {item.bounds_percent[1]}%-{item.bounds_percent[3]}% vertically"
        for item in elements
    ]
    detected = [f"{item.label} at confidence {item.score:.2f}" for item in detections]
    detailed_parts = [
        "faithful reconstruction of the reference image",
        f"{orientation} composition with aspect ratio {width}:{height}",
        f"predominantly {background} background",
        f"dominant palette: {', '.join(colors)}" if colors else "preserve the original color palette",
        "preserve exact object count, scale, spacing, alignment, silhouettes and negative space",
        "preserve the original rendering style: " + ", ".join(style_tags),
    ]
    if element_phrases:
        detailed_parts.append("layout elements: " + "; ".join(element_phrases))
    elif detected:
        detailed_parts.append("detected subjects: " + ", ".join(detected))
    if caption:
        detailed_parts.append(f"semantic reference only: {caption}")
    detailed_parts.append("structurally accurate rendering, no invented objects, no composition changes")

    # SD 1.5 的 CLIP 上限为 77 token。生成 Prompt 只保留最高价值信息，
    # 百分比坐标等完整细节留在 detailed_prompt 中供用户查看和审计。
    compact_elements = [
        f"{item.color} {item.shape} {item.position}"
        for item in elements[:3]
    ]
    compact_parts = [
        "faithful reconstruction",
        visual_medium,
        f"{orientation} composition",
        f"{background} background",
        f"{', '.join(colors[:4])} palette" if colors else "original palette",
        ", ".join(style_tags[1:4]),
    ]
    if compact_elements:
        compact_parts.append("layout: " + "; ".join(compact_elements))
    compact_parts.append("exact layout, silhouettes, spacing and negative space, no extra objects")
    negative = (
        "different composition, changed layout, extra objects, missing objects, altered colors, "
        "cropped elements, perspective shift, text changes, dramatic lighting, shallow depth of field, "
        "photorealistic texture, blur, watermark"
    )
    return {
        "prompt": ", ".join(compact_parts),
        "detailed_prompt": ", ".join(detailed_parts),
        "negative_prompt": negative,
        "elements": [asdict(item) for item in elements],
        "dominant_colors": colors,
        "background_color": background,
        "orientation": orientation,
        "style_tags": style_tags,
        "style_metrics": {
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "edge_ratio": round(edge_ratio, 4),
            "color_bins": color_bins,
        },
    }
