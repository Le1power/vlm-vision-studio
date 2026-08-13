"""生成 assets/examples 下的示例图像（纯代码合成，不使用网络资源）。

生成两类示例：
1. sample_shapes.png：随机几何图形（圆/矩形/三角形）+ 文字，模拟多目标场景；
2. sample_gradient.png：渐变背景 + 噪声 + 图形叠加，用于测试预处理与边缘提取。

用法::

    python scripts/make_sample_images.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "examples"


def _draw_shapes_bgr(rng: random.Random, width: int = 640, height: int = 480) -> np.ndarray:
    """绘制随机几何图形场景（BGR 数组）。"""
    image = np.full((height, width, 3), (245, 245, 240), dtype=np.uint8)
    palette = [
        (60, 76, 231),    # 红
        (46, 166, 86),    # 绿
        (238, 148, 40),   # 蓝
        (180, 100, 210),  # 紫
        (40, 200, 230),   # 黄
    ]
    for _ in range(8):
        color = palette[rng.randrange(len(palette))]
        shape = rng.choice(["circle", "rect", "triangle"])
        cx, cy = rng.randrange(60, width - 60), rng.randrange(60, height - 60)
        size = rng.randrange(25, 70)
        if shape == "circle":
            cv2.circle(image, (cx, cy), size, color, -1)
        elif shape == "rect":
            cv2.rectangle(image, (cx - size, cy - size), (cx + size, cy + size), color, -1)
        else:
            pts = np.array(
                [[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [pts], color)
    cv2.putText(
        image, "VLM Studio", (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2, cv2.LINE_AA,
    )
    return image


def _make_gradient_bgr(width: int = 640, height: int = 480) -> np.ndarray:
    """生成渐变 + 噪声 + 正弦条纹的测试图（BGR 数组）。"""
    xs = np.linspace(0, 1, width, dtype=np.float32)
    ys = np.linspace(0, 1, height, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    b = (gx * 200 + 30).astype(np.uint8)
    g = (gy * 180 + 40).astype(np.uint8)
    r = ((np.sin(gx * 4 * math.pi) * 0.5 + 0.5) * 220).astype(np.uint8)
    image = np.stack([b, g, r], axis=-1)
    noise = np.random.default_rng(42).integers(0, 25, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def main() -> None:
    """生成示例图像并打印输出路径。"""
    if not _HAS_CV2:
        raise RuntimeError("需要 OpenCV：pip install opencv-python-headless")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2024)

    shapes = _draw_shapes_bgr(rng)
    gradient = _make_gradient_bgr()

    paths = [
        (OUTPUT_DIR / "sample_shapes.png", shapes),
        (OUTPUT_DIR / "sample_gradient.png", gradient),
    ]
    for path, img in paths:
        cv2.imwrite(str(path), img)
        print(f"已生成: {path}")


if __name__ == "__main__":
    main()
