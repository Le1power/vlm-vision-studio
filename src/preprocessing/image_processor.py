"""OpenCV 图像预处理与经典特征提取模块。

职责：
- 图像读取、缩放、去噪、灰度化
- Canny 边缘特征提取
- 颜色直方图特征计算
- 可视化结果保存

设计说明：
- OpenCV（cv2）在模块导入时尝试加载；若未安装，则以 PIL/numpy 路径
  提供读取与缩放等基础能力的降级实现，保证轻量功能可用。
- 本模块不依赖任何深度学习框架。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:  # OpenCV 为可选依赖：无 GUI 环境可安装 opencv-python-headless
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - 取决于运行环境
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

# 图像数组的统一类型约定：H x W x C，**始终为 RGB**。
# cv2 读入的 BGR 会在 load() 出口统一转为 RGB，下游模块无需关心通道序；
# save() 在调用 cv2.imwrite 前内部转回 BGR，对外不暴露该差异。
ImageArray = np.ndarray
PathLike = Union[str, Path]


class ImageProcessor:
    """图像预处理器：封装 OpenCV 常用操作，支持特征提取与可视化。

    Attributes:
        has_cv2: 当前环境是否可用 OpenCV。
    """

    def __init__(self) -> None:
        self.has_cv2: bool = _HAS_CV2

    # ------------------------------------------------------------------
    # 读取与保存
    # ------------------------------------------------------------------
    def load(self, path: PathLike) -> ImageArray:
        """读取图像为 numpy 数组。

        Args:
            path: 图像文件路径。

        Returns:
            RGB 格式的 HxWx3 数组（无论底层使用 OpenCV 还是 PIL）。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件存在但无法解码为图像。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"图像文件不存在: {path}")
        if self.has_cv2:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"无法解码图像: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 统一输出 RGB
        from PIL import Image  # 降级路径：PIL 读取

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))

    def save(self, image: ImageArray, path: PathLike) -> Path:
        """保存图像数组到磁盘，自动创建父目录。

        Args:
            image: 图像数组。
            path: 目标路径。

        Returns:
            实际写入的路径。

        Raises:
            OSError: 写入失败（如不支持的扩展名或磁盘错误）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.has_cv2:
            # 内部数组约定为 RGB，cv2.imwrite 需要 BGR（灰度图无需转换）
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
            ok = cv2.imwrite(str(path), bgr)
            if not ok:
                raise OSError(f"图像写入失败: {path}")
        else:
            from PIL import Image

            Image.fromarray(image).save(path)
        return path

    # ------------------------------------------------------------------
    # 基础预处理
    # ------------------------------------------------------------------
    def resize(self, image: ImageArray, max_side: int = 512) -> ImageArray:
        """等比例缩放图像，使最长边不超过 ``max_side``。

        Args:
            image: 输入图像。
            max_side: 最长边像素上限；<=0 时返回原图。

        Returns:
            缩放后的图像（若原图已满足约束则返回副本）。
        """
        if max_side <= 0:
            return image.copy()
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("无法缩放空图像")
        scale = max_side / max(h, w)
        if scale >= 1.0:
            return image.copy()
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        if self.has_cv2:
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        from PIL import Image

        pil_img = Image.fromarray(image)
        return np.asarray(pil_img.resize(new_size, Image.BILINEAR))

    def to_gray(self, image: ImageArray) -> ImageArray:
        """转换为灰度图。

        Args:
            image: HxWx3 彩色图像。

        Returns:
            HxW 单通道灰度数组。
        """
        if image.ndim == 2:
            return image
        if self.has_cv2:
            code = cv2.COLOR_RGB2GRAY  # 内部数组统一为 RGB
            return cv2.cvtColor(image, code)
        # 降级路径：标准亮度加权公式
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    def denoise(self, image: ImageArray, strength: int = 5) -> ImageArray:
        """图像去噪。

        OpenCV 可用时使用双边滤波（保边去噪），否则退化为 3x3 均值滤波。

        Args:
            image: 输入图像。
            strength: 滤波强度（双边滤波的 sigma 值）。

        Returns:
            去噪后的图像。
        """
        if self.has_cv2:
            return cv2.bilateralFilter(image, d=5, sigmaColor=strength * 10, sigmaSpace=strength * 10)
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        return self._convolve3x3(image, kernel)

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------
    def extract_edges(self, image: ImageArray, low: int = 80, high: int = 160) -> ImageArray:
        """提取 Canny 边缘特征。

        Args:
            image: 输入图像（彩色或灰度）。
            low: Canny 低阈值。
            high: Canny 高阈值。

        Returns:
            二值边缘图（0/255）。
        """
        gray = self.to_gray(image)
        if self.has_cv2:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            return cv2.Canny(blurred, low, high)
        # 降级路径：Sobel 梯度幅值 + 阈值
        gx = self._convolve3x3(gray.astype(np.float32), np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32))
        gy = self._convolve3x3(gray.astype(np.float32), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32))
        mag = np.hypot(gx, gy)
        return ((mag > low) * 255).astype(np.uint8)

    def color_histogram(self, image: ImageArray, bins: int = 8) -> Dict[str, List[float]]:
        """计算归一化颜色直方图特征。

        Args:
            image: 输入彩色图像。
            bins: 每通道直方图 bin 数。

        Returns:
            形如 ``{"channel_0": [...], ...}`` 的归一化直方图（每通道和为 1）。
            通道序固定为 RGB（channel_0=R），与环境是否安装 OpenCV 无关。
        """
        if bins <= 0:
            raise ValueError("bins 必须大于 0")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        features: Dict[str, List[float]] = {}
        for ch in range(image.shape[2]):
            hist, _ = np.histogram(image[..., ch], bins=bins, range=(0, 256))
            total = hist.sum()
            features[f"channel_{ch}"] = (hist / total).tolist() if total else [0.0] * bins
        return features

    def edge_density(self, edges: ImageArray) -> float:
        """计算边缘密度（边缘像素占比），可作为图像复杂度特征。

        Args:
            edges: 二值边缘图。

        Returns:
            0~1 之间的浮点数。
        """
        if edges.size == 0:
            return 0.0
        return float(np.count_nonzero(edges) / edges.size)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def make_feature_panel(
        self,
        image: ImageArray,
        max_side: int = 512,
    ) -> Tuple[ImageArray, Dict[str, float]]:
        """生成「原图 | 灰度 | 边缘」三联可视化面板及特征摘要。

        Args:
            image: 输入图像。
            max_side: 面板中每张子图的最长边。

        Returns:
            (面板图像, 特征摘要字典)，摘要含边缘密度与尺寸信息。
        """
        resized = self.resize(image, max_side)
        gray = self.to_gray(resized)
        edges = self.extract_edges(resized)
        gray_bgr = np.stack([gray] * 3, axis=-1)
        edges_bgr = np.stack([edges] * 3, axis=-1)
        panel = np.concatenate([resized, gray_bgr, edges_bgr], axis=1)
        summary = {
            "height": float(image.shape[0]),
            "width": float(image.shape[1]),
            "edge_density": self.edge_density(edges),
        }
        return panel, summary

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _convolve3x3(image: ImageArray, kernel: np.ndarray) -> ImageArray:
        """不依赖 OpenCV 的 3x3 卷积（边缘零填充）。

        Args:
            image: 单通道或多通道图像。
            kernel: 3x3 卷积核。

        Returns:
            卷积结果，dtype 与卷积中间精度一致后裁剪回 uint8 范围。
        """
        src = image.astype(np.float32)
        if src.ndim == 2:
            src = src[..., None]
        pad = np.pad(src, ((1, 1), (1, 1), (0, 0)))
        out = np.zeros_like(src)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * pad[i : i + src.shape[0], j : j + src.shape[1]]
        out = np.clip(out, 0, 255)
        result = out.astype(np.uint8)
        if image.ndim == 2:
            return result[..., 0]
        return result
