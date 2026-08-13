"""目标检测模块：基于 torchvision Faster R-CNN 预训练权重。

设计要点：
- 懒加载：实例化时不导入 torch / 不下载权重，首次推理时才加载模型；
- 依赖缺失或权重下载失败时 ``is_available()`` 返回 False，``detect()``
  返回空列表并给出原因，保证上层流程不中断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Detection:
    """单个检测结果。

    Attributes:
        label: COCO 类别名称。
        score: 置信度，0~1。
        box: 边界框 ``[x1, y1, x2, y2]``（像素坐标）。
    """

    label: str
    score: float
    box: List[float]


class ObjectDetector:
    """Faster R-CNN 目标检测器（懒加载）。

    Args:
        score_threshold: 置信度过滤阈值。
        max_detections: 单图最多返回的检测框数量。
        max_side: 推理前图像最长边上限，超出则等比缩小（结果坐标会映射回原图），
            防止超大输入导致内存/显存溢出；<=0 表示不限制。
        device: 推理设备，默认自动选择（CUDA 可用则用 CUDA）。
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        max_detections: int = 20,
        max_side: int = 1024,
        device: Optional[str] = None,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold 必须在 0 到 1 之间")
        if max_detections <= 0:
            raise ValueError("max_detections 必须大于 0")
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.max_side = max_side
        self.device = device
        self._model: Any = None
        self._categories: List[str] = []
        self._unavailable_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # 可用性检查与懒加载
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测器是否可用（torch/torchvision 可导入且权重可加载）。"""
        return self._ensure_model()

    @property
    def unavailable_reason(self) -> Optional[str]:
        """不可用原因（可用时为 None）。"""
        if self._model is not None:
            return None
        self._ensure_model()
        return self._unavailable_reason

    def _ensure_model(self) -> bool:
        """确保模型已加载；返回是否成功。"""
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        try:
            import torch  # noqa: F401  (函数内懒加载)
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
        except Exception as exc:  # 依赖未安装
            self._unavailable_reason = f"torch/torchvision 不可用: {exc}"
            return False
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device).eval()
            self._categories = list(weights.meta["categories"])
        except Exception as exc:  # 权重下载失败等
            self._unavailable_reason = f"模型权重加载失败: {exc}"
            self._model = None
            return False
        return True

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Detection]:
        """对单张图像执行目标检测。

        Args:
            image: HxWx3 RGB 图像数组（由 ImageProcessor.load 保证通道序）。
                通道序对预训练模型有显著影响，请勿直接传入 OpenCV 读出的 BGR 数组。

        Returns:
            检测结果列表（坐标为原图像素坐标）；检测器不可用时返回空列表。
        """
        if not self._ensure_model():
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("目标检测输入必须是 HxWx3 RGB 图像")
        import torch  # 懒加载

        resized, coord_scale = self._downscale(image)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self._model([tensor])[0]
        results: List[Detection] = []
        for box, label_idx, score in zip(output["boxes"], output["labels"], output["scores"]):
            score_val = float(score)
            if score_val < self.score_threshold:
                break
            results.append(
                Detection(
                    label=self._categories[int(label_idx)],
                    score=round(score_val, 4),
                    box=[round(float(v) * coord_scale, 1) for v in box.tolist()],
                )
            )
            if len(results) >= self.max_detections:
                break
        return results

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _downscale(self, image: np.ndarray) -> tuple:
        """等比缩小超限图像，返回 (图像, 坐标映射回原图的缩放系数)。"""
        h, w = image.shape[:2]
        longest = max(h, w)
        if self.max_side <= 0 or longest <= self.max_side or longest == 0:
            return image, 1.0
        from PIL import Image

        scale = self.max_side / longest
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = np.asarray(Image.fromarray(image).resize(new_size, Image.BILINEAR))
        return resized, 1.0 / scale
