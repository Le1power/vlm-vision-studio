"""图像分类模块：基于 torchvision ResNet50 预训练权重（ImageNet-1K）。

与 detector 相同的设计原则：顶层不导入 torch，首次推理时懒加载，
依赖缺失时优雅降级（返回空预测并说明原因）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class Classification:
    """单条分类预测结果。

    Attributes:
        label: ImageNet 类别名称。
        score: softmax 概率，0~1。
    """

    label: str
    score: float


class ImageClassifier:
    """ResNet50 图像分类器（懒加载）。

    Args:
        top_k: 返回概率最高的 K 个类别。
        device: 推理设备，默认自动选择。
    """

    def __init__(self, top_k: int = 5, device: Optional[str] = None) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        self.top_k = top_k
        self.device = device
        self._model: Any = None
        self._categories: List[str] = []
        self._preprocess: Any = None
        self._unavailable_reason: Optional[str] = None

    def is_available(self) -> bool:
        """分类器是否可用。"""
        return self._ensure_model()

    @property
    def unavailable_reason(self) -> Optional[str]:
        """不可用原因（可用时为 None）。"""
        if self._model is not None:
            return None
        self._ensure_model()
        return self._unavailable_reason

    def _ensure_model(self) -> bool:
        """确保模型与预处理管线已加载。"""
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        try:
            import torch  # noqa: F401
            from torchvision.models import ResNet50_Weights, resnet50
        except Exception as exc:
            self._unavailable_reason = f"torch/torchvision 不可用: {exc}"
            return False
        try:
            weights = ResNet50_Weights.DEFAULT
            self._model = resnet50(weights=weights)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device).eval()
            self._categories = list(weights.meta["categories"])
            self._preprocess = weights.transforms()
        except Exception as exc:
            self._unavailable_reason = f"模型权重加载失败: {exc}"
            self._model = None
            return False
        return True

    def classify(self, image: np.ndarray) -> List[Classification]:
        """对单张图像执行 Top-K 分类。

        Args:
            image: HxWx3 RGB 图像数组（由 ImageProcessor.load 保证通道序）。

        Returns:
            Top-K 预测列表；分类器不可用时返回空列表。
        """
        if not self._ensure_model():
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("图像分类输入必须是 HxWx3 RGB 图像")
        import torch
        from PIL import Image

        pil_img = Image.fromarray(image)  # 内部数组已是 RGB，无需反转
        batch = self._preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self._model(batch), dim=1)[0]
        topk: Tuple[Any, Any] = probs.topk(min(self.top_k, len(self._categories)))
        scores, indices = topk
        return [
            Classification(label=self._categories[int(i)], score=round(float(s), 4))
            for s, i in zip(scores.tolist(), indices.tolist())
        ]
