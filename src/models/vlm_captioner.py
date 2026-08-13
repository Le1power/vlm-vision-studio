"""VLM 图像描述模块：基于 HuggingFace BLIP（视觉语言大模型）。

模型：Salesforce/blip-image-captioning-base

设计要点：
- 懒加载：顶层不导入 transformers/torch，首次调用时加载；
- 优雅降级：transformers 缺失或权重下载失败时，若开启
  ``fallback_enabled``，则基于检测/分类结果生成模板化描述，
  保证端到端流程与评估、报告环节可完整运行；
- 降级描述会显式标记 ``fallback=True``，便于报告中区分真实 VLM 输出。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, List, Optional, Sequence

import numpy as np

from .detector import Detection
from .classifier import Classification


@dataclass
class CaptionResult:
    """图像描述结果。

    Attributes:
        caption: 描述文本。
        fallback: 是否为降级生成（非真实 VLM 输出）。
        reason: 降级原因（正常生成时为 None）。
    """

    caption: str
    fallback: bool = False
    reason: Optional[str] = None


class VLMCaptioner:
    """BLIP 图像描述生成器（懒加载 + 降级）。

    Args:
        model_name: HuggingFace 模型标识。
        max_new_tokens: 生成的最大 token 数。
        fallback_enabled: VLM 不可用时是否启用模板降级。
        device: 推理设备，默认自动选择。
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        max_new_tokens: int = 30,
        fallback_enabled: bool = True,
        local_files_only: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.fallback_enabled = fallback_enabled
        self.local_files_only = local_files_only
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._unavailable_reason: Optional[str] = None

    def is_available(self) -> bool:
        """VLM 是否可用（transformers/torch 可导入且权重可加载）。"""
        return self._ensure_model()

    @property
    def unavailable_reason(self) -> Optional[str]:
        """不可用原因（可用时为 None）。"""
        if self._model is not None:
            return None
        self._ensure_model()
        return self._unavailable_reason

    def _ensure_model(self) -> bool:
        """确保 BLIP 处理器与模型已加载。"""
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        try:
            import torch  # noqa: F401
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except Exception as exc:
            self._unavailable_reason = f"transformers/torch 不可用: {exc}"
            return False
        try:
            # 某些网络无法连接 Hugging Face Xet/CAS；普通 HTTP 下载更稳定。
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            self._processor = BlipProcessor.from_pretrained(
                self.model_name,
                use_fast=True,
                local_files_only=self.local_files_only,
            )
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                use_safetensors=False,
                local_files_only=self.local_files_only,
            )
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device).eval()
        except Exception as exc:
            self._unavailable_reason = f"BLIP 权重加载失败: {exc}"
            self._processor = None
            self._model = None
            return False
        return True

    # ------------------------------------------------------------------
    # 描述生成
    # ------------------------------------------------------------------
    def caption(
        self,
        image: np.ndarray,
        detections: Optional[Sequence[Detection]] = None,
        classifications: Optional[Sequence[Classification]] = None,
    ) -> CaptionResult:
        """生成图像描述。

        Args:
            image: HxWx3 图像数组。
            detections: 可选的目标检测结果，仅用于降级路径。
            classifications: 可选的分类结果，仅用于降级路径。

        Returns:
            CaptionResult；VLM 可用时为模型生成，否则为模板降级。
        """
        if self._ensure_model():
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("图像描述输入必须是 HxWx3 RGB 图像")
            import torch
            from PIL import Image

            pil_img = Image.fromarray(image)  # 内部数组已是 RGB，无需反转
            inputs = self._processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            text = self._processor.decode(output[0], skip_special_tokens=True).strip()
            return CaptionResult(caption=text, fallback=False)

        if not self.fallback_enabled:
            return CaptionResult(caption="", fallback=True, reason=self._unavailable_reason)
        return CaptionResult(
            caption=self._fallback_caption(detections or [], classifications or []),
            fallback=True,
            reason=self._unavailable_reason,
        )

    # ------------------------------------------------------------------
    # 降级路径（不依赖任何深度学习框架）
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_caption(
        detections: Sequence[Detection],
        classifications: Sequence[Classification],
    ) -> str:
        """基于检测/分类结果的模板化描述（VLM 不可用时的降级方案）。

        Args:
            detections: 目标检测结果序列。
            classifications: 图像分类结果序列。

        Returns:
            英文模板描述；无任何视觉信息时返回通用描述。
        """
        objects = [d.label for d in detections]
        if objects:
            unique = list(dict.fromkeys(objects))  # 保序去重
            if len(unique) == 1:
                scene = f"a photo containing {unique[0]}"
            else:
                scene = f"a photo containing {', '.join(unique[:-1])} and {unique[-1]}"
            return scene
        if classifications:
            top = classifications[0]
            return f"a photo of {top.label.replace('_', ' ')}"
        return "an image with visual content (VLM unavailable, no detection context)"
