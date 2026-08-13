"""评估子包：检测统计、描述质量指标与批量评估流程。"""

from .metrics import (
    caption_metrics,
    detection_metrics,
    clip_score_placeholder,
)
from .evaluator import Evaluator

__all__ = [
    "caption_metrics",
    "detection_metrics",
    "clip_score_placeholder",
    "Evaluator",
]
