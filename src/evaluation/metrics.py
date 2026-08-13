"""评估指标模块：检测统计、描述质量与可选 CLIPScore 占位。

所有指标均为纯 Python / numpy 实现，不依赖深度学习框架，可独立测试。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..models.detector import Detection

_WORD_RE = re.compile(r"[a-z]+")


def detection_metrics(
    detections: Sequence[Detection],
    image_area: Optional[float] = None,
) -> Dict[str, object]:
    """统计目标检测结果。

    Args:
        detections: 单张图像的检测结果列表。
        image_area: 可选的图像面积（像素²），用于计算目标框面积占比。

    Returns:
        指标字典：
        - num_detections: 检测框数量
        - mean_score: 平均置信度（无检测时为 0）
        - min_score / max_score: 置信度极值
        - category_counts: 各类别出现次数
        - mean_box_area_ratio: 平均框面积占比（未提供 image_area 时为 None）
    """
    n = len(detections)
    scores = [d.score for d in detections]
    categories: Counter = Counter(d.label for d in detections)
    metrics: Dict[str, object] = {
        "num_detections": n,
        "mean_score": round(sum(scores) / n, 4) if n else 0.0,
        "min_score": round(min(scores), 4) if n else 0.0,
        "max_score": round(max(scores), 4) if n else 0.0,
        "category_counts": dict(categories),
        "mean_box_area_ratio": None,
    }
    if image_area and image_area > 0 and n:
        ratios = []
        for d in detections:
            x1, y1, x2, y2 = d.box
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            ratios.append(area / image_area)
        metrics["mean_box_area_ratio"] = round(sum(ratios) / len(ratios), 4)
    return metrics


def caption_metrics(
    caption: str,
    min_words: int = 5,
    min_ttr: float = 0.4,
) -> Dict[str, object]:
    """评估图像描述文本质量。

    注意：分词正则仅匹配英文单词（``[a-z]+``），中文等多字节语言的描述
    会得到 0 词数。当前 pipeline 的 caption 来源（BLIP / 英文模板）均为英文，
    若未来接入中文描述需先扩展分词逻辑。

    Args:
        caption: 描述文本。
        min_words: 合格描述的最少词数。
        min_ttr: 合格描述的词汇丰富度（TTR）下限。

    Returns:
        指标字典：
        - num_words: 词数
        - num_unique_words: 唯一词数
        - ttr: Type-Token Ratio（唯一词/总词数）
        - num_chars: 字符数
        - passed_length / passed_ttr: 是否达到质量阈值
        - quality_pass: 综合是否合格
    """
    words: List[str] = _WORD_RE.findall(caption.lower())
    n = len(words)
    unique = len(set(words))
    ttr = unique / n if n else 0.0
    passed_length = n >= min_words
    passed_ttr = ttr >= min_ttr if n else False
    return {
        "num_words": n,
        "num_unique_words": unique,
        "ttr": round(ttr, 4),
        "num_chars": len(caption),
        "passed_length": passed_length,
        "passed_ttr": passed_ttr,
        "quality_pass": passed_length and passed_ttr,
    }


def clip_score_placeholder(
    caption: str,
    image_path: Optional[str] = None,
) -> Optional[float]:
    """CLIPScore 图文一致性指标占位。

    说明：
        完整 CLIPScore 需加载 CLIP 模型（openai/clip 或 open_clip），
        为避免额外重依赖，本函数为占位实现，始终返回 None。
        后续可在此接入 ``open_clip`` 计算真实分数。

    Args:
        caption: 描述文本。
        image_path: 图像路径。

    Returns:
        当前始终为 None（未实现）。
    """
    _ = (caption, image_path)
    return None
