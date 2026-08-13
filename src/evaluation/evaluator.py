"""批量评估流程模块：对图像目录执行端到端评估并汇总指标。

流程：遍历图像 -> pipeline 处理 -> 汇总检测/描述/Prompt 指标。
模型不可用时自动走降级路径，评估与报告流程不中断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

from .metrics import caption_metrics, detection_metrics

PathLike = Union[str, Path]
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SampleResult:
    """单张图像的评估结果。

    Attributes:
        image_path: 图像路径。
        pipeline_output: pipeline 返回的完整输出字典。
        detection_stats: 检测统计指标。
        caption_stats: 描述质量指标。
        error: 处理失败原因（成功时为 None）；失败样本不计入聚合均值。
    """

    image_path: str
    pipeline_output: Dict[str, object]
    detection_stats: Dict[str, object]
    caption_stats: Dict[str, object]
    error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """批量评估汇总。

    Attributes:
        num_images: 评估图像总数。
        samples: 每张图像的评估结果。
        aggregate: 跨图像聚合指标。
    """

    num_images: int
    samples: List[SampleResult] = field(default_factory=list)
    aggregate: Dict[str, object] = field(default_factory=dict)


class PipelineProtocol(Protocol):
    """Evaluator 所需的最小流水线接口。"""

    def run(self, image_path: PathLike, save_panel: bool = True) -> Dict[str, object]: ...


class Evaluator:
    """批量评估器。

    Args:
        pipeline: 已构建的端到端 Pipeline 实例。
        min_caption_words: 描述最少词数阈值。
        min_ttr: 描述 TTR 阈值。
    """

    def __init__(
        self,
        pipeline: PipelineProtocol,
        min_caption_words: int = 5,
        min_ttr: float = 0.4,
    ) -> None:
        self.pipeline = pipeline
        self.min_caption_words = min_caption_words
        self.min_ttr = min_ttr

    def evaluate_directory(self, image_dir: PathLike) -> EvaluationSummary:
        """评估目录下所有支持的图像。

        Args:
            image_dir: 图像目录。

        Returns:
            EvaluationSummary，含逐样本结果与聚合指标。

        Raises:
            FileNotFoundError: 目录不存在。
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        summary = EvaluationSummary(num_images=len(paths))
        for path in paths:
            try:
                output = self.pipeline.run(path)
            except Exception as exc:
                # 单张坏图（无法解码、读取失败等）不中断整体评估
                summary.samples.append(
                    SampleResult(
                        image_path=str(path),
                        pipeline_output={},
                        detection_stats={},
                        caption_stats={},
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            det_stats = detection_metrics(output.get("detections", []))
            cap_stats = caption_metrics(
                output.get("caption", ""),
                min_words=self.min_caption_words,
                min_ttr=self.min_ttr,
            )
            summary.samples.append(
                SampleResult(
                    image_path=str(path),
                    pipeline_output=output,
                    detection_stats=det_stats,
                    caption_stats=cap_stats,
                )
            )
        summary.aggregate = self._aggregate(summary.samples)
        return summary

    @staticmethod
    def _aggregate(samples: List[SampleResult]) -> Dict[str, object]:
        """聚合跨样本指标。

        失败样本不计入均值；mean_score 只在「确有检出」的样本上平均，
        避免无检测图像的 0 分拉低整体置信度指标。
        """
        ok = [s for s in samples if s.error is None]
        num_failed = len(samples) - len(ok)
        if not ok:
            return {
                "mean_detections": 0.0,
                "mean_score": 0.0,
                "mean_caption_words": 0.0,
                "mean_ttr": 0.0,
                "caption_quality_pass_rate": 0.0,
                "vlm_fallback_rate": 0.0,
                "num_failed": num_failed,
            }
        n = len(ok)
        det_counts = [int(s.detection_stats["num_detections"]) for s in ok]
        detected = [s for s in ok if int(s.detection_stats["num_detections"]) > 0]
        words = [int(s.caption_stats["num_words"]) for s in ok]
        ttrs = [float(s.caption_stats["ttr"]) for s in ok]
        passes = [bool(s.caption_stats["quality_pass"]) for s in ok]
        fallbacks = [
            bool(s.pipeline_output.get("caption_fallback", False)) for s in ok
        ]
        return {
            "mean_detections": round(sum(det_counts) / n, 2),
            "mean_score": round(
                sum(float(s.detection_stats["mean_score"]) for s in detected) / len(detected), 4
            )
            if detected
            else 0.0,
            "mean_caption_words": round(sum(words) / n, 2),
            "mean_ttr": round(sum(ttrs) / n, 2),
            "caption_quality_pass_rate": round(sum(passes) / n, 4),
            "vlm_fallback_rate": round(sum(fallbacks) / n, 4),
            "num_failed": num_failed,
        }
