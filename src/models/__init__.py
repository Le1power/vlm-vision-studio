"""模型子包：目标检测 / 图像分类 / VLM 图像描述。

所有重型依赖（torch / torchvision / transformers）均在方法内部懒加载，
未安装对应依赖时实例化不报错，调用推理时给出明确提示或优雅降级。
"""

from .detector import ObjectDetector
from .classifier import ImageClassifier
from .vlm_captioner import VLMCaptioner

__all__ = ["ObjectDetector", "ImageClassifier", "VLMCaptioner"]
