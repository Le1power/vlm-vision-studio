"""端到端流程：图像 -> 预处理 -> 检测/分类 -> VLM 描述 -> Prompt 优化 -> 结果汇总。

用法示例::

    from src.pipeline import Pipeline
    pipe = Pipeline(config_path="configs/default.yaml")
    result = pipe.run("assets/examples/sample_shapes.png")

所有模型组件均为懒加载，未安装重依赖时自动降级，流程不中断。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from .preprocessing.image_processor import ImageProcessor
from .models.detector import ObjectDetector
from .models.classifier import ImageClassifier
from .models.vlm_captioner import VLMCaptioner
from .prompt_engineering.prompt_optimizer import PromptOptimizer
from .prompt_engineering.visual_context import describe_visual_context
from .prompt_engineering.reconstruction_prompt import build_reconstruction_prompt
from .generation.image_generator import ReferenceImageGenerator

PathLike = Union[str, Path]

# 配置缺失时的默认值
_DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "vlm-vision-studio", "version": "0.1.0"},
    "detection": {"score_threshold": 0.5, "max_detections": 20, "max_side": 1024},
    "classification": {"top_k": 5, "prompt_min_score": 0.2},
    "vlm": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "max_new_tokens": 30,
        "fallback_enabled": True,
        "local_files_only": False,
    },
    "prompt": {
        "language": "en",
        "default_style": "cinematic",
        "optimize_rounds": 2,
        "negative_prompt": True,
    },
    "evaluation": {"min_caption_words": 5, "min_ttr": 0.4},
    "generation": {
        "enabled": False,
        "model_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "mode": "img2img",
        "strength": 0.28,
        "guidance_scale": 7.0,
        "steps": 25,
        "seed": 42,
        "ip_adapter_repo": "h94/IP-Adapter",
        "ip_adapter_weight": "ip-adapter_sd15.bin",
        "ip_adapter_scale": 0.65,
        "controlnet_model": "lllyasviel/sd-controlnet-canny",
    },
    "model_storage": {"root_dir": "models"},
    "paths": {
        "output_dir": "outputs",
        "report_dir": "reports",
        "examples_dir": "assets/examples",
    },
}


def _merge_config(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge a mapping without mutating either input."""
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[PathLike]) -> Dict[str, Any]:
    """加载 YAML 配置，缺失字段以默认值补齐。

    Args:
        config_path: 配置文件路径；为 None 或文件不存在时返回默认配置。

    Returns:
        合并后的配置字典。
    """
    config = _merge_config(_DEFAULT_CONFIG, {})
    if config_path is None:
        return config
    path = Path(config_path)
    if not path.exists():
        return config
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 YAML 配置需要安装 PyYAML") from exc

    try:
        user_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取配置文件 {path}: {exc}") from exc
    if not isinstance(user_cfg, dict):
        raise ValueError(f"配置文件顶层必须是键值映射: {path}")
    return _merge_config(config, user_cfg)


class Pipeline:
    """多模态视觉理解与创作端到端流程。

    Args:
        config_path: YAML 配置路径；缺省使用内置默认配置。
        enable_classification: 是否启用分类分支（默认开启）。
    """

    def __init__(
        self,
        config_path: Optional[PathLike] = None,
        enable_classification: bool = True,
    ) -> None:
        self.config = load_config(config_path)
        self.processor = ImageProcessor()
        self.detector = ObjectDetector(
            score_threshold=float(self.config["detection"]["score_threshold"]),
            max_detections=int(self.config["detection"]["max_detections"]),
            max_side=int(self.config["detection"].get("max_side", 1024)),
        )
        self.classifier = (
            ImageClassifier(top_k=int(self.config["classification"]["top_k"]))
            if enable_classification
            else None
        )
        self.captioner = VLMCaptioner(
            model_name=str(self.config["vlm"]["model_name"]),
            max_new_tokens=int(self.config["vlm"]["max_new_tokens"]),
            fallback_enabled=bool(self.config["vlm"]["fallback_enabled"]),
            local_files_only=bool(self.config["vlm"].get("local_files_only", False)),
        )
        self.optimizer = PromptOptimizer(
            style=str(self.config["prompt"]["default_style"]),
            language=str(self.config["prompt"].get("language", "en")),
        )
        self.optimize_rounds = int(self.config["prompt"]["optimize_rounds"])
        self.negative_prompt_enabled = bool(
            self.config["prompt"].get("negative_prompt", True)
        )
        self.output_dir = Path(self.config["paths"]["output_dir"])
        generation = self.config["generation"]
        self.generation_enabled = bool(generation.get("enabled", False))
        self.generator = ReferenceImageGenerator(
            model_name=str(generation["model_name"]),
            mode=str(generation["mode"]),
            strength=float(generation["strength"]),
            guidance_scale=float(generation["guidance_scale"]),
            steps=int(generation["steps"]),
            ip_adapter_repo=str(generation["ip_adapter_repo"]),
            ip_adapter_weight=str(generation["ip_adapter_weight"]),
            ip_adapter_scale=float(generation["ip_adapter_scale"]),
            controlnet_model=str(generation["controlnet_model"]),
        )
        self.generation_seed = int(generation["seed"])

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def _release_analysis_gpu(self) -> None:
        """生成前卸载理解模型，为 8GB 显存和系统内存释放空间。"""
        try:
            import gc
            import torch

            for component in (self.detector, self.classifier, self.captioner):
                model = getattr(component, "_model", None) if component is not None else None
                if model is not None:
                    del model
                    component._model = None
                    # 允许下一次分析重新按需加载，而非永久标记不可用。
                    component._unavailable_reason = None
            if self.captioner is not None:
                self.captioner._processor = None
            if self.classifier is not None:
                self.classifier._preprocess = None
            gc.collect()
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def run(
        self,
        image_path: PathLike,
        save_panel: bool = True,
        generate: Optional[bool] = None,
        prompt_mode: str = "semantic",
    ) -> Dict[str, Any]:
        """对单张图像执行完整流程。

        Args:
            image_path: 图像路径。
            save_panel: 是否保存预处理特征面板图到输出目录。

        Returns:
            结果字典：
            - image_path: 输入路径
            - feature_summary: 预处理特征摘要（尺寸、边缘密度）
            - panel_path: 特征面板保存路径（save_panel=False 时为 None）
            - detections: Detection 列表
            - classifications: Classification 列表
            - caption / caption_fallback: 描述文本与是否降级
            - prompt / negative_prompt: 优化后的正负向提示词
            - prompt_score: Prompt 质量总分
            - degraded_notes: 各环节降级说明列表
        """
        if prompt_mode not in {"semantic", "reconstruction"}:
            raise ValueError("prompt_mode 必须是 semantic 或 reconstruction")
        image_path = Path(image_path)
        image = self.processor.load(image_path)
        degraded_notes: List[str] = []

        # 1) 预处理与特征提取
        panel, feature_summary = self.processor.make_feature_panel(image)
        panel_path: Optional[Path] = None
        if save_panel:
            panel_path = self.output_dir / f"{image_path.stem}_panel.png"
            self.processor.save(panel, panel_path)

        # 2) 目标检测与图像分类（模型不可用时自动降级为空结果）
        detections = self.detector.detect(image)
        if not detections and self.detector.unavailable_reason:
            degraded_notes.append(f"目标检测未启用：{self.detector.unavailable_reason}")
        classifications = self.classifier.classify(image) if self.classifier else []
        if self.classifier and not classifications and self.classifier.unavailable_reason:
            degraded_notes.append(f"图像分类未启用：{self.classifier.unavailable_reason}")

        # 3) VLM 图像描述（含降级路径）
        caption_result = self.captioner.caption(image, detections, classifications)
        if caption_result.fallback and caption_result.reason:
            degraded_notes.append(f"VLM 描述使用降级路径：{caption_result.reason}")

        # 4) Prompt 工程：构建 + 迭代优化
        objects = [d.label for d in detections]
        visual_context = describe_visual_context(image, detections)
        prompt_min_score = float(self.config["classification"].get("prompt_min_score", 0.2))
        scene_tags = [
            c.label.replace("_", " ")
            for c in classifications
            if c.score >= prompt_min_score
        ][:2]
        scene_tags.extend(str(tag) for tag in visual_context["prompt_tags"])
        reconstruction = build_reconstruction_prompt(image, caption_result.caption, detections)
        if prompt_mode == "semantic":
            optimized = self.optimizer.optimize(
                caption=caption_result.caption,
                objects=objects,
                scene_tags=scene_tags,
                rounds=self.optimize_rounds,
            )
            prompt = optimized.prompt
            negative_prompt = optimized.negative_prompt if self.negative_prompt_enabled else ""
            prompt_score = optimized.score.total
            prompt_history = optimized.history
        else:
            prompt = str(reconstruction["prompt"])
            negative_prompt = str(reconstruction["negative_prompt"]) if self.negative_prompt_enabled else ""
            prompt_score = None
            prompt_history = [{"prompt": prompt, "total": None}]

        should_generate = self.generation_enabled if generate is None else generate
        generated = None
        if should_generate:
            self._release_analysis_gpu()
            generated = self.generator.generate(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=self.output_dir / f"{image_path.stem}_generated.png",
                seed=self.generation_seed,
            )
            if generated.fallback and generated.reason:
                degraded_notes.append(f"参考图生成未完成：{generated.reason}")

        return {
            "image_path": str(image_path),
            "feature_summary": feature_summary,
            "panel_path": str(panel_path) if panel_path else None,
            "detections": detections,
            "classifications": classifications,
            "visual_context": visual_context,
            "reconstruction_context": reconstruction,
            "prompt_mode": prompt_mode,
            "caption": caption_result.caption,
            "caption_fallback": caption_result.fallback,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_score": prompt_score,
            "prompt_history": prompt_history,
            "generation": generated,
            "degraded_notes": degraded_notes,
        }
