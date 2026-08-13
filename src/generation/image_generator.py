"""Stable Diffusion img2img，支持可选 IP-Adapter 与 Canny ControlNet。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

PathLike = Union[str, Path]



@dataclass
class GenerationResult:
    image_path: Optional[str]
    mode: str
    seed: int
    strength: float
    fallback: bool = False
    reason: Optional[str] = None
    pixel_similarity: Optional[float] = None


class ReferenceImageGenerator:
    """按需加载 Diffusers 管线，以参考图结构为主要条件生成新图。"""

    def __init__(
        self,
        model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        mode: str = "img2img",
        strength: float = 0.28,
        guidance_scale: float = 7.0,
        steps: int = 25,
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_weight: str = "ip-adapter_sd15.bin",
        ip_adapter_scale: float = 0.65,
        controlnet_model: str = "lllyasviel/sd-controlnet-canny",
        device: Optional[str] = None,
    ) -> None:
        if mode not in {"img2img", "ip_adapter", "controlnet", "hybrid"}:
            raise ValueError("mode 必须是 img2img、ip_adapter、controlnet 或 hybrid")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength 必须在 0 到 1 之间")
        self.model_name = model_name
        self.mode = mode
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_weight = ip_adapter_weight
        self.ip_adapter_scale = ip_adapter_scale
        self.controlnet_model = controlnet_model
        self.device = device
        self._pipeline: Any = None
        self._loaded_mode: Optional[str] = None
        self._reason: Optional[str] = None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._reason

    def _ensure_pipeline(self) -> bool:
        if self._pipeline is not None and self._loaded_mode == self.mode:
            return True
        if self._pipeline is not None:
            try:
                import torch

                del self._pipeline
                self._pipeline = None
                torch.cuda.empty_cache()
            except Exception:
                self._pipeline = None
        self._reason = None
        if self._reason:
            return False
        try:
            import torch
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetImg2ImgPipeline,
                StableDiffusionImg2ImgPipeline,
            )
        except ImportError as exc:
            self._reason = f"生成依赖不可用: {exc}"
            return False
        if not torch.cuda.is_available():
            self._reason = "Stable Diffusion 生成需要 CUDA 版 PyTorch 和可用的 NVIDIA GPU"
            return False
        try:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            dtype = torch.float16
            if self.mode in {"controlnet", "hybrid"}:
                controlnet = ControlNetModel.from_pretrained(self.controlnet_model, torch_dtype=dtype)
                pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    self.model_name, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
                )
            else:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_name, torch_dtype=dtype, safety_checker=None
                )
                if self.mode == "ip_adapter":
                    pipeline.load_ip_adapter(
                        self.ip_adapter_repo,
                        subfolder="models",
                        weight_name=self.ip_adapter_weight,
                        image_encoder_folder=str(
                            Path(self.ip_adapter_repo).resolve() / "image_encoder"
                        ),
                    )
                    pipeline.set_ip_adapter_scale(self.ip_adapter_scale)
            pipeline.enable_attention_slicing()
            pipeline.vae.enable_slicing()
            # 8GB RTX 4060 可容纳 SD 1.5 + 单个条件模型。直接驻留 GPU
            # 比 CPU offload 更少占用系统内存，也避免 Windows 页面交换崩溃。
            pipeline.to("cuda")
            self._pipeline = pipeline
            self._loaded_mode = self.mode
        except Exception as exc:
            self._reason = f"生成模型加载失败: {exc}"
            return False
        return True

    @staticmethod
    def _prepare_image(image: np.ndarray) -> Any:
        from PIL import Image

        pil = Image.fromarray(image).convert("RGB")
        width, height = pil.size
        scale = min(1.0, 768 / max(width, height))
        size = (max(64, int(width * scale) // 8 * 8), max(64, int(height * scale) // 8 * 8))
        return pil.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _color_match(generated: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """将生成图逐通道均值和标准差匹配回参考图。"""
        source = generated.astype(np.float32)
        target = reference.astype(np.float32)
        for channel in range(3):
            src = source[..., channel]
            ref = target[..., channel]
            source[..., channel] = (
                (src - src.mean()) * (ref.std() / max(src.std(), 1e-6)) + ref.mean()
            )
        return np.clip(source, 0, 255).astype(np.uint8)

    @staticmethod
    def _pixel_similarity(generated: np.ndarray, reference: np.ndarray) -> float:
        mae = float(np.abs(generated.astype(np.float32) - reference.astype(np.float32)).mean())
        return round(max(0.0, 1.0 - mae / 255.0), 4)

    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        negative_prompt: str,
        output_path: PathLike,
        seed: int = 42,
    ) -> GenerationResult:
        if not self._ensure_pipeline():
            return GenerationResult(None, self.mode, seed, self.strength, True, self._reason)
        import torch
        from PIL import Image, ImageFilter

        init_image = self._prepare_image(image)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "image": init_image,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.steps,
            "generator": generator,
        }
        if self.mode == "ip_adapter":
            kwargs["ip_adapter_image"] = init_image
        if self.mode in {"controlnet", "hybrid"}:
            try:
                import cv2

                edges = cv2.Canny(np.asarray(init_image), 100, 200)
                control = Image.fromarray(np.repeat(edges[..., None], 3, axis=2))
            except ImportError:
                control = init_image.filter(ImageFilter.FIND_EDGES)
            kwargs["control_image"] = control
            kwargs["controlnet_conditioning_scale"] = 0.8
        try:
            generated = self._pipeline(**kwargs).images[0].convert("RGB")
            reference = np.asarray(init_image)
            generated_array = np.asarray(generated)
            if self.mode == "hybrid":
                matched = self._color_match(generated_array, reference)
                # strength 越低，保留的原图像素越多；最高保留 45%。
                reference_weight = min(0.45, max(0.15, 0.52 - self.strength))
                generated_array = np.clip(
                    matched.astype(np.float32) * (1.0 - reference_weight)
                    + reference.astype(np.float32) * reference_weight,
                    0,
                    255,
                ).astype(np.uint8)
                generated = Image.fromarray(generated_array)
            similarity = self._pixel_similarity(generated_array, reference)
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            generated.save(path)
            return GenerationResult(
                str(path), self.mode, seed, self.strength, pixel_similarity=similarity
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            return GenerationResult(None, self.mode, seed, self.strength, True, f"图像生成失败: {exc}")
