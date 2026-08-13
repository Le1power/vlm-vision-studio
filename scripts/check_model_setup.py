"""检查 GPU、依赖和手动下载的模型文件；本脚本不会访问网络。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import load_config  # noqa: E402


TORCHVISION_FILES = {
    "Faster R-CNN": "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
    "ResNet50": "resnet50-11ad3fa6.pth",
}


def _model_status(value: str) -> str:
    path = Path(value)
    if path.exists():
        return f"就绪（本地：{path.resolve()}）"
    try:
        from huggingface_hub import snapshot_download

        cached = snapshot_download(value, local_files_only=True)
        return f"就绪（Hugging Face 缓存：{cached}）"
    except Exception:
        pass
    return f"未发现本地目录（当前为在线标识：{value}）"


def _diffusers_model_status(value: str) -> str:
    path = Path(value)
    if path.exists():
        config = path / "config.json"
        weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        if config.is_file() and weights:
            return f"就绪（本地：{path.resolve()}）"
        return f"目录存在但不完整（需要 config.json 和权重文件）：{path.resolve()}"
    return _model_status(value)


def _ip_adapter_status(repo: str, weight_name: str) -> str:
    path = Path(repo)
    if path.exists():
        weight = path / "models" / weight_name
        image_encoder = path / "image_encoder"
        encoder_weights = list(image_encoder.glob("*.bin")) + list(image_encoder.glob("*.safetensors"))
        if not weight.is_file() or weight.stat().st_size <= 1024:
            return f"源码目录存在，但缺少权重：{weight.resolve()}"
        if not encoder_weights:
            return f"适配器权重已找到，但缺少 CLIP 图像编码器：{image_encoder.resolve()}"
        return f"就绪（权重：{weight.resolve()}；图像编码器：{image_encoder.resolve()}）"
    return _model_status(repo)


def main() -> int:
    parser = argparse.ArgumentParser(description="离线检查 VLM Vision Studio 模型环境")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    print("Python:", sys.executable)
    dependencies_ok = True
    for package in ("torch", "torchvision", "transformers", "diffusers", "accelerate"):
        present = importlib.util.find_spec(package) is not None
        dependencies_ok &= present
        print(f"依赖 {package}: {'已安装' if present else '缺失'}")

    try:
        import torch

        print("CUDA:", "可用" if torch.cuda.is_available() else "不可用")
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except ImportError:
        print("CUDA: 无法检查（torch 未安装）")

    checkpoints = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    for label, filename in TORCHVISION_FILES.items():
        path = checkpoints / filename
        print(f"{label}: {'就绪' if path.is_file() else '缺失'} - {path}")

    print("BLIP:", _model_status(str(config["vlm"]["model_name"])))
    generation = config["generation"]
    print("Stable Diffusion:", _model_status(str(generation["model_name"])))
    print(
        "IP-Adapter:",
        _ip_adapter_status(
            str(generation["ip_adapter_repo"]),
            str(generation["ip_adapter_weight"]),
        ),
    )
    print("ControlNet:", _diffusers_model_status(str(generation["controlnet_model"])))
    print("\n说明：标记为未发现的在线模型首次运行会尝试联网下载。")
    return 0 if dependencies_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
