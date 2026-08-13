"""单张图像端到端演示脚本。

用法::

    python scripts/run_demo.py            # 交互模式：输入路径并确认后运行
    python scripts/run_demo.py --image assets/examples/sample_shapes.png
    python scripts/run_demo.py --image path/to/photo.jpg --config configs/default.yaml --no-panel

模型权重（torch/transformers）未安装或未下载时自动降级，流程仍可完整运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许从项目根目录直接运行脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline  # noqa: E402


def _serialize(value: object) -> object:
    """将 dataclass / 路径等对象转换为可 JSON 序列化的结构。"""
    if hasattr(value, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def _prompt_image_path(skip_confirm: bool = False) -> Path:
    """交互式获取图像路径：输入 -> 校验存在且可解码 -> 用户确认后返回。

    直接回车退出；路径无效时允许重新输入；确认环节输入 n 可重新选择。
    skip_confirm=True 时读取成功即返回，不再询问。
    """
    from src.preprocessing.image_processor import ImageProcessor

    processor = ImageProcessor()
    while True:
        try:
            raw = input("请输入图像文件路径（直接回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            raise SystemExit(0)
        if not raw:
            print("已退出。")
            raise SystemExit(0)
        path = Path(raw.strip('"\''))  # 容忍资源管理器拖拽带入的引号
        try:
            image = processor.load(path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ✗ {exc}，请重新输入。")
            continue
        h, w = image.shape[:2]
        if skip_confirm:
            print(f"  已读取 {path}（{w}x{h}），跳过确认直接运行。")
            return path
        try:
            confirm = input(f"  已读取 {path}（{w}x{h}），确认运行该图像？[Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            raise SystemExit(0)
        if confirm in ("", "y", "yes"):
            return path
        print("  已放弃该路径，请重新选择。")


def main() -> None:
    parser = argparse.ArgumentParser(description="vlm-vision-studio 单图端到端演示")
    parser.add_argument("--image", help="输入图像路径（缺省时启动交互式选择与确认）")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--no-panel", action="store_true", help="不保存预处理特征面板")
    parser.add_argument("--generate", action="store_true", help="使用参考图条件生成新图")
    parser.add_argument(
        "--prompt-mode", choices=("semantic", "reconstruction"), default="semantic",
        help="Prompt 模式：语义风格扩写或原图结构复刻说明",
    )
    parser.add_argument(
        "--mode", choices=("img2img", "ip_adapter", "controlnet", "hybrid"),
        help="参考图生成模式（覆盖配置文件）",
    )
    parser.add_argument("--strength", type=float, help="img2img 去噪强度，越低越接近原图")
    parser.add_argument("--seed", type=int, help="生成随机种子")
    parser.add_argument(
        "--yes", action="store_true",
        help="跳过确认直接运行（仅交互模式下有意义；--image 直传时本就不询问）",
    )
    args = parser.parse_args()

    if args.image:
        image_path: Path = Path(args.image)
    else:
        image_path = _prompt_image_path(skip_confirm=args.yes)

    pipeline = Pipeline(config_path=args.config)
    if args.mode:
        pipeline.generator.mode = args.mode
    if args.strength is not None:
        if not 0.0 <= args.strength <= 1.0:
            parser.error("--strength 必须在 0 到 1 之间")
        pipeline.generator.strength = args.strength
    if args.seed is not None:
        pipeline.generation_seed = args.seed
    result = pipeline.run(
        image_path,
        save_panel=not args.no_panel,
        generate=args.generate or None,
        prompt_mode=args.prompt_mode,
    )

    print("=" * 60)
    print(f"图像: {result['image_path']}")
    print(f"特征摘要: {json.dumps(result['feature_summary'], ensure_ascii=False)}")
    if result["panel_path"]:
        print(f"特征面板: {result['panel_path']}")
    print("-" * 60)
    detections = result["detections"]
    if detections:
        print(f"检测目标 ({len(detections)}):")
        for d in detections[:10]:
            print(f"  - {d.label}: {d.score:.3f} @ {d.box}")
    else:
        print("检测目标: 无（模型可能未启用，见降级说明）")
    classifications = result["classifications"]
    if classifications:
        print(f"分类 Top-{len(classifications)}:")
        for c in classifications:
            print(f"  - {c.label}: {c.score:.3f}")
    print("-" * 60)
    tag = "（降级生成）" if result["caption_fallback"] else "（BLIP 生成）"
    print(f"图像描述{tag}: {result['caption']}")
    score_text = f" (得分 {result['prompt_score']})" if result["prompt_score"] is not None else ""
    print(f"Prompt [{result['prompt_mode']}]{score_text}: {result['prompt']}")
    print(f"负向 Prompt: {result['negative_prompt']}")
    context = result["visual_context"]
    print(f"主色: {', '.join(context['dominant_colors']) or '未知'}")
    generation = result["generation"]
    if generation and generation.image_path:
        print(f"参考图生成结果: {generation.image_path} ({generation.mode}, strength={generation.strength})")
    if result["degraded_notes"]:
        print("-" * 60)
        print("降级说明:")
        for note in result["degraded_notes"]:
            print(f"  * {note}")
    print("=" * 60)

    # 输出 JSON 结果便于调试与复现（与是否保存特征面板无关）
    out_json = pipeline.output_dir / f"{image_path.stem}_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(_serialize(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完整结果已保存: {out_json}")


if __name__ == "__main__":
    main()
