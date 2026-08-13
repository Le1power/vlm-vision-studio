"""批量评估脚本：对图像目录执行端到端评估并生成 Markdown 测试报告。

用法::

    python scripts/run_evaluation.py --image-dir assets/examples
    python scripts/run_evaluation.py --image-dir data/val --report reports/report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从项目根目录直接运行脚本
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline  # noqa: E402
from src.evaluation.evaluator import Evaluator  # noqa: E402
from src.report.report_generator import ReportGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="vlm-vision-studio 批量评估与报告生成")
    parser.add_argument("--image-dir", default="assets/examples", help="待评估图像目录")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--report", help="报告输出路径（缺省时使用配置中的 report_dir）")
    args = parser.parse_args()

    pipeline = Pipeline(config_path=args.config)
    evaluation_config = pipeline.config["evaluation"]
    evaluator = Evaluator(
        pipeline=pipeline,
        min_caption_words=int(evaluation_config["min_caption_words"]),
        min_ttr=float(evaluation_config["min_ttr"]),
    )
    summary = evaluator.evaluate_directory(args.image_dir)

    # 转换为报告生成器所需的字典结构
    samples = []
    degraded_any: set = set()
    for s in summary.samples:
        out = s.pipeline_output
        for note in out.get("degraded_notes", []):
            degraded_any.add(str(note))
        samples.append(
            {
                "image_path": s.image_path,
                "error": s.error,
                "detections": out.get("detections", []),
                "caption": out.get("caption", ""),
                "caption_fallback": out.get("caption_fallback", False),
                "prompt": out.get("prompt", ""),
                "negative_prompt": out.get("negative_prompt", ""),
                "prompt_score": out.get("prompt_score", "-"),
            }
        )
        if s.error:
            degraded_any.add(f"{Path(s.image_path).name} 处理失败：{s.error}")

    generator = ReportGenerator()
    markdown = generator.build_markdown(
        aggregate=summary.aggregate,
        samples=samples,
        extra_notes=sorted(degraded_any) or None,
    )
    report_path = Path(args.report) if args.report else Path(pipeline.config["paths"]["report_dir"]) / "test_report.md"
    report_path = generator.save(markdown, report_path)
    print(f"评估完成：{summary.num_images} 张图像")
    print(f"聚合指标: {summary.aggregate}")
    print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    main()
