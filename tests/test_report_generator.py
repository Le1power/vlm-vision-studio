"""ReportGenerator 单元测试：纯文本生成逻辑，不涉及模型下载。"""

from pathlib import Path

from src.models.detector import Detection
from src.report.report_generator import ReportGenerator


def _aggregate() -> dict:
    return {
        "mean_detections": 2.0,
        "mean_score": 0.85,
        "mean_caption_words": 8.0,
        "mean_ttr": 0.9,
        "caption_quality_pass_rate": 1.0,
        "vlm_fallback_rate": 0.0,
    }


def _samples() -> list:
    return [
        {
            "image_path": "assets/examples/sample_shapes.png",
            "detections": [Detection(label="dog", score=0.91, box=[1, 2, 3, 4])],
            "caption": "a dog sitting on green grass in a park",
            "caption_fallback": False,
            "prompt": "cinematic photo of a dog sitting on green grass, high detail",
            "negative_prompt": "blurry, low quality",
            "prompt_score": 82.5,
        },
        {
            "image_path": "assets/examples/sample_gradient.png",
            "detections": [],
            "caption": "an image with visual content",
            "caption_fallback": True,
            "prompt": "cinematic photo of an image with visual content",
            "negative_prompt": "blurry, low quality",
            "prompt_score": 60.0,
        },
    ]


class TestBuildMarkdown:
    def test_contains_all_sections(self) -> None:
        md = ReportGenerator().build_markdown(_aggregate(), _samples())
        assert md.startswith("# 算法测试报告")
        for section in ("## 一、聚合指标", "## 二、逐样本明细", "## 三、样例展示", "## 四、结论与局限"):
            assert section in md

    def test_metrics_in_table(self) -> None:
        md = ReportGenerator().build_markdown(_aggregate(), _samples())
        assert "| 平均检测置信度 | 0.85 |" in md
        assert "| 描述质量合格率 | 1.0 |" in md
        assert "sample_shapes.png" in md

    def test_fallback_marked(self) -> None:
        md = ReportGenerator().build_markdown(_aggregate(), _samples())
        assert "| 2 | sample_gradient.png | 0 |" in md
        # 第二样本降级列应为“是”
        lines = [l for l in md.splitlines() if l.startswith("| 2 |")]
        assert lines and lines[0].rstrip().endswith("| 是 |")

    def test_extra_notes_appended(self) -> None:
        md = ReportGenerator().build_markdown(
            _aggregate(), _samples(), extra_notes=["torch 未安装，走降级路径"]
        )
        assert "## 五、附加说明" in md
        assert "torch 未安装，走降级路径" in md

    def test_empty_samples(self) -> None:
        md = ReportGenerator().build_markdown({}, [])
        assert "评估样本数：0" in md


class TestSave:
    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        gen = ReportGenerator()
        path = gen.save("# test\n", tmp_path / "a" / "b" / "report.md")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "# test\n"
