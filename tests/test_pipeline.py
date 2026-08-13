"""Pipeline 配置加载与开关行为测试（不触发模型下载）。"""

from pathlib import Path

import pytest

from src.pipeline import Pipeline, load_config
from src.models.classifier import ImageClassifier
from src.models.detector import ObjectDetector
from src.models.vlm_captioner import VLMCaptioner


def test_default_config_contains_all_runtime_sections() -> None:
    config = load_config(None)
    assert config["evaluation"] == {"min_caption_words": 5, "min_ttr": 0.4}
    assert config["paths"]["report_dir"] == "reports"
    assert config["detection"]["max_side"] == 1024
    assert config["classification"]["prompt_min_score"] == 0.2


def test_config_recursively_merges_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("detection:\n  score_threshold: 0.75\n", encoding="utf-8")

    config = load_config(path)

    assert config["detection"]["score_threshold"] == 0.75
    assert config["detection"]["max_detections"] == 20
    assert config["vlm"]["fallback_enabled"] is True


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("detection: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="无法读取配置文件"):
        load_config(path)


def test_negative_prompt_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("prompt:\n  negative_prompt: false\n", encoding="utf-8")
    pipeline = Pipeline(path, enable_classification=False)

    assert pipeline.negative_prompt_enabled is False


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ObjectDetector(score_threshold=1.1), "score_threshold"),
        (lambda: ObjectDetector(max_detections=0), "max_detections"),
        (lambda: ImageClassifier(top_k=0), "top_k"),
        (lambda: VLMCaptioner(max_new_tokens=0), "max_new_tokens"),
    ],
)
def test_model_parameters_are_validated(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
