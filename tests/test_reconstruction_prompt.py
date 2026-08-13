"""原图复刻 Prompt 的几何与布局测试。"""

import numpy as np

from src.prompt_engineering.reconstruction_prompt import build_reconstruction_prompt


def test_reconstruction_prompt_avoids_cinematic_expansion() -> None:
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    image[25:75, 40:90] = (50, 100, 220)

    result = build_reconstruction_prompt(image, "a blue shape", [])

    assert "faithful reconstruction" in result["prompt"]
    assert "no extra objects" in result["prompt"]
    assert "preserve exact object count" in result["detailed_prompt"]
    assert "dramatic lighting" not in result["prompt"]
    assert "different composition" in result["negative_prompt"]
    assert result["orientation"] == "landscape"
    assert result["style_tags"]
    assert "color_bins" in result["style_metrics"]


def test_reconstruction_prompt_reports_elements() -> None:
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    image[30:90, 30:90] = (220, 50, 50)

    result = build_reconstruction_prompt(image, "a red square", [])

    assert result["elements"]
    assert any(item["shape"] in {"square", "rectangle"} for item in result["elements"])
