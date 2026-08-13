"""PromptOptimizer 单元测试：纯 Python 逻辑，不涉及模型下载。"""

import pytest

from src.prompt_engineering.prompt_optimizer import (
    FEW_SHOT_EXAMPLES,
    NEGATIVE_PROMPTS,
    STYLE_TEMPLATES,
    PromptOptimizer,
)


@pytest.fixture()
def optimizer() -> PromptOptimizer:
    return PromptOptimizer(style="cinematic")


class TestBuild:
    def test_unknown_style_raises(self) -> None:
        with pytest.raises(ValueError):
            PromptOptimizer(style="not_a_style")

    def test_build_prompt_contains_caption_and_objects(self, optimizer: PromptOptimizer) -> None:
        prompt = optimizer.build_prompt(
            caption="a dog on grass",
            objects=["dog", "frisbee"],
            scene_tags=["park"],
        )
        assert "a dog on grass" in prompt
        assert "dog" in prompt and "frisbee" in prompt and "park" in prompt
        # 模板要素应被填充，不留占位符
        assert "{" not in prompt and "}" not in prompt

    def test_build_prompt_empty_caption_fallback(self, optimizer: PromptOptimizer) -> None:
        prompt = optimizer.build_prompt(caption="", objects=None)
        assert "a scene" in prompt

    def test_negative_prompt_matches_style(self, optimizer: PromptOptimizer) -> None:
        assert optimizer.build_negative_prompt() == NEGATIVE_PROMPTS["cinematic"]

    def test_few_shot_examples_structure(self, optimizer: PromptOptimizer) -> None:
        examples = optimizer.get_few_shot_examples()
        assert len(examples) == len(FEW_SHOT_EXAMPLES)
        for item in examples:
            assert {"caption", "objects", "prompt"} <= set(item)
        # 返回副本，修改不影响内置示例
        examples[0]["caption"] = "mutated"
        assert FEW_SHOT_EXAMPLES[0]["caption"] != "mutated"


class TestScoring:
    def test_empty_prompt_scores_zero(self, optimizer: PromptOptimizer) -> None:
        score = optimizer.score_prompt("")
        assert score.total == 0.0
        assert score.suggestions  # 空提示词必须给出建议

    def test_good_prompt_scores_higher(self, optimizer: PromptOptimizer) -> None:
        good = (
            "cinematic photo of a golden retriever sitting on green grass in a sunny park, "
            "a red frisbee beside it, dramatic lighting, shallow depth of field, high detail"
        )
        bad = "dog dog dog dog"
        good_score = optimizer.score_prompt(good, reference_keywords=["dog", "park"])
        bad_score = optimizer.score_prompt(bad, reference_keywords=["dog", "park"])
        assert good_score.total > bad_score.total

    def test_coverage_reflects_keywords(self, optimizer: PromptOptimizer) -> None:
        prompt = "cinematic photo of a dog in a sunny park, high detail"
        full = optimizer.score_prompt(prompt, reference_keywords=["dog", "park"])
        partial = optimizer.score_prompt(prompt, reference_keywords=["dog", "elephant"])
        assert full.coverage_score == 30.0
        assert partial.coverage_score == pytest.approx(15.0)

    def test_total_is_sum_of_parts(self, optimizer: PromptOptimizer) -> None:
        score = optimizer.score_prompt("a red car on a rainy street at night, sharp focus, high detail")
        assert score.total == pytest.approx(
            score.length_score + score.coverage_score
            + score.specificity_score + score.diversity_score
        )


class TestOptimize:
    def test_optimize_returns_prompt_and_history(self, optimizer: PromptOptimizer) -> None:
        result = optimizer.optimize(
            caption="a cat on a sofa",
            objects=["cat", "couch"],
            rounds=2,
        )
        assert result.prompt
        assert result.negative_prompt == NEGATIVE_PROMPTS["cinematic"]
        # rounds 为改进轮数（不含初版），history 多一条初版记录
        assert result.rounds == len(result.history) - 1 >= 0
        assert result.score.total == result.history[-1]["total"]

    def test_optimize_never_degrades(self, optimizer: PromptOptimizer) -> None:
        result = optimizer.optimize(caption="a red car on a rainy street at night", rounds=3)
        totals = [float(h["total"]) for h in result.history]
        assert totals == sorted(totals)  # 迭代过程单调不降

    def test_all_styles_have_templates_and_negatives(self) -> None:
        for style in STYLE_TEMPLATES:
            opt = PromptOptimizer(style=style)
            assert opt.build_negative_prompt()
            assert opt.build_prompt(caption="test scene")
