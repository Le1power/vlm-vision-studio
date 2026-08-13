"""Prompt 工程模块：将视觉理解结果组织/优化为 AIGC 文生图提示词。

能力概览：
- 模板库：多种风格（cinematic / anime / oil_painting / photorealistic）；
- Few-shot 示例：内置「视觉理解结果 → 高质量提示词」示例对；
- 负向提示词：按风格附加常见负向词，提升文生图稳定性；
- 质量评分：从长度、关键词覆盖、具体性、重复度四个维度打分（0~100）；
- 迭代优化：依据评分短板自动补充细节，多轮迭代得到更优提示词。

本模块为纯 Python 实现，不依赖任何深度学习框架，可独立测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 模板库与 few-shot 示例
# ---------------------------------------------------------------------------

STYLE_TEMPLATES: Dict[str, str] = {
    "cinematic": (
        "cinematic photo of {subject}, {details}, dramatic lighting, "
        "shallow depth of field, 35mm film, high detail"
    ),
    "anime": (
        "anime style illustration of {subject}, {details}, vibrant colors, "
        "clean line art, studio quality"
    ),
    "oil_painting": (
        "oil painting of {subject}, {details}, visible brush strokes, "
        "rich texture, classical composition"
    ),
    "photorealistic": (
        "ultra realistic photo of {subject}, {details}, natural lighting, "
        "sharp focus, 8k resolution"
    ),
}

DEFAULT_STYLE = "cinematic"

# 风格 -> 负向提示词（避免文生图常见缺陷）
NEGATIVE_PROMPTS: Dict[str, str] = {
    "cinematic": "blurry, low quality, watermark, text, distorted, oversaturated",
    "anime": "blurry, low quality, watermark, text, bad anatomy, extra limbs",
    "oil_painting": "blurry, low quality, watermark, text, photo, digital artifacts",
    "photorealistic": "blurry, cartoon, painting, watermark, text, deformed",
}

# Few-shot 示例：视觉理解结果 -> 高质量提示词（供模板填充与迭代参考）
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "caption": "a dog sitting on grass in a park",
        "objects": "dog, frisbee",
        "prompt": (
            "cinematic photo of a golden retriever sitting on green grass in a sunny park, "
            "a red frisbee beside it, dramatic lighting, shallow depth of field, 35mm film, high detail"
        ),
    },
    {
        "caption": "a person riding a bicycle on a city street",
        "objects": "person, bicycle, car",
        "prompt": (
            "cinematic photo of a person riding a bicycle on a busy city street at dusk, "
            "cars passing by, dramatic lighting, shallow depth of field, 35mm film, high detail"
        ),
    },
]

# 具体性词表：命中越多说明描述越具体
_SPECIFITY_WORDS = {
    "red", "blue", "green", "golden", "black", "white", "sunny", "rainy",
    "night", "dusk", "dawn", "young", "old", "small", "large", "wooden",
    "metal", "grass", "street", "park", "beach", "mountain", "river",
}

_WORD_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> List[str]:
    """将文本切分为小写单词序列。"""
    return _WORD_RE.findall(text.lower())


@dataclass
class PromptScore:
    """Prompt 质量评分结果。

    Attributes:
        total: 总分（0~100）。
        length_score: 长度得分（0~30），过短信息不足、过长易稀释主题。
        coverage_score: 关键词覆盖得分（0~30），视觉要素在提示词中的保留度。
        specificity_score: 具体性得分（0~20）。
        diversity_score: 多样性得分（0~20），基于词重复度。
        suggestions: 针对短板的优化建议。
    """

    total: float
    length_score: float
    coverage_score: float
    specificity_score: float
    diversity_score: float
    suggestions: List[str] = field(default_factory=list)


@dataclass
class OptimizedPrompt:
    """优化后的提示词结果。

    Attributes:
        prompt: 正向提示词。
        negative_prompt: 负向提示词。
        score: 质量评分。
        rounds: 实际执行的改进轮数（不含初版构建；提前收敛时小于上限）。
        history: 每轮的 (prompt, total_score) 记录。
    """

    prompt: str
    negative_prompt: str
    score: PromptScore
    rounds: int
    history: List[Dict[str, object]] = field(default_factory=list)


class PromptOptimizer:
    """AIGC 文生图提示词优化器。

    Args:
        style: 风格模板名，见 ``STYLE_TEMPLATES``。
        language: 保留字段，当前模板均为英文（主流文生图模型对英文更友好）。
    """

    def __init__(self, style: str = DEFAULT_STYLE, language: str = "en") -> None:
        if style not in STYLE_TEMPLATES:
            raise ValueError(f"未知风格 '{style}'，可选: {sorted(STYLE_TEMPLATES)}")
        self.style = style
        self.language = language

    # ------------------------------------------------------------------
    # 提示词构建
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        caption: str,
        objects: Optional[Sequence[str]] = None,
        scene_tags: Optional[Sequence[str]] = None,
    ) -> str:
        """基于视觉理解结果构建初版提示词。

        Args:
            caption: VLM 图像描述。
            objects: 检测到的目标类别列表。
            scene_tags: 额外场景标签（如分类结果、边缘复杂度等）。

        Returns:
            填充模板后的提示词。
        """
        template = STYLE_TEMPLATES[self.style]
        details = self._collect_details(objects, scene_tags)
        prompt = template.format(subject=caption or "a scene", details=details or "natural setting")
        return self._normalize(prompt)

    def build_negative_prompt(self) -> str:
        """返回当前风格对应的负向提示词。"""
        return NEGATIVE_PROMPTS[self.style]

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """返回内置 few-shot 示例（caption/objects -> prompt）。"""
        return [dict(item) for item in FEW_SHOT_EXAMPLES]

    # ------------------------------------------------------------------
    # 评分
    # ------------------------------------------------------------------
    def score_prompt(
        self,
        prompt: str,
        reference_keywords: Optional[Sequence[str]] = None,
    ) -> PromptScore:
        """对提示词进行四维质量评分。

        Args:
            prompt: 待评分提示词。
            reference_keywords: 视觉理解得到的关键词（用于覆盖度评估）。

        Returns:
            PromptScore，含各维度得分与优化建议。
        """
        words = _tokenize(prompt)
        n = len(words)
        if n == 0:  # 空提示词直接零分并给出建议
            return PromptScore(
                total=0.0,
                length_score=0.0,
                coverage_score=0.0,
                specificity_score=0.0,
                diversity_score=0.0,
                suggestions=["提示词为空，请先基于视觉理解结果构建提示词"],
            )

        # 1) 长度得分：目标区间 15~45 词，区间外线性衰减
        if 15 <= n <= 45:
            length_score = 30.0
        elif n < 15:
            length_score = 30.0 * n / 15
        else:
            length_score = max(0.0, 30.0 * (1 - (n - 45) / 45))

        # 2) 关键词覆盖：参考词在提示词中出现的比例
        refs = [k.lower() for k in (reference_keywords or []) if k.strip()]
        if refs:
            word_set = set(words)
            hit = sum(1 for k in refs if k in word_set)
            coverage_score = 30.0 * hit / len(refs)
        else:
            coverage_score = 30.0  # 无参考词时视为全覆盖

        # 3) 具体性：命中具体性词表的比例（封顶）
        hits = sum(1 for w in set(words) if w in _SPECIFITY_WORDS)
        specificity_score = min(20.0, hits * 5.0)

        # 4) 多样性：唯一词占比（TTR），重复越多得分越低
        ttr = len(set(words)) / n if n else 0.0
        diversity_score = 20.0 * ttr

        total = length_score + coverage_score + specificity_score + diversity_score
        suggestions: List[str] = []
        if length_score < 20:
            suggestions.append("提示词长度偏离理想区间（15~45 词），请增删细节")
        if coverage_score < 20:
            suggestions.append("部分视觉要素未进入提示词，请补充关键目标/场景词")
        if specificity_score < 10:
            suggestions.append("描述偏抽象，请加入颜色、时间、材质等具体修饰词")
        if diversity_score < 14:
            suggestions.append("词汇重复较多，请用同义替换丰富表达")

        return PromptScore(
            total=round(total, 2),
            length_score=round(length_score, 2),
            coverage_score=round(coverage_score, 2),
            specificity_score=round(specificity_score, 2),
            diversity_score=round(diversity_score, 2),
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # 迭代优化
    # ------------------------------------------------------------------
    def optimize(
        self,
        caption: str,
        objects: Optional[Sequence[str]] = None,
        scene_tags: Optional[Sequence[str]] = None,
        rounds: int = 2,
    ) -> OptimizedPrompt:
        """构建提示词并按评分短板迭代优化。

        每轮依据评分建议自动补充：缺失关键词、具体性修饰词。

        Args:
            caption: VLM 图像描述。
            objects: 检测目标列表。
            scene_tags: 场景标签列表。
            rounds: 最大迭代轮数。

        Returns:
            OptimizedPrompt，含最终提示词、负向提示词、评分与迭代历史。
        """
        keywords = self._reference_keywords(caption, objects, scene_tags)
        prompt = self.build_prompt(caption, objects, scene_tags)
        history: List[Dict[str, object]] = []
        best_prompt, best_score = prompt, self.score_prompt(prompt, keywords)
        history.append({"prompt": prompt, "total": best_score.total})

        for _ in range(max(0, rounds)):
            improved = self._improve_once(best_prompt, best_score, keywords)
            if improved == best_prompt:
                break  # 无可改进项，提前收敛
            score = self.score_prompt(improved, keywords)
            if score.total <= best_score.total:
                break  # 不再提升则保留上一版，且不记录退化轮
            best_prompt, best_score = improved, score
            history.append({"prompt": improved, "total": score.total})

        return OptimizedPrompt(
            prompt=best_prompt,
            negative_prompt=self.build_negative_prompt(),
            score=best_score,
            rounds=len(history) - 1,  # history 首条为初版，其余为改进轮
            history=history,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        """压缩多余空格并清理重复逗号。"""
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"(,\s*){2,}", ", ", text)

    @staticmethod
    def _collect_details(
        objects: Optional[Sequence[str]],
        scene_tags: Optional[Sequence[str]],
    ) -> str:
        """汇总检测目标与场景标签为细节短语。"""
        parts: List[str] = []
        seen = set()
        for item in list(objects or []) + list(scene_tags or []):
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                parts.append(key)
        return ", ".join(parts)

    @staticmethod
    def _reference_keywords(
        caption: str,
        objects: Optional[Sequence[str]],
        scene_tags: Optional[Sequence[str]],
    ) -> List[str]:
        """汇总用于覆盖度评估的参考关键词。"""
        keywords = [w for w in _tokenize(caption) if len(w) > 3]
        keywords += [k.strip().lower() for k in (objects or [])]
        keywords += [k.strip().lower() for k in (scene_tags or [])]
        return list(dict.fromkeys(k for k in keywords if k))

    def _improve_once(
        self,
        prompt: str,
        score: PromptScore,
        keywords: Sequence[str],
    ) -> str:
        """依据评分短板对提示词做一轮自动改进。

        - 覆盖度不足：追加缺失的参考关键词；
        - 具体性不足：追加通用具体修饰词；
        - 长度过短：追加细节短语。
        """
        additions: List[str] = []
        if score.coverage_score < 30 and keywords:
            word_set = set(_tokenize(prompt))
            missing = [k for k in keywords if k not in word_set][:3]
            additions.extend(missing)
        if score.specificity_score < 10:
            for word in ("detailed", "natural lighting", "sharp focus"):
                if word not in prompt:
                    additions.append(word)
                    break
        if score.length_score < 20:
            additions.append("intricate details")
        if not additions:
            return prompt
        return self._normalize(prompt + ", " + ", ".join(additions))
