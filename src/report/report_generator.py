"""Markdown 算法测试报告生成模块。

输入评估汇总（EvaluationSummary 或等价的字典结构），输出结构化
Markdown 报告：概览、聚合指标表、逐样本明细、结论与局限说明。
纯 Python 实现，不依赖深度学习框架，可独立测试。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]


class ReportGenerator:
    """算法测试报告生成器。

    Args:
        title: 报告标题。
        project_name: 项目名称。
    """

    def __init__(
        self,
        title: str = "算法测试报告",
        project_name: str = "vlm-vision-studio",
    ) -> None:
        self.title = title
        self.project_name = project_name

    # ------------------------------------------------------------------
    # 报告构建
    # ------------------------------------------------------------------
    def build_markdown(
        self,
        aggregate: Dict[str, object],
        samples: List[Dict[str, object]],
        extra_notes: Optional[List[str]] = None,
    ) -> str:
        """构建 Markdown 报告文本。

        Args:
            aggregate: 聚合指标字典（键见 Evaluator._aggregate）。
            samples: 逐样本字典列表，每项需含 image_path / detections /
                caption / prompt / prompt_score / caption_fallback 等键。
            extra_notes: 附加说明（如运行环境、降级情况）。

        Returns:
            完整 Markdown 文本。
        """
        lines: List[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"- 项目：`{self.project_name}`")
        lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 评估样本数：{len(samples)}")
        lines.append("")

        lines.append("## 一、聚合指标")
        lines.append("")
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("| --- | --- | --- |")
        agg_rows = [
            ("平均检测目标数", aggregate.get("mean_detections", "-"), "Faster R-CNN 检出框数量均值"),
            ("平均检测置信度", aggregate.get("mean_score", "-"), "检出框置信度均值"),
            ("平均描述词数", aggregate.get("mean_caption_words", "-"), "VLM 描述长度均值"),
            ("平均词汇丰富度 TTR", aggregate.get("mean_ttr", "-"), "唯一词/总词数"),
            ("描述质量合格率", aggregate.get("caption_quality_pass_rate", "-"), "长度与 TTR 双达标比例"),
            ("VLM 降级率", aggregate.get("vlm_fallback_rate", "-"), "未使用真实 BLIP 的样本比例"),
        ]
        for name, value, note in agg_rows:
            lines.append(f"| {name} | {value} | {note} |")
        lines.append("")

        lines.append("## 二、逐样本明细")
        lines.append("")
        lines.append("| # | 图像 | 检测数 | 描述（截断） | Prompt 总分 | VLM 降级 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for idx, s in enumerate(samples, 1):
            image = self._cell(Path(str(s.get("image_path", "-"))).name)
            det = s.get("detections") or []
            caption = self._cell(str(s.get("caption", ""))[:40])
            prompt_score = s.get("prompt_score", "-")
            fallback = "是" if s.get("caption_fallback") else "否"
            if s.get("error"):
                caption = self._cell(f"处理失败: {s['error']}")[:40]
            lines.append(f"| {idx} | {image} | {len(det)} | {caption} | {prompt_score} | {fallback} |")
        lines.append("")

        lines.append("## 三、样例展示")
        lines.append("")
        for idx, s in enumerate(samples[:3], 1):
            lines.append(f"### 样例 {idx}：`{Path(str(s.get('image_path', '-'))) .name}`")
            lines.append("")
            lines.append(f"- **图像描述**：{s.get('caption', '')}")
            det = s.get("detections") or []
            if det:
                det_str = "、".join(
                    f"{d.label}({d.score:.2f})" if hasattr(d, "label") else str(d) for d in det[:5]
                )
                lines.append(f"- **检测目标**：{det_str}")
            else:
                lines.append("- **检测目标**：无（或检测模型不可用）")
            lines.append(f"- **生成 Prompt**：{s.get('prompt', '')}")
            neg = s.get("negative_prompt", "")
            if neg:
                lines.append(f"- **负向 Prompt**：{neg}")
            lines.append("")

        lines.append("## 四、结论与局限")
        lines.append("")
        lines.extend(self._conclusions(aggregate))
        lines.append("")
        if extra_notes:
            lines.append("## 五、附加说明")
            lines.append("")
            for note in extra_notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)

    def save(self, markdown: str, path: PathLike) -> Path:
        """保存报告到磁盘，自动创建父目录。

        Args:
            markdown: 报告文本。
            path: 目标路径。

        Returns:
            实际写入路径。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _cell(text: str) -> str:
        """转义 Markdown 表格单元格内容：``|`` 与换行会破坏表格结构。"""
        return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _conclusions(aggregate: Dict[str, object]) -> List[str]:
        """根据聚合指标生成结论文本。"""
        lines: List[str] = []
        pass_rate = float(aggregate.get("caption_quality_pass_rate") or 0.0)
        fallback_rate = float(aggregate.get("vlm_fallback_rate") or 0.0)
        if pass_rate >= 0.8:
            lines.append("- 描述质量整体良好：多数样本在长度与词汇丰富度上达标。")
        elif pass_rate >= 0.5:
            lines.append("- 描述质量中等：部分样本偏短或词汇重复，可通过 Prompt 约束改进。")
        else:
            lines.append("- 描述质量偏低：建议检查输入图像质量或调整生成参数。")
        if fallback_rate > 0:
            lines.append(
                f"- 有 {fallback_rate:.0%} 的样本使用降级描述（未加载真实 BLIP），"
                "相应指标仅反映模板输出，不代表 VLM 真实能力。"
            )
        lines.append("- 检测精度指标基于预训练模型零样本输出，未针对特定场景微调。")
        lines.append("- CLIPScore 图文一致性指标当前为占位，后续可接入 open_clip 补充。")
        return lines
