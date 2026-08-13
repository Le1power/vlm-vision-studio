"""vlm-vision-studio：基于 VLM 视觉语言大模型的多模态视觉内容理解与创作平台。

模块结构：
- preprocessing：OpenCV 图像预处理与特征提取
- models：目标检测 / 图像分类 / VLM 图像描述（全部懒加载，支持优雅降级）
- prompt_engineering：AIGC 文生图提示词模板、优化与评分
- evaluation：检测与描述质量评估指标
- report：Markdown 算法测试报告生成
- pipeline：端到端流程串联
"""

__version__ = "0.1.0"
