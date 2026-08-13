# VLM Vision Studio

一个在本地运行的图像理解、Prompt 生成与参考图重建工具。项目以 BLIP 为视觉语言模型核心，结合 Faster R-CNN、ResNet50、Stable Diffusion 1.5、IP-Adapter 和 ControlNet，将输入图片转换为可阅读、可编辑、可用于图像生成的结构化描述。

> 仓库不包含模型权重。模型需按 [MODEL_SETUP.md](MODEL_SETUP.md) 单独准备。完整参数、配置和故障排查请阅读 [READ.md](READ.md)。

## 核心能力

- **图片理解**：BLIP 图像描述、Faster R-CNN 目标检测、ResNet50 分类。
- **视觉分析**：提取主色、画面尺寸、边缘密度、元素位置和空间关系。
- **双 Prompt 模式**：兼顾创意扩写与原图结构复刻说明。
- **参考图生成**：支持 `img2img`、`ip_adapter`、`controlnet` 和 `hybrid`。
- **本地交互界面**：通过 Streamlit 上传图片、调整参数并比较生成结果。
- **离线与降级机制**：模型懒加载，组件不可用时返回明确说明。

## 工作流程

```text
输入图片
  -> OpenCV 特征分析
  -> Faster R-CNN 检测 + ResNet50 分类
  -> BLIP 图像描述
  -> 颜色、尺寸、位置和空间关系提取
  -> Semantic / Reconstruction Prompt
  -> 可选 Stable Diffusion 参考图生成
  -> 图片、JSON 和分析结果
```

## Prompt 模式

### Semantic

概括图片主体和场景，并按所选风格扩写 Prompt，适合生成同主题的新作品。

支持风格：

- `cinematic`
- `anime`
- `oil_painting`
- `photorealistic`

### Reconstruction

描述画面比例、背景、颜色、几何轮廓、元素位置、尺寸、间距和留白，适合结合原图条件进行高相似度重建。

该模式返回两种文本：

- `prompt`：适合 Stable Diffusion 1.5 CLIP 长度限制的紧凑 Prompt。
- `reconstruction_context.detailed_prompt`：包含坐标和布局信息的完整复刻说明。

## 生成模式

| 模式 | 主要作用 | 建议 strength |
| --- | --- | ---: |
| `img2img` | 轻度润色并保留整体布局 | `0.20-0.30` |
| `ip_adapter` | 保留主体外观和视觉特征 | `0.20-0.35` |
| `controlnet` | 保留边缘、轮廓和构图 | `0.25-0.40` |
| `hybrid` | 兼顾结构、色彩和部分原图像素 | `0.15-0.30` |

扩散模型不会只凭文字 Prompt 精确恢复原图。需要较高相似度时，推荐使用：

```text
Reconstruction + Hybrid + strength 0.15-0.25
```

## 环境要求

- Python 3.9 或兼容版本
- NVIDIA GPU 与 CUDA 版 PyTorch（执行 Stable Diffusion 生成时需要）
- 项目已在 RTX 4060 Laptop 8GB、PyTorch `2.7.1+cu118` 环境验证

安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

检查依赖、CUDA 和本地模型：

```powershell
python scripts/check_model_setup.py
```

检查脚本不会主动下载模型。

## 模型准备

默认配置期望以下本地目录：

```text
models/
|- stable-diffusion-v1-5/
|- IP-Adapter/
`- sd-controlnet-canny/
```

模型权重体积较大并已被 `.gitignore` 排除。目录结构、文件要求与离线配置见 [MODEL_SETUP.md](MODEL_SETUP.md)。

## 快速开始

### 图形界面

```powershell
python scripts/start_app.py
```

默认访问：

```text
http://127.0.0.1:8501
```

基本操作：

1. 上传图片。
2. 选择 Semantic 或 Reconstruction 模式。
3. 只需要 Prompt 时关闭图片生成。
4. 需要新图时选择生成模式、strength 和 seed。
5. 点击“分析图片”查看 Prompt、特征面板和生成结果。

### 命令行

只生成 Prompt：

```powershell
python scripts/run_demo.py --image assets/examples/sample_shapes.png
```

生成复刻说明：

```powershell
python scripts/run_demo.py `
  --image assets/examples/sample_shapes.png `
  --prompt-mode reconstruction `
  --no-panel
```

执行高保真参考图生成：

```powershell
python scripts/run_demo.py `
  --image path/to/image.png `
  --prompt-mode reconstruction `
  --generate `
  --mode hybrid `
  --strength 0.20 `
  --seed 42
```

## Python API

网站或其他 Python 应用应直接复用 `Pipeline`，不要解析命令行输出：

```python
from src.pipeline import Pipeline

pipeline = Pipeline("configs/default.yaml")
pipeline.generator.mode = "hybrid"
pipeline.generator.strength = 0.20
pipeline.generation_seed = 42

result = pipeline.run(
    "path/to/image.png",
    save_panel=True,
    generate=True,
    prompt_mode="reconstruction",
)

print(result["caption"])
print(result["prompt"])
print(result["reconstruction_context"]["detailed_prompt"])
```

将核心功能嵌入网站时，请阅读 [WEB_INTEGRATION_CONTEXT.md](WEB_INTEGRATION_CONTEXT.md)。其中包含推荐 API、返回字段、GPU 队列、上传安全和验收标准。

## 输出内容

默认结果保存在 `outputs/`：

```text
outputs/
|- <name>_panel.png
|- <name>_result.json
`- <name>_generated.png
```

主要结果字段包括：

- `caption`
- `detections`
- `classifications`
- `visual_context`
- `prompt` / `negative_prompt`
- `reconstruction_context`
- `generation.pixel_similarity`
- `degraded_notes`

`pixel_similarity` 是基于像素误差的辅助指标，不等同于语义或感知质量评分。

## 测试

```powershell
python -m pytest -q
```

当前测试结果：

```text
49 passed
```

## 项目结构

```text
configs/                    默认配置
scripts/                    CLI、Streamlit 与环境检查入口
src/models/                 检测、分类和 BLIP 封装
src/preprocessing/          图像预处理与特征面板
src/prompt_engineering/     双模式 Prompt 构建
src/generation/             Stable Diffusion 条件生成
src/evaluation/             评估指标
src/report/                 报告生成
tests/                      单元测试
READ.md                     完整使用手册
MODEL_SETUP.md              模型准备说明
WEB_INTEGRATION_CONTEXT.md  网站集成任务说明
```

## 文档

- [完整使用手册](READ.md)
- [模型安装说明](MODEL_SETUP.md)
- [网站集成说明](WEB_INTEGRATION_CONTEXT.md)

## License

[MIT](LICENSE)
