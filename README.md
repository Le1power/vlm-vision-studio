# VLM Vision Studio

基于视觉语言模型与扩散模型的图像理解、Prompt 优化和参考图重绘流水线。

项目可以从输入图像中提取目标、分类、描述、主色和空间关系，生成结构化英文 Prompt，并通过 Stable Diffusion 1.5、IP-Adapter 或 Canny ControlNet 生成与原图相似的新图。

## 功能概览

- OpenCV 图像读取、缩放、灰度、边缘和颜色特征分析。
- Faster R-CNN COCO 目标检测。
- ResNet50 ImageNet Top-K 图像分类。
- BLIP 真实图像描述与不可用时的模板降级。
- 主色、目标尺寸、画面位置及目标间空间关系提取。
- cinematic、anime、oil_painting、photorealistic Prompt 模板。
- Prompt 长度、覆盖度、具体性和多样性评分与迭代优化。
- Stable Diffusion 1.5 img2img 参考图重绘。
- IP-Adapter 图像嵌入约束。
- Canny ControlNet 边缘和构图约束。
- 批量评估、JSON 结果和 Markdown 报告。
- 模型懒加载、离线运行和明确的降级说明。

## 处理流程

```text
输入图像
  -> OpenCV 预处理与特征提取
  -> Faster R-CNN 检测 + ResNet50 分类
  -> BLIP 图像描述
  -> 主色、尺寸、位置、空间关系分析
  -> Prompt 构建、评分与优化
  -> Stable Diffusion img2img
       |- 普通 img2img
       |- IP-Adapter 图像嵌入
       `- Canny ControlNet 边缘约束
  -> PNG / JSON / Markdown 报告
```

## 当前验证环境

本项目已在以下本地环境中验证：

| 项目 | 状态 |
| --- | --- |
| Python | `D:\miniconda\envs\yolov8\python.exe` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU 8GB |
| CUDA | 可用，PyTorch `2.7.1+cu118` |
| torchvision | `0.22.1+cu118` |
| Faster R-CNN | 已就绪 |
| ResNet50 | 已就绪 |
| BLIP | 已就绪 |
| Stable Diffusion 1.5 | 已就绪 |
| IP-Adapter SD 1.5 | 已就绪 |
| Canny ControlNet | 已就绪 |
| 单元测试 | `47 passed` |

检查当前机器的依赖、CUDA 和模型状态：

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/check_model_setup.py
```

该脚本不会主动下载模型。

## 安装

建议直接使用已经配置好的 CUDA 环境：

```powershell
conda activate yolov8
python -m pip install -r requirements.txt
```

也可以始终使用解释器完整路径：

```powershell
D:\miniconda\envs\yolov8\python.exe -m pip install -r requirements.txt
```

> Windows 直接执行 `pip install torch` 可能安装 CPU 版 PyTorch。需要根据显卡驱动安装对应 CUDA 版本，并用 `torch.cuda.is_available()` 确认 GPU 是否真正可用。

## 模型目录

当前项目使用以下本地模型结构：

```text
models/
|- stable-diffusion-v1-5/
|  |- model_index.json
|  |- scheduler/
|  |- text_encoder/
|  |- tokenizer/
|  |- unet/
|  `- vae/
|- IP-Adapter/
|  |- image_encoder/
|  |  |- config.json
|  |  `- pytorch_model.bin
|  `- models/
|     `- ip-adapter_sd15.bin
`- sd-controlnet-canny/
   |- config.json
   `- diffusion_pytorch_model.safetensors
```

torchvision 权重位于用户缓存：

```text
C:\Users\25947\.cache\torch\hub\checkpoints\
|- fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
`- resnet50-11ad3fa6.pth
```

BLIP 当前位于 Hugging Face 本地缓存。详细的手动下载和离线配置见 [MODEL_SETUP.md](MODEL_SETUP.md)。

## 快速开始

### 图形界面（推荐）

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/start_app.py
```

浏览器会自动打开：

```text
http://127.0.0.1:8501
```

界面操作顺序：

1. 上传输入图片。
2. 选择“语义概括与风格扩写”或“原图精确复刻说明”。
3. 只需要 Prompt 时不要勾选“同时生成新图片”。
4. 需要生成图片时选择 img2img、IP-Adapter 或 ControlNet，并设置改动强度。
5. 点击“分析图片”，复制正向和负向 Prompt，或查看生成结果。

两种 Prompt 模式：

- **语义概括与风格扩写**：使用 BLIP、可靠的检测/分类结果和风格模板，适合创意生成。
- **原图精确复刻说明**：强调画面比例、背景、主色、几何轮廓、元素坐标、尺寸、间距和留白，不主动加入电影光效等风格词；同时分析照片/插画/扁平矢量媒介、饱和度、对比度、边缘、纹理和光照。

### 1. 生成示例图

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/make_sample_images.py
```

### 2. 仅执行图像理解和 Prompt 生成

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py `
  --image assets/examples/sample_shapes.png
```

精确复刻说明模式：

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py `
  --image assets/examples/sample_shapes.png `
  --prompt-mode reconstruction `
  --no-panel
```

不传 `--image` 时进入交互模式：

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py
```

### 3. 普通 img2img

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py `
  --image path/to/image.jpg `
  --generate `
  --mode img2img `
  --strength 0.25 `
  --seed 42
```

特点：

- 直接将原图作为 Stable Diffusion 初始图像。
- 速度和显存占用相对较低。
- 适合轻度润色、风格调整和保留整体布局。

### 4. IP-Adapter

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py `
  --image path/to/image.jpg `
  --generate `
  --mode ip_adapter `
  --strength 0.25 `
  --seed 42
```

特点：

- 同时使用原图像素和 CLIP 图像嵌入。
- 通常比纯 Prompt 更容易保持主体视觉特征。
- `generation.ip_adapter_scale` 越高，参考图影响越强。

### 5. Canny ControlNet

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py `
  --image path/to/image.jpg `
  --generate `
  --mode controlnet `
  --strength 0.30 `
  --seed 42
```

特点：

- 自动提取输入图像的 Canny 边缘。
- 强调轮廓、布局和空间构图。
- 适合建筑、产品、标志、线稿和主体位置需要稳定的场景。

## 生成模式选择

| 需求 | 推荐模式 | 建议 strength |
| --- | --- | ---: |
| 轻度润色，尽量保留原图 | `img2img` | `0.20-0.30` |
| 保留主体外观和整体视觉特征 | `ip_adapter` | `0.20-0.35` |
| 保留轮廓、边缘和构图 | `controlnet` | `0.25-0.40` |
| 8GB 显存下兼顾原图色调与轮廓构图 | `hybrid` | `0.15-0.30` |
| 更明显的风格变化 | 任意模式 | `0.40-0.60` |

`strength` 越低越接近原图，越高则模型自由发挥越明显。完全复刻原图并不是扩散模型的目标，即使使用 IP-Adapter 或 ControlNet 也可能改变文字、细节和纹理。

`hybrid` 是当前推荐的高保真模式：先以 ControlNet 约束轮廓和构图，再将生成结果匹配回原图的颜色统计，并按 strength 自适应融合部分原图像素。该实现针对 RTX 4060 8GB 优化，避免同时驻留 ControlNet 与大型 IP-Adapter 图像编码器造成内存不足。

精确模式会产生两份说明：

- `prompt`：控制在 SD 1.5 CLIP 的 77-token 限制内，实际送入扩散模型。
- `reconstruction_context.detailed_prompt`：保留完整元素尺寸和百分比坐标，供界面查看、人工编辑和结果审计。

## 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--image PATH` | 输入图像路径；省略时进入交互模式 |
| `--config PATH` | YAML 配置路径，默认 `configs/default.yaml` |
| `--no-panel` | 不保存预处理特征面板 |
| `--generate` | 执行参考图生成；未指定时只进行理解与 Prompt 生成 |
| `--prompt-mode` | `semantic` 语义扩写或 `reconstruction` 精确复刻说明 |
| `--mode` | `img2img`、`ip_adapter`、`controlnet` 或 `hybrid` |
| `--strength` | 去噪强度，范围 `0-1` |
| `--seed` | 随机种子，用于复现结果 |
| `--yes` | 交互模式中跳过确认 |

查看完整帮助：

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py --help
```

## 输出文件

默认输出目录为 `outputs/`：

```text
outputs/
|- <name>_panel.png       原图、灰度图和边缘图面板
|- <name>_result.json     完整结构化结果
`- <name>_generated.png  img2img/IP-Adapter/ControlNet 生成图
```

JSON 主要字段：

| 字段 | 含义 |
| --- | --- |
| `feature_summary` | 图像尺寸和边缘密度 |
| `detections` | Faster R-CNN 检测框、类别和置信度 |
| `classifications` | ResNet50 Top-K 分类结果 |
| `caption` | BLIP 图像描述 |
| `caption_fallback` | 是否使用了降级描述 |
| `visual_context` | 主色、目标位置、尺寸和空间关系 |
| `prompt` | 优化后的正向 Prompt |
| `negative_prompt` | 负向 Prompt |
| `prompt_score` | Prompt 质量评分 |
| `prompt_history` | Prompt 每轮优化记录 |
| `generation` | 模式、输出路径、seed、strength 和错误信息 |
| `degraded_notes` | 模型缺失或运行失败说明 |

## 批量评估

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/run_evaluation.py `
  --image-dir assets/examples `
  --report reports/test_report.md
```

评估内容包括：

- 平均检测目标数和置信度。
- 描述平均词数和 TTR 词汇丰富度。
- 描述质量合格率。
- BLIP 降级率。
- 失败样本数量及错误信息。
- Markdown 汇总报告。

批量评估默认不执行 Stable Diffusion 生成，避免不必要的显存和时间开销。

## 配置说明

默认配置位于 [configs/default.yaml](configs/default.yaml)。用户配置会和内置默认值递归合并，只需覆盖需要修改的字段。

### 检测与分类

```yaml
detection:
  score_threshold: 0.5
  max_detections: 20
  max_side: 1024

classification:
  top_k: 5
  prompt_min_score: 0.2
```

### BLIP 与 Prompt

```yaml
vlm:
  model_name: Salesforce/blip-image-captioning-base
  max_new_tokens: 30
  fallback_enabled: true
  local_files_only: true

prompt:
  language: en
  default_style: cinematic
  optimize_rounds: 2
  negative_prompt: true
```

支持的 Prompt 风格：

- `cinematic`
- `anime`
- `oil_painting`
- `photorealistic`

### 图像生成

```yaml
generation:
  enabled: false
  model_name: models/stable-diffusion-v1-5
  mode: img2img
  strength: 0.28
  guidance_scale: 7.0
  steps: 25
  seed: 42
  ip_adapter_repo: models/IP-Adapter
  ip_adapter_weight: ip-adapter_sd15.bin
  ip_adapter_scale: 0.65
  controlnet_model: models/sd-controlnet-canny
```

重要参数：

- `enabled`：设为 `true` 后，Python API 默认执行生成；命令行也可通过 `--generate` 单次开启。
- `mode`：默认生成模式。
- `strength`：原图改动幅度。
- `guidance_scale`：Prompt 对结果的约束强度。
- `steps`：扩散推理步数；更多步数通常更慢，不保证一定更好。
- `seed`：相同模型、输入和参数下用于复现结果。
- `ip_adapter_scale`：IP-Adapter 参考图影响强度。
- `classification.prompt_min_score`：低于该置信度的分类不会进入语义 Prompt，减少误判污染。

## Python API

仅进行图像理解：

```python
from src.pipeline import Pipeline

pipeline = Pipeline("configs/default.yaml")
result = pipeline.run("assets/examples/sample_shapes.png", generate=False)

print(result["caption"])
print(result["visual_context"])
print(result["prompt"])
```

执行配置中指定的生成模式：

```python
from src.pipeline import Pipeline

pipeline = Pipeline("configs/default.yaml")
result = pipeline.run("path/to/image.jpg", generate=True)

generation = result["generation"]
print(generation.image_path if generation else None)
```

## 降级行为

- Faster R-CNN 或 ResNet50 不可用时返回空结果，并写入 `degraded_notes`。
- BLIP 不可用时根据检测和分类结果生成英文模板描述。
- `vlm.local_files_only: true` 时 BLIP 只读取本地缓存，不会在运行期间自动补下载其他权重格式。
- Stable Diffusion、IP-Adapter 或 ControlNet 加载失败时，不影响前面的图像理解和 Prompt 输出。
- 单张坏图不会中断整个批量评估。
- 所有重模型均在首次使用时懒加载。

## 测试

```powershell
D:\miniconda\envs\yolov8\python.exe -m pytest tests -q
D:\miniconda\envs\yolov8\python.exe -m compileall -q src scripts
```

当前验证结果：

```text
47 passed
```

## 常见问题

### CUDA 不可用

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果版本包含 `+cpu` 或返回 `False`，说明当前解释器安装的是 CPU 版 PyTorch，或者使用了错误的 Python 环境。

### IP-Adapter 找不到图像编码器

确认以下两部分同时存在：

```text
models/IP-Adapter/models/ip-adapter_sd15.bin
models/IP-Adapter/image_encoder/config.json
models/IP-Adapter/image_encoder/pytorch_model.bin
```

### ControlNet 无法加载

确认目录至少包含：

```text
models/sd-controlnet-canny/config.json
models/sd-controlnet-canny/diffusion_pytorch_model.safetensors
```

### 生成图与原图差异过大

- 将 `strength` 调低到 `0.20-0.30`。
- 构图优先使用 `controlnet`。
- 主体外观优先使用 `ip_adapter`。
- 固定 `seed` 后再比较参数变化。
- Prompt 中避免加入与原图明显冲突的风格和场景描述。

### 显存不足

- 减小输入图像尺寸。
- 降低 `steps`。
- 关闭其他占用 GPU 的程序。
- 优先使用普通 `img2img`。
- 项目已启用 attention slicing、VAE slicing 和 CPU offload，以适配 8GB 显存。

## 项目结构

```text
configs/default.yaml                  默认配置
models/                               本地模型目录
src/pipeline.py                       端到端流程编排
src/preprocessing/image_processor.py  图像预处理与特征面板
src/models/                           检测、分类和 BLIP 描述
src/prompt_engineering/               Prompt 与视觉上下文分析
src/generation/                       img2img、IP-Adapter、ControlNet
src/evaluation/                       指标与批量评估
src/report/                           Markdown 报告生成
scripts/run_demo.py                   单图理解和生成入口
scripts/run_app.py                    Streamlit 界面页面
scripts/start_app.py                  一条命令启动本地界面
scripts/run_evaluation.py             批量评估入口
scripts/check_model_setup.py           离线环境与模型检查
tests/                                单元测试
```

## English Summary

VLM Vision Studio is a modular image-understanding and reference-image generation pipeline. It combines Faster R-CNN, ResNet50, BLIP, prompt optimization, Stable Diffusion 1.5 img2img, IP-Adapter, and Canny ControlNet. All heavy models are loaded lazily and can run from local model directories without network access.

```powershell
# Check environment and models
D:\miniconda\envs\yolov8\python.exe scripts/check_model_setup.py

# Plain img2img
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py --image path/to/image.jpg --generate --mode img2img --strength 0.25

# IP-Adapter
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py --image path/to/image.jpg --generate --mode ip_adapter --strength 0.25

# Canny ControlNet
D:\miniconda\envs\yolov8\python.exe scripts/run_demo.py --image path/to/image.jpg --generate --mode controlnet --strength 0.30
```

## License

[MIT](LICENSE)
