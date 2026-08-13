# VLM Vision Studio 网站集成任务说明

> 用途：将本文件交给另一个编程对话，使其能够快速理解并把本项目的核心功能嵌入现有网站。
> 项目目录：`D:\vlm-vision-studio-master`

## 1. 集成目标

请把本项目实现为网站中的一个“图片理解与参考图生成”功能，而不是重新实现模型算法。网站用户应能够：

1. 上传 PNG、JPG、JPEG、WebP 或 BMP 图片。
2. 选择“语义概括与风格扩写”或“原图精确复刻说明”。
3. 查看 BLIP 描述、正向 Prompt、负向 Prompt、目标检测、分类、主色和结构信息。
4. 可选地使用 `img2img`、`ip_adapter`、`controlnet` 或 `hybrid` 生成参考图。
5. 设置风格、去噪强度和随机种子，并查看生成结果与像素相似度。
6. 获得清晰的加载、成功、降级和错误状态。

集成时必须保持现有核心算法和输出含义不变。优先封装 Python 后端 API，再由网站前端调用，不要让浏览器直接加载模型。

## 2. 项目与 VLM 的关系

项目定位是“VLM 驱动的图像分析、Prompt 构造和受控重建工作台”。

- BLIP 是核心 VLM，负责把图片转换为自然语言描述。
- Faster R-CNN 检测物体类别、置信度和位置框。
- ResNet50 提供图像分类候选。
- OpenCV 和项目内分析代码提取颜色、边缘、尺寸、布局及空间关系。
- Prompt 模块融合上述信息，生成语义 Prompt 或复刻 Prompt。
- Stable Diffusion 1.5、IP-Adapter 和 ControlNet 负责条件图像生成，不属于图像描述 VLM 本身。

完整链路：

```text
上传图片
  -> OpenCV 特征提取
  -> Faster R-CNN 检测 + ResNet50 分类
  -> BLIP 图像描述
  -> 颜色、尺寸、位置、空间关系与风格分析
  -> 双模式 Prompt 构造
  -> 可选的 Stable Diffusion 参考图生成
  -> JSON 数据 + 面板图 + 生成图
```

## 3. 应直接复用的核心入口

核心类位于 `src/pipeline.py`：

```python
from src.pipeline import Pipeline

pipeline = Pipeline("configs/default.yaml")

pipeline.optimizer.style = "cinematic"
pipeline.generator.mode = "hybrid"
pipeline.generator.strength = 0.20
pipeline.generation_seed = 42

result = pipeline.run(
    image_path="outputs/uploads/example.png",
    save_panel=True,
    generate=True,
    prompt_mode="reconstruction",
)
```

不要通过 `subprocess` 调用 `scripts/run_demo.py`，也不要抓取终端文字。`Pipeline.run()` 才是网站后端应调用的稳定业务入口。

`scripts/run_app.py` 是现成的 Streamlit 参考界面，可用于理解控件和结果展示，但若目标网站使用 React、Vue 或其他框架，应复用业务管线而非嵌套 Streamlit。

## 4. 两种 Prompt 模式

### `semantic`

“语义概括与风格扩写”模式。融合 BLIP 描述、可靠检测/分类标签、主色和风格模板，适合创意生成。可选风格目前包括：

- `cinematic`
- `anime`
- `oil_painting`
- `photorealistic`

该模式不追求原图逐像素一致。

### `reconstruction`

“原图精确复刻说明”模式。强调画面比例、背景、颜色、几何轮廓、元素位置、尺寸、间距和留白。

- `result["prompt"]`：紧凑 Prompt，实际送入 SD 1.5，考虑了 CLIP 77-token 限制。
- `result["reconstruction_context"]["detailed_prompt"]`：完整结构说明，适合展示、编辑和审计，不应直接假定它全部进入扩散模型。

Prompt 只能描述图片，不能保存全部像素信息。需要高相似度时必须把原图同时作为 img2img、IP-Adapter 或 ControlNet 条件。

## 5. 四种生成模式

| 模式 | 用途 | 建议 strength |
| --- | --- | ---: |
| `img2img` | 轻度润色并保留整体布局 | `0.20-0.30` |
| `ip_adapter` | 加强主体外观与视觉特征参考 | `0.20-0.35` |
| `controlnet` | 保留边缘、轮廓和构图 | `0.25-0.40` |
| `hybrid` | 8GB 显存环境下兼顾结构、颜色和原图像素 | `0.15-0.30` |

`strength` 越低越接近原图，越高则模型改动越明显。精确复刻默认推荐 `reconstruction + hybrid + 0.15~0.25`。

当前 `hybrid` 的实现为 ControlNet 约束生成，再执行参考图颜色统计匹配和强度相关的原图像素融合。它没有同时常驻完整 IP-Adapter 与 ControlNet，以适配 RTX 4060 Laptop 8GB 显存。

## 6. 返回数据

`Pipeline.run()` 返回字典，主要字段如下：

| 字段 | 前端用途 |
| --- | --- |
| `caption` | BLIP 生成的图片概述 |
| `caption_fallback` | 是否使用了降级描述 |
| `prompt_mode` | 当前 Prompt 模式 |
| `prompt` | 正向 Prompt |
| `negative_prompt` | 负向 Prompt |
| `prompt_score` | 语义 Prompt 质量分；复刻模式可能为 `None` |
| `detections` | 检测类别、置信度和边界框 |
| `classifications` | Top-K 分类结果 |
| `feature_summary` | 宽、高和边缘密度等摘要 |
| `visual_context` | 主色、位置、尺寸及空间关系 |
| `reconstruction_context` | 复刻元素、详细 Prompt、风格标签和指标 |
| `panel_path` | 特征面板图片路径 |
| `generation` | 生成模式、图片路径、seed、strength、错误和相似度 |
| `degraded_notes` | 模型不可用或步骤失败的解释 |

检测、分类和生成结果中含有 dataclass 对象。对外返回 JSON 前需要递归序列化，可参考 `scripts/run_app.py` 中的 `_serialize()`。

`generation` 成功时常用属性：

```python
generation.image_path
generation.mode
generation.seed
generation.strength
generation.pixel_similarity
generation.fallback
generation.reason
```

## 7. 推荐的网站后端 API

建议由网站后端提供一个 multipart 接口：

```http
POST /api/vlm/analyze
Content-Type: multipart/form-data
```

请求字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image` | file | 必填图片 |
| `prompt_mode` | string | `semantic` 或 `reconstruction` |
| `style` | string | 语义模式风格 |
| `generate` | boolean | 是否生成图片 |
| `generation_mode` | string | 四种生成模式之一 |
| `strength` | number | `0-1`，前端建议限制为 `0.05-0.80` |
| `seed` | integer | 非负整数 |

建议响应结构：

```json
{
  "success": true,
  "data": {
    "caption": "...",
    "prompt": "...",
    "negative_prompt": "...",
    "detailed_prompt": "...",
    "detections": [],
    "classifications": [],
    "dominant_colors": [],
    "style_tags": [],
    "style_metrics": {},
    "panel_url": "/media/...",
    "generated_image_url": "/media/...",
    "pixel_similarity": 0.994,
    "degraded_notes": []
  },
  "error": null
}
```

文件系统路径不能直接暴露给浏览器。后端应将 `panel_path` 和 `generation.image_path` 转换为受控静态资源 URL，或通过鉴权下载接口返回。

## 8. 后端实现要求

1. 应在应用启动时创建单例 `Pipeline`，不要每次请求重新加载所有模型。
2. GPU 推理请求应进入队列或受互斥锁保护。8GB 显存不适合并发运行多个生成任务。
3. 网站请求不要共用可变的模式参数而无锁修改。当前 `optimizer.style`、`generator.mode`、`generator.strength` 和 `generation_seed` 都是实例状态。
4. 若需要并发，优先建立单 GPU worker 队列；分析与生成任务按顺序执行。
5. 上传文件须生成安全且唯一的服务器文件名，校验扩展名、MIME、实际图像解码结果及文件大小。
6. 不得允许用户传入任意服务器路径，也不得把模型路径暴露为请求参数。
7. 生成过程可能耗时较长。正式网站建议采用任务接口：创建任务、轮询状态、获取结果。
8. 保留 `degraded_notes` 并展示给用户，不能把模型降级伪装成完整成功。
9. `detailed_prompt` 应使用 `.get("detailed_prompt", result["prompt"])` 读取，以兼容旧结果。
10. 不要让应用自动联网下载模型；当前配置使用本地模型和 BLIP 本地缓存。

## 9. 前端交互建议

页面首屏应直接提供工作区，不需要营销型首页。建议布局：

- 左侧或顶部：图片上传与原图预览。
- 设置区：Prompt 模式、风格、是否生成、生成模式、strength、seed。
- 主操作：一个明确的“分析图片”按钮。
- 结果区：正向 Prompt、负向 Prompt，均提供复制操作。
- 复刻模式：可折叠展示完整复刻说明。
- 图片区：原图、特征面板和生成结果并排比较。
- 详情区：检测、分类、颜色、风格指标和降级信息。
- 生成阶段展示排队、模型加载、推理和完成状态，避免用户重复提交。

`pixel_similarity` 是像素 MAE 推导出的简单指标，只表示像素接近程度，不等同于语义或感知质量。前端应标为“像素相似度”，不要表述成绝对准确率。

## 10. 本地模型与运行环境

配置文件：`configs/default.yaml`

本地模型目录：

```text
models/
|- stable-diffusion-v1-5/
|- IP-Adapter/
`- sd-controlnet-canny/
```

当前验证环境：

- Python：`D:\miniconda\envs\yolov8\python.exe`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU 8GB
- CUDA PyTorch：`2.7.1+cu118`
- 单元测试：`49 passed`

检查环境和模型：

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/check_model_setup.py
```

运行现有参考界面：

```powershell
D:\miniconda\envs\yolov8\python.exe scripts/start_app.py
```

参考地址：`http://127.0.0.1:8501/`

## 11. 关键文件

| 文件 | 作用 |
| --- | --- |
| `src/pipeline.py` | 网站应调用的总业务管线 |
| `src/models/vlm_captioner.py` | BLIP 图片描述 |
| `src/models/detector.py` | Faster R-CNN 检测 |
| `src/models/classifier.py` | ResNet50 分类 |
| `src/prompt_engineering/visual_context.py` | 颜色和空间关系 |
| `src/prompt_engineering/reconstruction_prompt.py` | 精确复刻 Prompt |
| `src/prompt_engineering/prompt_optimizer.py` | 语义风格扩写 |
| `src/generation/image_generator.py` | 四种参考图生成模式 |
| `scripts/run_app.py` | Streamlit UI 与序列化参考 |
| `configs/default.yaml` | 阈值、模型路径和生成参数 |
| `README.md` | 完整使用说明 |
| `MODEL_SETUP.md` | 模型安装说明 |

## 12. 集成验收标准

1. 上传有效图片后能返回 BLIP 描述和两类 Prompt。
2. 切换 Prompt 模式后结果字段正确，复刻模式能显示 `detailed_prompt`。
3. 未开启生成时不会加载 Stable Diffusion，也不会产生生成图。
4. 四种生成模式均能通过网站选择，并正确传给 `Pipeline`。
5. `hybrid` 低 strength 能输出生成图和 `pixel_similarity`。
6. 页面能展示原图、特征面板、生成图、检测分类和降级说明。
7. 错误上传、模型缺失、CUDA 不可用和生成失败均返回可理解的错误，不导致网站进程退出。
8. 连续执行多个任务时不会因文件重名覆盖其他用户结果。
9. GPU 生成任务不会并发争抢显存。
10. 原有命令行与 Streamlit 功能仍能正常运行，现有测试继续通过。

## 13. 给后续编程对话的直接任务

请先阅读本文件及上述关键源码，然后检查目标网站的技术栈与目录结构。在不重写模型算法的前提下：

1. 将 `Pipeline` 封装为网站后端服务。
2. 建立安全的图片上传、任务执行、结果序列化和媒体访问机制。
3. 在现有网站设计语言内实现完整交互界面。
4. 对 GPU 任务增加单 worker 队列或互斥控制。
5. 保留两种 Prompt 模式、四种生成模式及所有降级信息。
6. 补充后端接口测试与关键前端流程测试。
7. 更新网站项目 README，写明模型目录、启动方式和使用方法。

在开始修改前先阅读目标网站代码，沿用其现有框架、组件、路由、状态管理和样式体系。
