# 模型手动下载与离线配置

项目代码和 Python 依赖已经配置完成。模型权重可由你自行下载，并统一放在项目根目录的 `models/` 下。不要改变模型仓库内部的文件结构。

## 运行环境

建议先创建并激活独立的 Python 环境：

```powershell
conda create -n vlm-vision-studio python=3.9 -y
conda activate vlm-vision-studio
python -m pip install -r requirements.txt
```

执行 Stable Diffusion 生成需要 NVIDIA GPU 和 CUDA 版 PyTorch。请根据本机 CUDA 与驱动版本安装匹配的 PyTorch，并通过以下命令确认 CUDA 可用：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 需要的模型

### 1. BLIP 图像描述

下载完整仓库 `Salesforce/blip-image-captioning-base` 到：

```text
models/blip-image-captioning-base/
```

然后修改 `configs/default.yaml`：

```yaml
vlm:
  model_name: models/blip-image-captioning-base
```

目录至少应包含 `config.json`、处理器/分词器配置和模型权重文件。

### 2. torchvision 检测与分类

torchvision 默认会在用户缓存目录中管理以下两个权重文件：

```text
<TORCH_HOME>/hub/checkpoints/
```

未设置 `TORCH_HOME` 时，常见默认位置为：

```text
Windows: %USERPROFILE%\.cache\torch\hub\checkpoints\
Linux/macOS: ~/.cache/torch/hub/checkpoints/
```

文件名必须保持为：

```text
fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
resnet50-11ad3fa6.pth
```

torchvision 会自动从该缓存目录读取，不需要修改 YAML。

### 3. Stable Diffusion img2img

可使用 AtomGit 镜像下载 Diffusers 格式的完整仓库：

```powershell
git clone https://atomgit.com/hf_mirrors/bdsqlsz/stable-diffusion-v1-5.git models/stable-diffusion-v1-5
```

模型目录应为：

```text
models/stable-diffusion-v1-5/
```

配置：

```yaml
generation:
  model_name: models/stable-diffusion-v1-5
```

必须是 Diffusers 目录结构，而不是只有单个 `.ckpt` 或 `.safetensors` 文件。

### 4. IP-Adapter（可选，高相似度）

当前项目使用本地 IP-Adapter 目录，并保留其中 `models/` 子目录：

```text
models/IP-Adapter/models/ip-adapter_sd15.bin
```

同时还需要 IP-Adapter 仓库中的 CLIP 图像编码器，完整放入：

```text
models/IP-Adapter/image_encoder/
├── config.json
├── preprocessor_config.json
└── model.safetensors                 # 或 pytorch_model.bin
```

Diffusers 会用该编码器将参考图转成图像嵌入。只有 `ip-adapter_sd15.bin`
而没有 `image_encoder/` 时，`--mode ip_adapter` 无法运行。

配置：

```yaml
generation:
  mode: ip_adapter
  ip_adapter_repo: models/IP-Adapter
  ip_adapter_weight: ip-adapter_sd15.bin
  ip_adapter_scale: 0.65
```

### 5. Canny ControlNet（可选，强构图约束）

当前项目使用本地 `sd-controlnet-canny` 模型目录：

```text
models/sd-controlnet-canny/
```

配置：

```yaml
generation:
  mode: controlnet
  controlnet_model: models/sd-controlnet-canny
```

## 检查与运行

检查环境和本地模型路径（不会下载任何内容）：

```powershell
python scripts/check_model_setup.py
```

普通低强度 img2img：

```powershell
python scripts/run_demo.py --image assets/examples/sample_shapes.png --generate --mode img2img --strength 0.25
```

更强调参考图视觉特征：

```powershell
python scripts/run_demo.py --image path/to/photo.jpg --generate --mode ip_adapter --strength 0.25
```

更强调边缘与构图：

```powershell
python scripts/run_demo.py --image path/to/photo.jpg --generate --mode controlnet --strength 0.30
```

输出图像保存为 `outputs/<原文件名>_generated.png`。`strength` 越低越接近原图；建议从 `0.20` 到 `0.35` 调整。
