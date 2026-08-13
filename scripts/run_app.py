"""Streamlit 本地交互界面。使用 scripts/start_app.py 启动。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.pipeline import Pipeline


def _serialize(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _serialize(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline("configs/default.yaml")


def render_app() -> None:
    st.set_page_config(page_title="VLM Vision Studio", page_icon="VS", layout="wide")
    st.title("VLM Vision Studio")
    st.caption("图片理解、双模式 Prompt 与参考图生成")

    with st.sidebar:
        st.header("分析设置")
        prompt_mode_label = st.radio(
            "Prompt 模式",
            ("语义概括与风格扩写", "原图精确复刻说明"),
        )
        style = st.selectbox(
            "语义模式风格",
            ("cinematic", "anime", "oil_painting", "photorealistic"),
        )
        st.header("生成设置")
        should_generate = st.toggle("同时生成新图片", value=False)
        generation_mode = st.segmented_control(
            "生成模式",
            ("img2img", "ip_adapter", "controlnet", "hybrid"),
            default="hybrid",
        )
        strength = st.slider("改动强度", 0.05, 0.80, 0.25, 0.05)
        seed = st.number_input("随机种子", min_value=0, value=42, step=1)

    uploaded = st.file_uploader("上传输入图片", type=("png", "jpg", "jpeg", "webp", "bmp"))
    if uploaded is None:
        st.info("上传图片后选择 Prompt 模式。只需要 Prompt 时无需开启图片生成。")
        return

    upload_dir = Path("outputs/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / Path(uploaded.name).name
    input_path.write_bytes(uploaded.getbuffer())
    left, right = st.columns(2)
    with left:
        st.image(str(input_path), caption="输入图片", use_container_width=True)
    with right:
        st.markdown("**模式说明**")
        if prompt_mode_label.startswith("语义"):
            st.write("概括主体和场景，并按所选风格扩写，适合创意生成。")
        else:
            st.write("描述比例、颜色、几何、位置和留白，适合结构复刻与 ControlNet。")

    if not st.button("分析图片", type="primary", use_container_width=True):
        return

    pipeline = get_pipeline()
    pipeline.optimizer.style = style
    pipeline.generator.mode = generation_mode or "controlnet"
    pipeline.generator.strength = float(strength)
    pipeline.generation_seed = int(seed)
    prompt_mode = "semantic" if prompt_mode_label.startswith("语义") else "reconstruction"

    with st.status("正在分析图片...", expanded=True) as status:
        st.write("运行检测、分类和 BLIP 描述")
        result = pipeline.run(
            input_path,
            save_panel=True,
            generate=should_generate,
            prompt_mode=prompt_mode,
        )
        status.update(label="处理完成", state="complete", expanded=False)

    st.subheader("Prompt")
    prompt_col, negative_col = st.columns(2)
    with prompt_col:
        st.text_area("正向 Prompt", result["prompt"], height=240)
    with negative_col:
        st.text_area("负向 Prompt", result["negative_prompt"], height=240)
    if prompt_mode == "reconstruction":
        with st.expander("查看完整复刻说明（包含百分比坐标）"):
            st.text_area(
                "详细复刻说明",
                result["reconstruction_context"].get("detailed_prompt", result["prompt"]),
                height=260,
            )

    preview_col, generated_col = st.columns(2)
    with preview_col:
        st.subheader("特征面板")
        if result["panel_path"]:
            st.image(result["panel_path"], use_container_width=True)
    with generated_col:
        st.subheader("生成结果")
        generation = result["generation"]
        if generation and generation.image_path:
            st.image(generation.image_path, use_container_width=True)
        elif should_generate:
            st.error(generation.reason if generation else "生成未完成")
        else:
            st.caption("未开启图片生成")

    summary = {
        "caption": result["caption"],
        "prompt_mode": result["prompt_mode"],
        "dominant_colors": result["visual_context"]["dominant_colors"],
        "detections": _serialize(result["detections"]),
        "classifications": _serialize(result["classifications"]),
        "reconstruction_elements": result["reconstruction_context"]["elements"],
        "style_tags": result["reconstruction_context"]["style_tags"],
        "style_metrics": result["reconstruction_context"]["style_metrics"],
        "degraded_notes": result["degraded_notes"],
    }
    with st.expander("查看识别详情"):
        st.code(json.dumps(summary, ensure_ascii=False, indent=2), language="json")


render_app()
