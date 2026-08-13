"""ImageProcessor 单元测试：仅依赖 numpy / OpenCV / PIL，不涉及模型下载。"""

from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.image_processor import ImageProcessor


@pytest.fixture()
def processor() -> ImageProcessor:
    return ImageProcessor()


@pytest.fixture()
def color_image() -> np.ndarray:
    """构造 100x200 三通道测试图：左半红、右半蓝。"""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :100] = (255, 0, 0)
    img[:, 100:] = (0, 0, 255)
    return img


class TestLoadSave:
    def test_load_missing_file_raises(self, processor: ImageProcessor, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            processor.load(tmp_path / "not_exist.png")

    def test_save_and_load_roundtrip(
        self, processor: ImageProcessor, color_image: np.ndarray, tmp_path: Path
    ) -> None:
        path = processor.save(color_image, tmp_path / "out" / "img.png")
        assert path.exists()
        loaded = processor.load(path)
        assert loaded.shape == color_image.shape
        # PNG 无损，允许通道序差异但内容应一致
        assert sorted(np.unique(loaded.reshape(-1, 3), axis=0).tolist()) == sorted(
            np.unique(color_image.reshape(-1, 3), axis=0).tolist()
        )


class TestPreprocess:
    def test_resize_down_scales(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        resized = processor.resize(color_image, max_side=50)
        assert max(resized.shape[:2]) == 50
        assert resized.shape[0] == 25 and resized.shape[1] == 50

    def test_resize_upscale_returns_copy(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        resized = processor.resize(color_image, max_side=1024)
        assert resized.shape == color_image.shape
        assert resized is not color_image

    def test_to_gray_shape_and_range(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        gray = processor.to_gray(color_image)
        assert gray.shape == (100, 200)
        assert gray.dtype == np.uint8
        assert gray.min() >= 0 and gray.max() <= 255

    def test_denoise_preserves_shape(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        out = processor.denoise(color_image)
        assert out.shape == color_image.shape
        assert out.dtype == np.uint8


class TestFeatures:
    def test_edges_binary(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        edges = processor.extract_edges(color_image)
        assert edges.shape == (100, 200)
        assert set(np.unique(edges)).issubset({0, 255})

    def test_edge_density_bounds(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        edges = processor.extract_edges(color_image)
        density = processor.edge_density(edges)
        assert 0.0 <= density <= 1.0

    def test_histogram_normalized(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        hist = processor.color_histogram(color_image, bins=8)
        assert set(hist) == {"channel_0", "channel_1", "channel_2"}
        for values in hist.values():
            assert len(values) == 8
            assert abs(sum(values) - 1.0) < 1e-6

    def test_histogram_rejects_non_positive_bins(
        self, processor: ImageProcessor, color_image: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="bins"):
            processor.color_histogram(color_image, bins=0)

    def test_feature_panel(self, processor: ImageProcessor, color_image: np.ndarray) -> None:
        panel, summary = processor.make_feature_panel(color_image, max_side=50)
        # 面板宽度 = 3 张子图横向拼接
        assert panel.shape[1] == 50 * 3
        assert panel.shape[0] == 25
        assert 0.0 <= summary["edge_density"] <= 1.0
        assert summary["width"] == 200 and summary["height"] == 100
