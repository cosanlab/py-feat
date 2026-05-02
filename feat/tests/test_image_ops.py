import numpy as np
import pytest
import torch
from skimage.feature import hog as skimage_hog
from torchvision.io import read_image
from feat.transforms import Rescale
from feat.utils.image_operations import HOGLayer
from torchvision.transforms import Compose
from feat.data import ImageDataset


def test_rescale_single_image(single_face_img):
    img = read_image(single_face_img)

    # Test Int
    for scale in [0.5, 1.0, 2]:
        output_size = int(img.shape[-1] * scale)

        transform = Compose(
            [Rescale(output_size, preserve_aspect_ratio=False, padding=False)]
        )
        transformed_img = transform(img)

        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        transform = Compose(
            [Rescale(output_size, preserve_aspect_ratio=True, padding=False)]
        )
        transformed_img = transform(img)

        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        transform = Compose(
            [Rescale(output_size, preserve_aspect_ratio=True, padding=True)]
        )
        transformed_img = transform(img)

        assert transformed_img["Image"].shape[-1] == output_size
        assert transformed_img["Image"].shape[-2] == output_size
        assert transformed_img["Scale"] == scale
        assert transformed_img["Padding"]["Top"] + transformed_img["Padding"][
            "Bottom"
        ] == (output_size - img.shape[-2] * scale)
        assert (
            transformed_img["Padding"]["Left"] + transformed_img["Padding"]["Right"] == 0
        )

    # Test Tuple
    for scale in [0.5, 1.0, 2]:
        output_size = tuple((np.array(img.shape[1:]) * scale).astype(int))

        transform = Compose(
            [Rescale(output_size, preserve_aspect_ratio=False, padding=False)]
        )
        transformed_img = transform(img)

        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        transform = Compose(
            [Rescale(output_size, preserve_aspect_ratio=True, padding=False)]
        )
        transformed_img = transform(img)

        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        output_size = (600 * scale, img.shape[-1] * scale)
        transform = Compose(
            [Rescale(output_size, preserve_aspect_ratio=True, padding=True)]
        )
        transformed_img = transform(img)
        assert transformed_img["Image"].shape[1] == output_size[0]
        assert transformed_img["Image"].shape[2] == output_size[1]
        assert transformed_img["Scale"] == scale
        assert transformed_img["Padding"]["Top"] + transformed_img["Padding"][
            "Bottom"
        ] == (600 * scale - img.shape[1] * scale)
        assert (
            transformed_img["Padding"]["Left"] + transformed_img["Padding"]["Right"] == 0
        )


def test_imagedataset(single_face_img):
    n_img = 10
    image_file_list = [single_face_img] * n_img

    img_data = ImageDataset(
        image_file_list, output_size=None, preserve_aspect_ratio=False, padding=False
    )
    assert len(img_data) == n_img
    img = img_data[0]["Image"]

    # Test Int
    for scale in [0.5, 1.0, 2]:
        output_size = int(img.shape[-1] * scale)

        img_data = ImageDataset(
            image_file_list,
            output_size=output_size,
            preserve_aspect_ratio=False,
            padding=False,
        )
        assert len(img_data) == n_img
        transformed_img = img_data[0]
        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        img_data = ImageDataset(
            image_file_list,
            output_size=output_size,
            preserve_aspect_ratio=True,
            padding=False,
        )
        assert len(img_data) == n_img
        transformed_img = img_data[0]
        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        img_data = ImageDataset(
            image_file_list,
            output_size=output_size,
            preserve_aspect_ratio=False,
            padding=True,
        )
        assert len(img_data) == n_img
        transformed_img = img_data[0]
        assert transformed_img["Image"].shape[-1] == output_size
        assert transformed_img["Image"].shape[-2] == output_size
        assert transformed_img["Scale"] == scale
        assert transformed_img["Padding"]["Top"] + transformed_img["Padding"][
            "Bottom"
        ] == (output_size - img.shape[-2] * scale)
        assert (
            transformed_img["Padding"]["Left"] + transformed_img["Padding"]["Right"] == 0
        )

    # Test Tuple
    for scale in [0.5, 1.0, 2]:
        output_size = tuple((np.array(img.shape[1:]) * scale).astype(int))

        img_data = ImageDataset(
            image_file_list,
            output_size=output_size,
            preserve_aspect_ratio=False,
            padding=False,
        )
        assert len(img_data) == n_img
        transformed_img = img_data[0]
        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        img_data = ImageDataset(
            image_file_list,
            output_size=output_size,
            preserve_aspect_ratio=True,
            padding=False,
        )
        assert len(img_data) == n_img
        transformed_img = img_data[0]
        assert transformed_img["Image"].shape[-1] == img.shape[-1] * scale
        assert transformed_img["Image"].shape[-2] == img.shape[-2] * scale
        assert transformed_img["Scale"] == scale
        for x in ["Left", "Top", "Right", "Bottom"]:
            assert transformed_img["Padding"][x] == 0

        output_size = (600 * scale, img.shape[-1] * scale)
        img_data = ImageDataset(
            image_file_list,
            output_size=output_size,
            preserve_aspect_ratio=True,
            padding=True,
        )
        assert len(img_data) == n_img
        transformed_img = img_data[0]
        assert transformed_img["Image"].shape[1] == output_size[0]
        assert transformed_img["Image"].shape[2] == output_size[1]
        assert transformed_img["Scale"] == scale
        assert transformed_img["Padding"]["Top"] + transformed_img["Padding"][
            "Bottom"
        ] == (600 * scale - img.shape[1] * scale)
        assert (
            transformed_img["Padding"]["Left"] + transformed_img["Padding"]["Right"] == 0
        )


# TODO: write me
def test_registration():
    pass


# TODO: write me
def test_extract_face_from_bbox():
    pass


# TODO: write me
def test_extract_face_from_landmarks():
    pass


# TODO: write me
def test_convert68to49():
    pass


# TODO: write me
def test_align_face():
    pass


# TODO: write me
def test_BBox_class():
    pass


# TODO: write me
def test_convert_image_to_tensor():
    pass


# TODO: write me
def test_convert_color_vector_to_tensor():
    pass


# TODO: write me
def test_mask_image():
    pass


# TODO: write me
def test_convert_to_euler():
    pass


# TODO: write me
def test_py_cpu_nms():
    pass


# TODO: write me
def test_decode():
    pass


def _skimage_reference(img_chw_float, *, orientations, pixels_per_cell, cells_per_block, block_norm):
    """skimage.feature.hog operates on HxWxC float images; mirror the call
    extract_hog_features makes (channel_axis=-1, default L2-Hys block norm)."""
    img_hwc = img_chw_float.permute(1, 2, 0).cpu().numpy()
    return skimage_hog(
        img_hwc,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        channel_axis=-1,
        feature_vector=True,
    )


@pytest.mark.parametrize("block_norm", ["L1", "L1-sqrt", "L2", "L2-Hys"])
def test_HOGLayer_matches_skimage(block_norm):
    """HOGLayer must produce the same feature vector as skimage.feature.hog
    for the parameter set used by extract_hog_features (orientations=8,
    pixels_per_cell=(8,8), cells_per_block=(2,2), channel_axis=-1)."""
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 112, 112)

    expected = np.stack(
        [
            _skimage_reference(
                batch[i],
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm=block_norm,
            )
            for i in range(batch.shape[0])
        ]
    )

    layer = HOGLayer(
        orientations=8,
        pixels_per_cell=8,
        cells_per_block=2,
        block_normalization=block_norm,
        feature_vector=True,
        device="cpu",
    )
    actual = layer(batch).cpu().numpy()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_HOGLayer_handles_3_channel_input():
    """Calling HOGLayer with an RGB face crop must not raise. (The current
    Sobel weight has shape [2, 1, 3, 3] with groups=1, which expects a
    1-channel tensor and errors on RGB inputs.)"""
    torch.manual_seed(0)
    batch = torch.rand(1, 3, 112, 112)
    layer = HOGLayer(
        orientations=8,
        pixels_per_cell=8,
        cells_per_block=2,
        block_normalization="L2-Hys",
        feature_vector=True,
        device="cpu",
    )
    out = layer(batch)
    assert out.ndim == 2
    assert out.shape[0] == 1
    assert torch.isfinite(out).all()


def test_HOGLayer_matches_skimage_grayscale():
    """Single-channel (grayscale) input must also match skimage."""
    torch.manual_seed(0)
    batch = torch.rand(2, 1, 112, 112)
    expected = np.stack(
        [
            skimage_hog(
                batch[i, 0].cpu().numpy(),
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                visualize=False,
                feature_vector=True,
            )
            for i in range(batch.shape[0])
        ]
    )
    layer = HOGLayer(
        orientations=8,
        pixels_per_cell=8,
        cells_per_block=2,
        block_normalization="L2-Hys",
        feature_vector=True,
        device="cpu",
    )
    actual = layer(batch).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_HOGLayer_feature_vector_false_returns_block_grid():
    """With feature_vector=False the layer should return the unflattened
    block grid in skimage's layout:
    [N, n_blocks_row, n_blocks_col, b_row, b_col, orientations]."""
    torch.manual_seed(0)
    batch = torch.rand(1, 3, 112, 112)
    layer = HOGLayer(
        orientations=8,
        pixels_per_cell=8,
        cells_per_block=2,
        block_normalization="L2-Hys",
        feature_vector=False,
        device="cpu",
    )
    out = layer(batch)
    # 112 / 8 = 14 cells per side; 14 - 2 + 1 = 13 blocks per side.
    # skimage's `normalized_blocks` shape:
    # (n_blocks_row, n_blocks_col, b_row, b_col, orientations).
    assert out.shape == (1, 13, 13, 2, 2, 8)
    assert torch.isfinite(out).all()
