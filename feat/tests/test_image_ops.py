import numpy as np
from torchvision.io import read_image
from feat.transforms import Rescale
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


# TODO: write me
def test_HOGLayer_class():
    pass
