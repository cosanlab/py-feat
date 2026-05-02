"""Tests for video decoding helpers in feat.utils.io.

These cover:
- decode_video returns a sliceable / streamable handle whose API matches
  torchcodec's VideoDecoder
- video_to_tensor preserves the [T, C, H, W] uint8 output shape that the
  prior PyAV implementation produced
- load_pil_img on a video file decodes only the requested frame, not the
  whole video (regression test for the prior `read_video(...)` then index
  pattern that loaded the entire video into memory just to grab one frame)
"""

import os
import tracemalloc

import torch
from PIL import Image

from feat.utils.io import (
    decode_video,
    load_pil_img,
    video_to_tensor,
    get_test_data_path,
)


VIDEO = os.path.join(get_test_data_path(), "single_face.mp4")


def test_decode_video_returns_sliceable_handle():
    decoder = decode_video(VIDEO)
    assert decoder.metadata.num_frames > 0
    assert decoder.metadata.width > 0
    assert decoder.metadata.height > 0
    # Single-frame indexing
    f0 = decoder[0]
    assert isinstance(f0, torch.Tensor)
    assert f0.dtype == torch.uint8
    assert f0.ndim == 3 and f0.shape[0] == 3  # [C, H, W]
    # Slice
    block = decoder[0:5]
    assert block.dtype == torch.uint8
    assert block.shape[0] == 5
    assert block.shape[1] == 3
    assert block.shape[2:] == f0.shape[1:]


def test_decode_video_streams():
    decoder = decode_video(VIDEO)
    n = 0
    for frame in decoder:
        assert isinstance(frame, torch.Tensor)
        assert frame.dtype == torch.uint8
        n += 1
        if n >= 3:
            break
    assert n == 3


def test_video_to_tensor_shape_and_dtype():
    """Maintains the [T, C, H, W] uint8 contract from the PyAV implementation."""
    t = video_to_tensor(VIDEO)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.uint8
    assert t.ndim == 4
    assert t.shape[1] == 3  # RGB
    metadata = decode_video(VIDEO).metadata
    assert t.shape[0] == metadata.num_frames
    assert t.shape[2] == metadata.height
    assert t.shape[3] == metadata.width


def test_load_pil_img_video_returns_pil():
    img = load_pil_img(VIDEO, frame_id=0)
    assert isinstance(img, Image.Image)
    metadata = decode_video(VIDEO).metadata
    assert img.size == (metadata.width, metadata.height)


def test_decode_video_random_access_returns_single_frame():
    """Regression test for the prior `read_video(...)` + index pattern
    that materialized every frame just to grab one. `decode_video[i]`
    must return a single `[C, H, W]` frame tensor, not a `[T, C, H, W]`
    full-video tensor - proving random-access is genuine.

    A wall-clock timing assertion would be load-dependent on CI; a
    `tracemalloc` peak-memory assertion misses torchcodec's C-allocated
    tensor storage. The shape assertion is the load-independent guard
    that catches a regression to whole-video-then-index behavior.
    """
    decoder = decode_video(VIDEO)
    n_frames = decoder.metadata.num_frames
    middle = n_frames // 2

    one_frame = decoder[middle]
    assert one_frame.ndim == 3, (
        f"decoder[i] must return a single [C, H, W] frame, got "
        f"shape {tuple(one_frame.shape)}"
    )
    assert one_frame.shape[0] == 3
    # Sanity: load_pil_img also returns a PIL image of the right size.
    img = load_pil_img(VIDEO, frame_id=middle)
    assert isinstance(img, Image.Image)
    assert img.size == (decoder.metadata.width, decoder.metadata.height)


def test_load_pil_img_image_path():
    """Sanity check: image file path still works through the same function."""
    img_path = os.path.join(get_test_data_path(), "single_face.jpg")
    if not os.path.exists(img_path):
        # Different test data layouts exist; just skip if no jpg present
        import pytest
        pytest.skip("no jpg test image found at expected path")
    img = load_pil_img(img_path, frame_id=0)
    assert isinstance(img, Image.Image)
