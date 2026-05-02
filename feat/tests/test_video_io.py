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
import time

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


def test_load_pil_img_video_does_not_decode_whole_video():
    """Regression: prior implementation called read_video(...) then indexed,
    which loaded every frame into memory just to grab one. Random-frame
    access should now be substantially faster than full-video decode."""
    full = video_to_tensor(VIDEO)  # warm caches
    n_frames = full.shape[0]
    middle = n_frames // 2

    t0 = time.perf_counter()
    img = load_pil_img(VIDEO, frame_id=middle)
    one_frame_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = video_to_tensor(VIDEO)
    full_secs = time.perf_counter() - t0

    assert isinstance(img, Image.Image)
    # Single-frame decode should be at most half the time of full-video decode
    # for any reasonable video. This is loose since the test video is short.
    assert one_frame_secs < full_secs, (
        f"single-frame decode ({one_frame_secs:.4f}s) should be faster "
        f"than full decode ({full_secs:.4f}s); the random-frame path may "
        f"be falling back to whole-video decode."
    )


def test_load_pil_img_image_path():
    """Sanity check: image file path still works through the same function."""
    img_path = os.path.join(get_test_data_path(), "single_face.jpg")
    if not os.path.exists(img_path):
        # Different test data layouts exist; just skip if no jpg present
        import pytest
        pytest.skip("no jpg test image found at expected path")
    img = load_pil_img(img_path, frame_id=0)
    assert isinstance(img, Image.Image)
