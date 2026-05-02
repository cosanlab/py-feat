"""Tests for the VideoDataset torchcodec migration.

Locks down the contract that VideoDataset.__getitem__ produces:
- 'Image' tensor in [C, H, W] uint8 layout (matching torchvision.io.read_image
  and the prior PyAV path's downstream output)
- correct frame count via __len__
- correct metadata fields (num_frames, fps, width, height, shape)
- skip_frames returns the right strided indices

Replaces the implicit guarantee the PyAV implementation gave; without these,
a future change to load_frame could silently flip dim order and break
downstream detectors.
"""

import os

import torch

from feat.data import VideoDataset
from feat.utils.io import get_test_data_path


VIDEO = os.path.join(get_test_data_path(), "single_face.mp4")


def test_metadata_fields_present():
    ds = VideoDataset(VIDEO)
    md = ds.metadata
    assert md["num_frames"] > 0
    assert md["fps"] > 0
    assert md["width"] > 0
    assert md["height"] > 0
    assert md["shape"] == (md["height"], md["width"])


def test_len_matches_num_frames_no_skip():
    ds = VideoDataset(VIDEO)
    assert len(ds) == ds.metadata["num_frames"]


def test_len_with_skip_frames():
    ds_full = VideoDataset(VIDEO)
    skip = 4
    ds_skip = VideoDataset(VIDEO, skip_frames=skip)
    expected = len(range(0, ds_full.metadata["num_frames"], skip))
    assert len(ds_skip) == expected


def test_getitem_returns_chw_uint8_tensor():
    ds = VideoDataset(VIDEO)
    sample = ds[0]
    img = sample["Image"]
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.uint8
    assert img.ndim == 3
    assert img.shape[0] == 3
    assert img.shape[1] == ds.metadata["height"]
    assert img.shape[2] == ds.metadata["width"]


def test_getitem_frame_idx_respects_skip():
    skip = 3
    ds = VideoDataset(VIDEO, skip_frames=skip)
    assert ds[0]["Frame"] == 0
    assert ds[1]["Frame"] == skip
    assert ds[2]["Frame"] == 2 * skip


def test_getitem_with_output_size_preserves_chw_shape():
    """When output_size is set, Rescale is applied. The result must still
    be a [C, H, W] tensor on the Image key (downstream detectors require this)."""
    ds = VideoDataset(VIDEO, output_size=128)
    sample = ds[0]
    img = sample["Image"]
    assert img.ndim == 3
    assert img.shape[0] == 3
