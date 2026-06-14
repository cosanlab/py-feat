"""Tests for the VideoDataset torchcodec migration.

Locks down the contract that VideoDataset.__getitem__ produces:
- 'Image' tensor in [C, H, W] uint8 layout (matching torchvision.io.read_image
  and the prior PyAV path's downstream output)
- correct frame count via __len__
- correct metadata fields (num_frames, fps, width, height, shape)
- skip_frames returns the right strided indices
- channel order is RGB (consistent with the prior `format='rgb24'` av path)
- DataLoader workers can re-open their own decoder (regression test for the
  pickling bug where the cached _decoder deserializes into a broken state)

Replaces the implicit guarantee the PyAV implementation gave; without these,
a future change to load_frame could silently flip dim order and break
downstream detectors.
"""

import os
import pickle

import pytest
import torch
from torch.utils.data import DataLoader

from feat.data import VideoDataset
from feat.utils.io import get_test_data_path


VIDEO = os.path.join(get_test_data_path(), "single_face.mp4")


@pytest.fixture(autouse=True)
def _skip_if_video_missing():
    if not os.path.exists(VIDEO):
        pytest.skip(f"test video not found at {VIDEO}")


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


def test_channel_order_is_rgb():
    """torchcodec returns RGB by default; the prior av path used `format='rgb24'`.
    `single_face.mp4` is a face video where the red channel mean exceeds the blue
    channel mean (skin tones). This pins the channel order so a future torchcodec
    default change to BGR would be caught here."""
    ds = VideoDataset(VIDEO)
    img = ds[0]["Image"].float()
    # img is [C, H, W]; channel-wise means
    r_mean = img[0].mean().item()
    g_mean = img[1].mean().item()
    b_mean = img[2].mean().item()
    # Skin tones in a face video: R > B by a comfortable margin
    assert r_mean > b_mean, (
        f"channel 0 mean ({r_mean:.1f}) should exceed channel 2 mean ({b_mean:.1f}) "
        f"for an RGB face video; if not, channel order may have flipped to BGR"
    )
    # Sanity: G between R and B is the typical face profile
    assert b_mean < g_mean < r_mean, (
        f"unexpected channel ordering R={r_mean:.1f} G={g_mean:.1f} B={b_mean:.1f}"
    )


def test_pickle_drops_cached_decoder():
    """Critical: VideoDataset must be picklable for DataLoader multi-worker.
    The cached torchcodec decoder is a C++ object; if it survives pickle
    round-trip, the deserialized decoder is broken (stream registration
    is lost). __getstate__ must drop _decoder; __setstate__ recovers."""
    ds = VideoDataset(VIDEO)
    # Force decoder construction
    _ = ds[0]
    assert ds._decoder is not None

    blob = pickle.dumps(ds)
    ds2 = pickle.loads(blob)
    # _decoder must be None after unpickling
    assert ds2._decoder is None
    # First access must successfully reconstruct the decoder and return a frame
    sample = ds2[0]
    assert sample["Image"].shape[0] == 3


def test_dataloader_multi_worker_iterates_full_video():
    """Regression test for the multi-worker decoder pickling crash.
    `Detectorv1.detect_video(num_workers>0)` is a public path; if VideoDataset
    can't survive a fork, every multi-worker video call dies with
    'validateActiveStream … stream index=0 was not previously added'."""
    ds = VideoDataset(VIDEO)
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    total = 0
    for batch in loader:
        # batch["Image"] is collated as [B, C, H, W]
        total += batch["Image"].shape[0]
    assert total == len(ds)
