import pytest
import numpy as np
from os.path import join
from feat.utils.io import (
    get_test_data_path,
    read_feat,
    read_openface,
)
from feat.utils.image_operations import registration
from feat.plotting import load_viz_model
from feat.utils.stats import softmax
from feat import Fex


def test_read_feat():
    fex = read_feat(join(get_test_data_path(), "Feat_Test.csv"))
    assert isinstance(fex, Fex)
    # A v1/legacy file reads back as the "Feat" detector with lowercase emotions
    assert fex.detector == "Feat"
    assert "happiness" in fex.emotion_columns


def test_read_feat_v2(tmp_path):
    """read_feat auto-detects Detectorv2 CSVs and restores v2 column groups."""
    import pandas as pd
    from feat.multitask import EMOTION_COLUMNS_V2

    d = pd.DataFrame(
        {
            "AU01": [0.1], "AU12": [0.9],
            **{e: [1.0 / len(EMOTION_COLUMNS_V2)] for e in EMOTION_COLUMNS_V2},
            "valence": [0.3], "arousal": [0.2],
            "gaze_pitch": [0.0], "gaze_yaw": [0.0], "gaze_angle": [0.0],
            "Pitch": [0.0], "Roll": [0.0], "Yaw": [0.0],
            "X": [0.0], "Y": [0.0], "Z": [0.0],
            "mesh_x_0": [0.5], "mesh_y_0": [0.5], "mesh_z_0": [0.0],
            "input": ["clip.mp4"], "frame": [0],
        }
    )
    path = tmp_path / "v2.csv"
    d.to_csv(path, index=False)

    fex = read_feat(str(path))
    assert isinstance(fex, Fex)
    assert fex.detector == "Detectorv2"
    # Capitalized v2 emotion names, not the lowercase v1 set
    assert list(fex.emotion_columns) == list(EMOTION_COLUMNS_V2)
    assert "Happy" in fex.emotion_columns
    # Gaze is restored (the v1 path leaves it unset)
    assert list(fex.gaze_columns) == ["gaze_pitch", "gaze_yaw", "gaze_angle"]
    assert "AU12" in fex.au_columns


def test_utils():
    sample = read_openface(join(get_test_data_path(), "OpenFace_Test.csv"))
    lm_cols = ["x_" + str(i) for i in range(0, 68)] + [
        "y_" + str(i) for i in range(0, 68)
    ]
    sample_face = np.array([sample[lm_cols].values[0]])
    registered_lm = registration(sample_face)
    assert registered_lm.shape == (1, 136)

    with pytest.raises(ValueError):
        registration(sample_face, method="badmethod")
    with pytest.raises(TypeError):
        registration(sample_face, method=np.array([1, 2, 3, 4]))
    with pytest.raises(AssertionError):
        registration([sample_face[0]])
    with pytest.raises(AssertionError):
        registration(sample_face[0])
    with pytest.raises(AssertionError):
        registration(sample_face[:, :-1])

    # Test softmax
    assert softmax(0) == 0.5
    # Test badfile.
    with pytest.raises(Exception):
        load_viz_model("badfile")


# TODO: write me
def test_set_torch_device():
    pass


def test_hf_hub_download_with_fallback_uses_v2_when_present(monkeypatch):
    """Helper returns the primary file when hf_hub_download succeeds."""
    from feat.utils import hf_hub_download_with_fallback

    calls = []

    def fake_download(repo_id, filename, cache_dir):
        calls.append(filename)
        return f"/fake/{filename}"

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_download, raising=False
    )
    result = hf_hub_download_with_fallback(
        repo_id="x/y", filename="primary.skops",
        fallback_filename="fallback.skops", cache_dir="/tmp",
    )
    assert result == "/fake/primary.skops"
    assert calls == ["primary.skops"]


def test_hf_hub_download_with_fallback_falls_back_on_404(monkeypatch):
    """Helper retries with fallback_filename if primary raises EntryNotFoundError."""
    from feat.utils import hf_hub_download_with_fallback
    from huggingface_hub.utils import EntryNotFoundError

    calls = []

    def fake_download(repo_id, filename, cache_dir):
        calls.append(filename)
        if filename == "primary.skops":
            raise EntryNotFoundError("primary.skops not found")
        return f"/fake/{filename}"

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_download, raising=False
    )
    result = hf_hub_download_with_fallback(
        repo_id="x/y", filename="primary.skops",
        fallback_filename="fallback.skops", cache_dir="/tmp",
    )
    assert result == "/fake/fallback.skops"
    assert calls == ["primary.skops", "fallback.skops"]
