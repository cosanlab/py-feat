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
