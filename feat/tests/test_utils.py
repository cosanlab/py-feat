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
