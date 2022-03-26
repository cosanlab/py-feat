import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
from os import remove
from feat import Fex
from feat.utils import read_openface, get_test_data_path
from feat.plotting import (
    plot_face,
    draw_lineface,
    draw_vectorfield,
    predict,
    animate_face,
)
import matplotlib
import pytest

matplotlib.use("Agg")


def assert_plot_shape(ax):
    assert ax.get_ylim() == (240.0, 50.0)
    assert ax.get_xlim() == (25.0, 172.0)


feature_length = 20
au = np.ones(feature_length)
au2 = np.ones(feature_length) * 3


def testpredict():
    landmarks = predict(au)
    assert landmarks.shape == (2, 68)
    with pytest.raises(ValueError):
        predict(au, model=[0])
    with pytest.raises(ValueError):
        predict(au[:-1])


def test_draw_lineface():
    landmarks = predict(au)
    draw_lineface(currx=landmarks[0, :], curry=landmarks[1, :])
    assert_plot_shape(plt.gca())
    plt.close()


def test_draw_vectorfield():
    draw_vectorfield(reference=predict(au), target=predict(au=au2))
    assert_plot_shape(plt.gca())
    plt.close()
    with pytest.raises(ValueError):
        draw_vectorfield(
            reference=predict(au).reshape(4, 2 * feature_length), target=predict(au=au2)
        )
    with pytest.raises(ValueError):
        draw_vectorfield(
            reference=predict(au), target=predict(au=au2).reshape(4, 2 * feature_length)
        )


def test_plot_face():
    # test plotting method
    fx = Fex(
        filename=join(get_test_data_path(), "iMotions_Test_v6.txt"),
        sampling_freq=30,
        detector="FACET",
    )
    fx = fx.read_file()
    ax = fx.plot_aus(row_n=0)
    assert_plot_shape(ax)
    plt.close()

    fx = Fex(
        filename=join(get_test_data_path(), "OpenFace_Test.csv"),
        sampling_freq=30,
        detector="OpenFace",
    )
    fx = fx.read_file()
    ax = fx.plot_aus(row_n=0)
    assert_plot_shape(ax)
    plt.close()

    fx = Fex(
        filename=join(get_test_data_path(), "sample_affectiva-api-app_output.json"),
        sampling_freq=30,
        detector="Affectiva",
    )
    fx = fx.read_file(orig_cols=False)
    ax = fx.plot_aus(row_n=0)
    assert_plot_shape(ax)
    plt.close()

    # test plot in util
    plot_face()
    assert_plot_shape(plt.gca())
    plt.close()

    plot_face(au=au, vectorfield={"reference": predict(au2)}, feature_range=(0, 1))
    assert_plot_shape(plt.gca())
    plt.close()

    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={"reference": predict(au2)})
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield=[])
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={"noreference": predict(au2)})


def test_plot_muscle():
    test_file = join(get_test_data_path(), "OpenFace_Test.csv")
    _, ax = plt.subplots(figsize=(4, 5))
    openface = read_openface(test_file)
    ax = openface.plot_aus(12, ax=ax, muscles={"all": "heatmap"}, gaze=None)
    assert_plot_shape(plt.gca())
    plt.close()


def test_plot_detections(default_detector, single_face_img):
    image_prediction = default_detector.detect_image(single_face_img)
    axes = image_prediction.plot_detections()
    assert axes[1].get_xlim() == (0.0, 1.1)
    plt.close()

    axes = image_prediction.plot_detections(muscle=True)
    assert axes[1].get_xlim() == (0.0, 1.1)
    plt.close()

    axes = image_prediction.plot_detections(pose=True)
    assert axes[1].get_xlim() == (0.0, 1.1)
    plt.close()

    image_prediction2 = image_prediction.copy()
    image_prediction2["input"] = "NO_SUCH_FILE_EXISTS"
    axes = image_prediction2.plot_detections()
    assert axes[1].get_xlim() == (0.0, 1.1)
    plt.close()


def test_animate_face():

    # Start with neutral face
    starting_aus = np.zeros(20)
    ending_aus = np.zeros(20)
    # Just animate the intensity of the first AU
    ending_aus[0] = 3

    animation = animate_face(start=starting_aus, end=ending_aus, save="test.gif")

    assert animation is not None
    assert exists("test.gif")

    # # Clean up
    remove("test.gif")

    # Test different init style
    animation = animate_face(AU=1, start=0, end=3, save="test.gif")

    assert animation is not None
    assert exists("test.gif")

    # Clean up
    remove("test.gif")
