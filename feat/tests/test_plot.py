import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from os import remove
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


def test_predict(au):
    landmarks = predict(au)
    assert landmarks.shape == (2, 68)
    with pytest.raises(ValueError):
        predict(au, model=[0])
    with pytest.raises(ValueError):
        predict(au[:-1])


def test_draw_lineface(au):
    landmarks = predict(au)
    _ = draw_lineface(currx=landmarks[0, :], curry=landmarks[1, :])
    assert_plot_shape(plt.gca())
    plt.close()


def test_draw_vectorfield(au, au2):
    _ = draw_vectorfield(reference=predict(au), target=predict(au=au2))
    assert_plot_shape(plt.gca())
    plt.close()
    with pytest.raises(ValueError):
        _ = draw_vectorfield(
            reference=predict(au).reshape(4, 2 * 20), target=predict(au=au2)
        )
    with pytest.raises(ValueError):
        _ = draw_vectorfield(
            reference=predict(au), target=predict(au=au2).reshape(4, 2 * 20)
        )


def test_plot_face(au, au2):
    plot_face(au=au, vectorfield={"reference": predict(au2)}, feature_range=(0, 1))
    assert_plot_shape(plt.gca())
    plt.close("all")

    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={"reference": predict(au2)})
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield=[])
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={"noreference": predict(au2)})

    plt.close("all")


def test_plot_detections(default_detector, single_face_img, multi_face_img):
    single_image_single_face_prediction = default_detector.detect_image(single_face_img)
    figs = single_image_single_face_prediction.plot_detections()
    assert len(figs) == 1
    plt.close("all")

    multi_image_single_face = default_detector.detect_image(
        [single_face_img, single_face_img]
    )
    # With AU landmark model and muscles
    figs = multi_image_single_face.plot_detections(faces="aus", muscles=True)
    assert len(figs) == 2
    plt.close("all")

    single_image_multi_face_prediction = default_detector.detect_image(multi_face_img)
    figs = single_image_multi_face_prediction.plot_detections()
    assert len(figs) == 1
    plt.close("all")

    multi_image_multi_face_prediction = default_detector.detect_image(
        [multi_face_img, multi_face_img]
    )
    figs = multi_image_multi_face_prediction.plot_detections()
    assert len(figs) == 2
    plt.close("all")

    multi_image_mix_prediction = default_detector.detect_image(
        [single_face_img, multi_face_img], batch_size=1
    )
    figs = multi_image_mix_prediction.plot_detections()
    assert len(figs) == 2
    plt.close("all")

    # Don't currently support projecting through the AU model if there are multiple faces
    with pytest.raises(NotImplementedError):
        multi_image_mix_prediction.plot_detections(faces="aus")


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
