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
    # Legacy v1 PLS viz model bounds; still the defaults of
    # _create_empty_figure() so draw_lineface / draw_vectorfield smoke
    # tests pass against them. plot_face itself now auto-derives the
    # viewport from the actual landmark range, so don't use this assertion
    # against plot_face output — use assert_plot_face_visible() instead.
    assert ax.get_ylim() == (240.0, 50.0)
    assert ax.get_xlim() == (25.0, 172.0)


def assert_plot_face_visible(ax):
    """Sanity check that plot_face actually rendered face content into the
    visible viewport. The new auto-derived viewport depends on which viz
    model is active and varies per AU vector, so check pixel content
    instead of hardcoded bounds: any dark pixels inside the axis frame
    means lines/patches drew successfully.
    """
    import numpy as np
    fig = ax.figure
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    H = arr.shape[0]
    x0, y0, x1, y1 = (int(v * fig.dpi) for v in (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
    inner = arr[max(H - y1, 0) + 5:max(H - y0, 0) - 5, x0 + 5:x1 - 5, :]
    if inner.size == 0:
        return  # plot too small to inspect; tolerate
    dark_pixels = int(np.sum(np.any(inner < 200, axis=-1)))
    assert dark_pixels > 100, f"plot_face axis appears empty ({dark_pixels} dark px)"


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
    # plot_face auto-derives the viewport from landmark coords, so the
    # fixed v1 bounds no longer apply. Verify pixel content instead.
    assert_plot_face_visible(plt.gca())
    plt.close("all")

    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={"reference": predict(au2)})
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield=[])
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={"noreference": predict(au2)})

    plt.close("all")


@pytest.mark.skip
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

    animate_face(start=starting_aus, end=ending_aus, save="test.gif")
    assert exists("test.gif")

    # # Clean up
    remove("test.gif")

    # Test different init style
    animate_face(AU=1, start=0, end=3, save="test.gif")
    assert exists("test.gif")

    # Clean up
    remove("test.gif")
