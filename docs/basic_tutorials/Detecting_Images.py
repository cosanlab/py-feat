import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch

    # Use the best available device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    # Pass this to Detectorv1(device=...) so the tutorial uses your GPU when present.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Detecting facial expressions from images

    In this tutorial we'll explore the `Detectorv1` class in more depth, demonstrating how to detect faces, facial landmarks, action units, and emotions from images.
    """)
    return


@app.cell
def _():
    # Uncomment the line below and run this only if you're using Google Collab
    # !pip install -q py-feat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Setting up a detector

    The recommended way to extract facial features in Py-Feat 0.7+ is **`Detectorv2`** — a single **multi-task neural network** that, in one forward pass, predicts Action Units, emotions, **valence/arousal**, **gaze**, head pose, 68-point landmarks, and a **478-point 3D MediaPipe FaceMesh**. It's fast (especially on single frames) and is what the rest of this tutorial uses. Passing `identity_model="arcface"` also adds a face-identity embedding.

    The first time you initialize a detector, Py-Feat downloads the required pretrained weights from [our HuggingFace Repository](https://huggingface.co/py-feat) and caches them to disk; subsequent runs reuse the cached weights.

    You can find a list of default models [on this page](/models.md). For the older modular detector, see the [legacy `Detectorv1` (v1)](#legacy-the-modular-detectorv1-v1) section just below.
    """)
    return


@app.cell
def _(device):
    from feat import Detectorv2

    # One multi-task model: AUs, emotions, valence/arousal, gaze, head pose,
    # 68-pt landmarks, and a 478-pt 3D FaceMesh. identity_model="arcface" adds a
    # face-identity embedding. device was selected above (cuda/mps/cpu).
    detector_v2 = Detectorv2(device=device, identity_model="arcface")
    return (detector_v2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Legacy: the modular `Detectorv1` (v1)

    Before `Detectorv2`, Py-Feat used **`Detectorv1`** — a *modular* pipeline that glues together a **separate pre-trained model per sub-task** (face, landmarks, Action Units, emotion, head pose, identity). Reach for it when you want to **swap or disable a specific model** (e.g. `Detectorv1(emotion_model='svm')`) or need the classic modular behavior. It exposes the **same `.detect()` API** and returns the same kind of `Fex` object, so everything below works with either detector.

    `Detectorv2` is the recommended default for new work; see the [two-detector overview](/#two-detectors-detectorv1-and-detectorv2) for a full comparison.
    """)
    return


@app.cell
def _(device):
    from feat import Detectorv1

    # The modular Detectorv1. Swap individual models via kwargs, e.g.
    # Detectorv1(emotion_model='svm'). device was selected above (cuda/mps/cpu).
    detector = Detectorv1(device=device)
    return (detector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Processing a single image
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's process a single image with a single face. Py-feat includes a demo image for this purpose called `single_face.jpg` so lets use that. You can also use the convenient `imshow` function which will automatically load an image into a numpy array if provided a path unlike matplotlib:
    """)
    return


@app.cell
def _():
    from feat.utils.io import get_test_data_path
    from feat.plotting import imshow
    import os

    # Helper to point to the test data folder
    test_data_dir = get_test_data_path()

    # Get the full path
    single_face_img_path = os.path.join(test_data_dir, "single_face.jpg")

    # Plot it
    imshow(single_face_img_path)
    return os, single_face_img_path, test_data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we use our initialized `detector` instance to make predictions with the `.detect()` method, passing `data_type="image"`. This is the main workhorse method that will perform face, landmark, au, and emotion detection using the loaded models. It always returns a `Fex` data instance:
    """)
    return


@app.cell
def _(detector_v2, single_face_img_path):
    single_face_prediction = detector_v2.detect(single_face_img_path, data_type="image")

    type(single_face_prediction)  # instance of a Fex class

    # Show results
    single_face_prediction
    return (single_face_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Working with `Fex` outputs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output of any detection always returns a `Fex` data class instance. This class is a lightweight wrapper around a pandas dataframe that contains columns with values for detection type.

    So you can use any pandas methods you're already familiar with:
    """)
    return


@app.cell
def _(single_face_prediction):
    # We always return a dataframe even if there's just a single row,
    # i.e. no Series
    single_face_prediction.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Fex` provides convenient attributes to access specific groups of columns so you don't have to write a bunch of pandas code to get the data you need:
    """)
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.faceboxes
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.aus
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.emotions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Detectorv2` also predicts continuous **valence** (unpleasant → pleasant) and
    **arousal** (calm → excited) — the two affective dimensions the modular v1
    `Detectorv1` does not produce. They're plain `Fex` columns:
    """)
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction[["valence", "arousal"]]
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.poses
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.identities
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 Saving and Loading detections from a file

    Since a `Fex` object is just a sub-classed `DataFrames` we can use the `.to_csv` method to save our detections toa file:
    """)
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.to_csv("output.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To create a new `Fex` instance from a csv file use our custom `read_feat()` function instead pf `pd.read_csv`:
    """)
    return


@app.cell
def _():
    from feat.utils.io import read_feat

    input_prediction = read_feat("output.csv")

    # We can quick access features like before
    input_prediction.aus
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Real-time saving during detection (low-memory mode)
    You can also write `Fex` outputs to a file during detection by passing a `save` argument to `detect`. This will save the `Fex` output to a csv file every time a face is detected.

    This can be useful when processing multiple images or videos (as we'll see later).
    """)
    return


@app.cell
def _(detector_v2, single_face_img_path):
    fex = detector_v2.detect(inputs=single_face_img_path, data_type="image", save='detections.csv')

    fex.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use our terminal to see that `detections.csv` exists and contains the same content as `fex`
    """)
    return


@app.cell
def _():
    import subprocess
    subprocess.run('head detections.csv', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.5 Visualizing detection results.

    `Fex` objects have a method called `.plot_detections()` to generate a summary figure of detected faces, action units and emotions. It always returns a list of matplotlib figures:
    """)
    return


@app.cell
def _(single_face_prediction):
    _figs = single_face_prediction.plot_detections(poses=True)
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default `.plot_detections()` will overlay facial lines on top of the input image. However, it's also possible to visualize a face using Py-Feat's standardized AU landmark model, which takes the detected AUs and projects them onto a template face. You can control this by setting `faces='aus'` instead of the default `faces='landmarks'`. For more details about this kind of visualization see the [visualizing facial expressions](Plotting.md) tutorial:
    """)
    return


@app.cell
def _(detector, single_face_img_path):
    # AU-projection visualization (faces='aus') uses Detectorv1's named xgb
    # AU model and its trained landmark viz model; Detectorv2's AUs have no
    # projection model, so we use the legacy detector here. See tutorial 03 for
    # more on AU visualization.
    _v1_fex = detector.detect(single_face_img_path, data_type="image")
    _figs = _v1_fex.plot_detections(faces='aus', muscles=True)
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overlaying gaze direction

    `plot_detections(gazes=True)` overlays a yellow arrow on each detected
    face showing where it's looking. The arrow direction comes from
    `gaze_pitch` / `gaze_yaw` columns produced by whichever gaze model is
    active — in v0.7+ the default is **L2CS** (Abdelrahman et al. 2022, a
    ResNet50 trained on Gaze360 + MPIIGaze). Angles are in radians,
    head-centric: positive pitch = looking up, positive yaw = subject's
    gaze drifts toward the viewer's right. Pass `gaze_model='geometric'`
    to `MPDetector` for a lightweight landmark-only fallback that doesn't
    need the ResNet50 download.
    """)
    return


@app.cell
def _(single_face_prediction):
    # fex.gaze_columns lists which columns hold the gaze model's output;
    # for L2CS that's gaze_pitch and gaze_yaw (radians).
    print('gaze columns:', single_face_prediction.gaze_columns)
    print(single_face_prediction[['gaze_pitch', 'gaze_yaw']])
    _figs = single_face_prediction.plot_detections(faces='landmarks', gazes=True, muscles=False)
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive Plotting

    You can also use the `.iplot_detections()` method to generate an interactive plotly figure that lets you interactively enable/disable various detector outputs:
    """)
    return


@app.cell
def _(detector, single_face_img_path):
    # Interactive plotting uses the v1 detector here: Detectorv2's emotion
    # columns (Neutral/Happy/...) aren't yet wired into iplot_detections.
    _v1_fex = detector.detect(single_face_img_path, data_type="image")
    _v1_fex.iplot_detections(bounding_boxes=True, emotions=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.6 Detecting multiple faces from a single image

    A `Detectorv1` will automatically find multiple faces in a single image and will create 1 row per detected face in the `Fex` object it outputs.

    Notice how `image_prediction` is now a `Fex` instance with 5 rows, one for each detected face. We can confirm this by plotting our detection results like before:
    """)
    return


@app.cell
def _(detector_v2, os, test_data_dir):
    multi_face_image_path = os.path.join(test_data_dir, "multi_face.jpg")
    multi_face_prediction = detector_v2.detect(multi_face_image_path, data_type="image")

    # Show results
    multi_face_prediction
    return multi_face_image_path, multi_face_prediction


@app.cell
def _(multi_face_prediction):
    _figs = multi_face_prediction.plot_detections(add_titles=False)
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.7 Working with multiple images

    `Detectorv1` is also flexible enough to process multiple image files if `.detect()` is passed a list of images. By default images will be processed serially, but you can set `batch_size > 1` to process multiple images in a *batch* and speed up processing. **NOTE: All images in a batch must have the same dimensions for batch processing.** This is because behind the scenes, `Detectorv1` is assembling a *tensor* by stacking images together. You can ask `Detectorv1` to rescale images by padding and preserving proportions using the `output_size` in conjunction with `batch_size`. For example, the following would process a list of images in batches of 5 images at a time resizing each so one axis is 512:

    `detector_v2.detect(img_list, batch_size=5, output_size=512) # without output_size this would raise an error if image sizes differ!`

    In the example below we keep things simple, by process both our single and multi-face example serislly by setting `batch_size = 1`.

    Notice how the returned Fex data class instance has 6 rows: 1 for the first face in the first image, and 5 for the faces in the second image:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **NOTE: Currently batch processing images gives slightly different AU detection results due to the way that py-feat integrates the underlying models. You can examine the degree of tolerance by checking out the results of `test_detection_and_batching_with_diff_img_sizes` in our test-suite**
    """)
    return


@app.cell
def _(detector_v2, multi_face_image_path, single_face_img_path):
    img_list = [single_face_img_path, multi_face_image_path]

    mixed_prediction = detector_v2.detect(img_list, batch_size=1, data_type="image")
    mixed_prediction
    return (mixed_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calling `.plot_detections()` will now plot detections for all images the detector was passed:
    """)
    return


@app.cell
def _(mixed_prediction):
    _figs = mixed_prediction.plot_detections()
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, it's easy to use pandas slicing syntax to just grab predictions for the image you want. For example you can use `.loc` and chain it to `.plot_detections()`:
    """)
    return


@app.cell
def _(mixed_prediction):
    # Just plot the detection corresponding to the first row in the Fex data
    _figs = mixed_prediction.loc[0].plot_detections()
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Likewise you can use `.query()` and chain it to `.plot_detections()`. `Fex` data classes store each file path in the `'input'` column. So we can use regular pandas methods like `.unique()` to get all the unique images (2 in our case) and pick the second one.
    """)
    return


@app.cell
def _(mixed_prediction):
    # Choose plot based on image file name
    img_name = mixed_prediction["input"].unique()[1]
    axes = mixed_prediction.query("input == @img_name").plot_detections()
    axes[0]
    return


if __name__ == "__main__":
    app.run()
