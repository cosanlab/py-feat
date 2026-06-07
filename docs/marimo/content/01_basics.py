import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Detecting facial expressions from images

    *Written by Jin Hyun Cheong and Eshin Jolly*

    In this tutorial we'll explore the `Detector` class in more depth, demonstrating how to detect faces, facial landmarks, action units, and emotions from images. You can try it out interactively in Google Collab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosanlab/py-feat/blob/master/notebooks/content/02_detector_imgs.ipynb)
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
    ## 1.1 Downloading models from HuggingFace and setting up a `Detector`

    A `Detector` is a swiss-army-knife class that "glues" together a combination of *pre-trained* Face, Emotion, Pose, etc detection models into a single Python object. This allows us to provide a very easy-to-use high-level API, e.g. `detector.detect('my_image.jpg',data_type='image')`, which will automatically make use of the correct underlying model to solve the sub-tasks of identifying face locations, getting landmarks, extracting action units, etc.

    The first time you initialize a `Detector` instance on your computer will take a moment as Py-Feat will automatically download required pretrained model weights for you from [our HuggingFace Repository](https://huggingface.co/py-feat) and save them to disk. Every time after that it will use existing model weights.

    You can find a list of default models [on this page](/models.md).
    """)
    return


@app.cell
def _():
    from feat import Detector

    detector = Detector()

    # You can change which models you want during initialization, e.g.
    # detector = Detector(emotion_model='svm')
    return (detector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1b A faster alternative: `Detectorv2`

    Py-Feat also ships **`Detectorv2`**, which runs a single **multi-task neural network** instead of the modular pipeline above. In one forward pass it predicts Action Units, emotions, valence/arousal, gaze, head pose, and a **478-point 3D MediaPipe FaceMesh** — so it is **much faster, especially on single frames**, and adds valence/arousal + gaze that `Detector` does not produce.

    Use **`Detector` (v1)** when you want to pick or disable specific models or need the classic 68-point landmarks; use **`Detectorv2` (v2)** when you want speed, the 478-point 3D mesh, or valence/arousal + gaze. Both return the same kind of `Fex` object, so the rest of this tutorial applies to either. See the [two-detector overview](/intro.md#two-detectors-detector-and-detectorv2) for a full comparison.
    """)
    return


@app.cell
def _():
    from feat import Detectorv2

    # One multi-task model: AUs, emotions, valence/arousal, gaze, 478-pt 3D mesh, head pose
    detector_v2 = Detectorv2(device="cpu")  # use device="cuda" or "mps" if available

    # Detect exactly like the v1 Detector — same Fex output:
    # fex = detector_v2.detect(single_face_img_path, data_type="image")
    return


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
    Now we use our initialized `detector` instance to make predictions with the `detect_image()` method. This is the main workhorse method that will perform face, landmark, au, and emotion detection using the loaded models. It always returns a `Fex` data instance:
    """)
    return


@app.cell
def _(detector, single_face_img_path):
    single_face_prediction = detector.detect(single_face_img_path, data_type="image")

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
def _(detector, single_face_img_path):
    fex = detector.detect(inputs=single_face_img_path, data_type="image", save='detections.csv')

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
    single_face_prediction.plot_detections(poses=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default `.plot_detections()` will overlay facial lines on top of the input image. However, it's also possible to visualize a face using Py-Feat's standardized AU landmark model, which takes the detected AUs and projects them onto a template face. You an control this by change by setting `faces='aus'` instead of the default `faces='landmarks'`. For more details about this kind of visualization see the [visualizing facial expressions](./04_plotting.ipynb) and the [creating an AU visualization model](../extra_tutorials/06_trainAUvisModel.ipynb) tutorials:
    """)
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.plot_detections(faces='aus', muscles=True)
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
    single_face_prediction.plot_detections(faces='landmarks', gazes=True, muscles=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive Plotting

    You can also use the `.iplot_detections()` method to generate an interactive plotly figure that lets you interactively enable/disable various detector outputs:
    """)
    return


@app.cell
def _(single_face_prediction):
    single_face_prediction.iplot_detections(bounding_boxes=True, emotions=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.6 Detecting multiple faces from a single image

    A `Detector` will automatically find multiple faces in a single image and will create 1 row per detected face in the `Fex` object it outputs.

    Notice how `image_prediction` is now a `Fex` instance with 5 rows, one for each detected face. We can confirm this by plotting our detection results like before:
    """)
    return


@app.cell
def _(detector, os, test_data_dir):
    multi_face_image_path = os.path.join(test_data_dir, "multi_face.jpg")
    multi_face_prediction = detector.detect(multi_face_image_path, data_type="image")

    # Show results
    multi_face_prediction
    return multi_face_image_path, multi_face_prediction


@app.cell
def _(multi_face_prediction):
    multi_face_prediction.plot_detections(add_titles=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.7 Working with multiple images

    `Detector` is also flexible enough to process multiple image files if `.detect()` is passed a list of images. By default images will be processed serially, but you can set `batch_size > 1` to process multiple images in a *batch* and speed up processing. **NOTE: All images in a batch must have the same dimensions for batch processing.** This is because behind the scenes, `Detector` is assembling a *tensor* by stacking images together. You can ask `Detector` to rescale images by padding and preserving proportions using the `output_size` in conjunction with `batch_size`. For example, the following would process a list of images in batches of 5 images at a time resizing each so one axis is 512:

    `detector.detect(img_list, batch_size=5, output_size=512) # without output_size this would raise an error if image sizes differ!`

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
def _(detector, multi_face_image_path, single_face_img_path):
    img_list = [single_face_img_path, multi_face_image_path]

    mixed_prediction = detector.detect(img_list, batch_size=1, data_type="image")
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
    mixed_prediction.plot_detections()
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
    mixed_prediction.loc[0].plot_detections()
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
    return


if __name__ == "__main__":
    app.run()
