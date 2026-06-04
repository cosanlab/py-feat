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
    # 2. Detecting facial expressions from videos

    *Written by Jin Hyun Cheong and Eshin Jolly*

    In this tutorial we'll explore how to use the `Detector` class to process video files. You can try it out interactively in Google Collab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosanlab/py-feat/blob/master/notebooks/content/03_detector_vids.ipynb)
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
    ## 2.1 Setting up the Detector

    We'll begin by creating a new `Detector` instance just like the previous tutorial and using the defaults:
    """)
    return


@app.cell
def _():
    from feat import Detector

    detector = Detector()

    detector
    return (detector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Processing videos

    Detecting facial expressions in videos is easy to do using the `.detect_video()` method. This sample video included in Py-Feat is by [Wolfgang Langer](https://www.pexels.com/@wolfgang-langer-1415383?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/video/a-woman-exhibits-different-emotions-through-facial-expressions-3063838/).
    """)
    return


@app.cell
def _():
    from feat.utils.io import get_test_data_path
    import os

    test_data_dir = get_test_data_path()
    test_video_path = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

    # Show video
    from IPython.core.display import Video

    Video(test_video_path, embed=False)
    return (test_video_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Just like before we can call the `.detect()` method, but this time we tell Py-feat that `data_type='video'`.

    Here we also set `skip_frames=24` which tells the detector to process only every 24th frame for the sake of speed.

    We also set `face_detection_threshold=0.95` which tells the detector to be extremely conservative in what it considers a face. Since we already know that this video is a continuous front-on shot of one person, raising this value from the default of 0.5 will result in a much fewer false positive detections of more than one face per frame.
    """)
    return


@app.cell
def _(detector, test_video_path):
    video_prediction = detector.detect(
        test_video_path, data_type="video", skip_frames=24, face_detection_threshold=0.95
    )
    video_prediction.head()
    return (video_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that our 20s long video, recorded at 24 frames-per-second, produces 20 predictions because we set `skip_frames=24`:
    """)
    return


@app.cell
def _(video_prediction):
    video_prediction.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Visualizing predictions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also plot the detection results from a video. The frames are not extracted from the video (that will result in thousands of images) so the visualization only shows the detected face without the underlying image.

    The video has 24 fps and the actress show sadness around the 0:02, and happiness at 0:14 seconds.
    """)
    return


@app.cell
def _(video_prediction):
    # Frame 48 = ~0:02
    # Frame 408 = ~0:14
    video_prediction.query("frame in [48, 408]").plot_detections(
        faceboxes=False, add_titles=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also leverage existing pandas plotting functions to show how emotions unfold over time. We can clearly see how her emotions change from sadness to happiness.
    """)
    return


@app.cell
def _(video_prediction):
    axes = video_prediction.emotions.plot()
    return


if __name__ == "__main__":
    app.run()
