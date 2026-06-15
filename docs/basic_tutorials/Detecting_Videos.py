import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch

    # Use the best available device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    # Pass this to Detectorv2(device=...) so the tutorial uses your GPU when present.
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
    # 2. Detecting facial expressions from videos

    In this tutorial we'll use **`Detectorv2`** — Py-Feat's single multi-task model — to process a video file, first one frame at a time and then in batches to speed things up on a GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Setting up the detector

    We create a `Detectorv2` instance just like in the previous tutorial. One network predicts Action Units, emotions, valence/arousal, gaze, head pose, a 478-point 3D FaceMesh, and blendshapes in a single forward pass.
    """)
    return


@app.cell
def _(device):
    from feat import Detectorv2

    detector = Detectorv2(device=device)  # device selected above (cuda/mps/cpu)
    return (detector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Processing a video

    Detecting facial expressions in a video uses the same `.detect()` method with `data_type="video"`. This sample video included in Py-Feat is by [Wolfgang Langer](https://www.pexels.com/@wolfgang-langer-1415383?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/video/a-woman-exhibits-different-emotions-through-facial-expressions-3063838/).
    """)
    return


@app.cell
def _():
    from feat.utils.io import get_test_data_path
    import os

    test_data_dir = get_test_data_path()
    test_video_path = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

    # (The input video is processed below; an inline preview is omitted in the
    # static docs. Download the notebook or open it in molab to view it.)
    test_video_path
    return (test_video_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We pass `skip_frames=24` to process only every 24th frame for speed, and `face_detection_threshold=0.95` to be conservative about what counts as a face — we know this clip is a continuous front-on shot of one person, so raising it from the default `0.5` avoids spurious extra detections.

    By default `.detect()` processes **one frame at a time** (`batch_size=1`):
    """)
    return


@app.cell
def _(detector, test_video_path):
    # Without batching: one frame at a time (batch_size=1, the default).
    video_prediction = detector.detect(
        test_video_path,
        data_type="video",
        skip_frames=24,
        face_detection_threshold=0.95,
    )
    video_prediction.head()
    return (video_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our 20-second clip recorded at 24 fps yields 20 predictions because of `skip_frames=24`:
    """)
    return


@app.cell
def _(video_prediction):
    video_prediction.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Speeding things up with batching

    Passing `batch_size > 1` runs several frames through the network in a single forward pass instead of one at a time. This is **much faster on a GPU** (CUDA or MPS) and is the recommended way to process video. On CUDA you can squeeze out a bit more by also passing `pin_memory=True`, which page-locks host memory for faster CPU→GPU transfers. The predictions are identical — only throughput changes:
    """)
    return


@app.cell
def _(detector, test_video_path):
    # With batching: 8 frames per forward pass — much faster on a GPU.
    # On CUDA, pin_memory=True further speeds host->device transfers.
    video_prediction_batched = detector.detect(
        test_video_path,
        data_type="video",
        batch_size=8,
        skip_frames=24,
        face_detection_threshold=0.95,
    )
    video_prediction_batched.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 Visualizing predictions

    You can plot detection results from a video. The frames aren't extracted from the video (that would produce thousands of images), so the visualization shows the detected face geometry without the underlying image.

    The clip runs at 24 fps; the actress shows sadness around 0:02 and happiness around 0:14.
    """)
    return


@app.cell
def _(mo, video_prediction):
    # Frame 48 ~ 0:02 (sadness), Frame 408 ~ 0:14 (happiness)
    _figs = video_prediction.query("frame in [48, 408]").plot_detections(
        faceboxes=False, add_titles=False
    )
    mo.vstack(_figs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also use pandas plotting to show how emotions unfold over time — the shift from sadness to happiness is clearly visible:
    """)
    return


@app.cell
def _(video_prediction):
    _ax = video_prediction.emotions.plot()
    _ax.figure
    return


if __name__ == "__main__":
    app.run()
