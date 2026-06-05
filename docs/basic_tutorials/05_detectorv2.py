import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch

    # Use the best available device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
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
    mo.md(
        r"""
        # 5. Beyond detection: everything `Detectorv2` adds

        *`Detectorv2` is the recommended detector in Py-Feat 0.7+.* Tutorial&nbsp;1
        used it for the basics; here we focus on the outputs it produces in a
        **single forward pass** that the modular v1 `Detector` does *not*:
        **valence/arousal**, **gaze**, 6-DoF **head pose**, and a **478-point 3D
        FaceMesh** — plus the familiar Action Units and emotions.
        """
    )
    return


@app.cell
def _(device):
    import os

    from feat import Detectorv2
    from feat.utils.io import get_test_data_path

    # identity_model="arcface" adds a face-identity embedding alongside everything else.
    detector = Detectorv2(device=device, identity_model="arcface")

    img_path = os.path.join(get_test_data_path(), "single_face.jpg")
    fex = detector.detect(img_path, data_type="image")
    fex.head()
    return detector, fex, img_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.1 Valence & arousal

        v2 predicts continuous **valence** (unpleasant → pleasant) and **arousal**
        (calm → excited) — the two affective dimensions the v1 detector can't
        produce. They're plain columns on the `Fex`:
        """
    )
    return


@app.cell
def _(fex):
    fex[["valence", "arousal"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.2 Gaze

        The gaze head (L2CS) outputs `gaze_pitch` / `gaze_yaw` in radians.
        `plot_detections(gazes=True)` draws a yellow arrow from each face in the
        predicted direction.
        """
    )
    return


@app.cell
def _(fex):
    print("gaze columns:", fex.gaze_columns)
    fex[["gaze_pitch", "gaze_yaw"]]
    return


@app.cell
def _(fex):
    _figs = fex.plot_detections(faces="landmarks", gazes=True, muscles=False)
    _figs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.3 Head pose

        Six degrees of freedom — rotation (`Pitch` / `Roll` / `Yaw`) and
        translation (`X` / `Y` / `Z`):
        """
    )
    return


@app.cell
def _(fex):
    fex.poses
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.4 Action Units & emotions

        v2 still produces the familiar 20 Action Units and a 7-class emotion
        distribution. Note v2's emotion columns are **capitalized** (`Neutral`,
        `Happy`, …) where the v1 detector used lowercase:
        """
    )
    return


@app.cell
def _(fex):
    fex.aus
    return


@app.cell
def _(fex):
    fex.emotions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.5 The 478-point 3D FaceMesh

        v2 also returns a dense 478-point 3D MediaPipe FaceMesh. `plot_face_mesh`
        draws it as a 3D wireframe (here driven by the detected Action Units); for
        an interactive version use `plot_face_mesh_plotly` and open this notebook
        in molab to spin it around.
        """
    )
    return


@app.cell
def _(fex):
    from feat.plotting import plot_face_mesh

    _ax = plot_face_mesh(au=fex.aus.iloc[0].to_numpy())
    _ax.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        One `Detectorv2` forward pass gives you AUs, emotions, valence/arousal,
        gaze, head pose, 68-point landmarks, a 478-point 3D mesh, and (with
        `identity_model="arcface"`) an identity embedding — all in a single `Fex`.
        See [tutorial 1](01_basics.md) for the basics and the modular v1 detector.
        """
    )
    return


if __name__ == "__main__":
    app.run()
