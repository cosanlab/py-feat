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
    return device, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 6. Performance & hardware

        A practical guide to running Py-Feat fast: choosing the right device,
        batching, and a few defaults that matter more than they look.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.1 Pick a device

        Detectors run on CPU by default. Pass `device=...` to use a GPU. The
        portable pattern selects **CUDA** (NVIDIA) → **MPS** (Apple Silicon) →
        **CPU**, so the same notebook runs anywhere:

        ```python
        import torch
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        detector = Detectorv2(device=device)
        ```
        """
    )
    return


@app.cell
def _(device):
    print(f"Selected device: {device!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.2 Time a detection

        We warm up once (first call loads weights / compiles kernels) and time the
        second call:
        """
    )
    return


@app.cell
def _(device):
    import os
    import time

    from feat import Detectorv2
    from feat.utils.io import get_test_data_path

    detector = Detectorv2(device=device, identity_model="arcface")
    img_path = os.path.join(get_test_data_path(), "single_face.jpg")

    detector.detect(img_path, data_type="image")  # warmup
    _t0 = time.perf_counter()
    detector.detect(img_path, data_type="image")
    print(f"single-image detect: {time.perf_counter() - _t0:.3f}s on {device}")
    return detector, get_test_data_path, os, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.3 Batch images and video

        Processing inputs one at a time leaves the GPU idle between calls. Pass
        `batch_size > 1` to stack inputs into a single tensor — a large speedup on
        a GPU.

        - **Images:** `detector.detect(img_list, batch_size=8)`. All images in a
          batch must share dimensions; pass `output_size=...` to pad/resize
          mismatched images.
        - **Video:** `detector.detect(video, data_type="video", batch_size=8)`.
          Add `skip_frames=N` to process every *N*-th frame when you don't need
          every frame.
        """
    )
    return


@app.cell
def _(detector, get_test_data_path, os, time):
    multi = os.path.join(get_test_data_path(), "multi_face.jpg")
    img_list = [multi] * 8

    for _bs in (1, 8):
        detector.detect(img_list, batch_size=_bs, data_type="image")  # warmup
        _t0 = time.perf_counter()
        detector.detect(img_list, batch_size=_bs, data_type="image")
        print(f"8 images, batch_size={_bs}: {time.perf_counter() - _t0:.3f}s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.4 Leave `num_workers=0`

        `detect()` accepts a DataLoader `num_workers` argument. **Keep the default
        `num_workers=0`.** On Apple Silicon + Python 3.13 with Py-Feat's
        `OMP_NUM_THREADS=1` default, `num_workers > 0` is consistently *slower*
        (worst case ~33× for image batches) because worker processes contend for
        the same single-threaded BLAS/OMP pool. If a DataLoader feels slow, this
        is usually why.

        ## 6.5 Large datasets

        - Pass `save="out.csv"` to write results incrementally instead of holding
          every frame in memory.
        - Use `skip_frames` on long videos.
        - Reuse one detector instance across many files — model weights load once.
        """
    )
    return


if __name__ == "__main__":
    app.run()
