# 6. Performance & hardware

A practical guide to running Py-Feat fast: choosing the right device,
batching, and a few defaults that matter more than they look.

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

```python
print(f"Selected device: {device!r}")
```

<pre class="marimo-book-output-text marimo-stream-stdout">Selected device: &#x27;cuda&#x27;
</pre>

## 6.2 Time a detection

We warm up once (first call loads weights / compiles kernels) and time the
second call:

```python
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
```

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:02&lt;00:00,  2.64s/it]100%|██████████| 1/1 [00:02&lt;00:00,  2.64s/it]
  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 37.09it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">single-image detect: 0.029s on cuda
</pre>

## 6.3 Batch images and video

Batching is the single biggest lever on GPU throughput. Processing inputs
one at a time leaves the GPU idle between calls; passing `batch_size > 1`
stacks inputs into one tensor so the network runs them in parallel. The
sweep below shows throughput (images/second) climbing with batch size —
**pick the largest batch that fits in VRAM** (drop it back down if you hit
an out-of-memory error).

- **Images:** `detector.detect(img_list, batch_size=8)`. All images in a
  batch must share dimensions; pass `output_size=(H, W)` to pad/resize
  mismatched images so they stack.
- **Video:** `detector.detect(video, data_type="video", batch_size=8)`.
  Add `skip_frames=N` to process every *N*-th frame when you don't need
  every frame — see the *Detecting Videos* tutorial for a full example.

```python
multi = os.path.join(get_test_data_path(), "multi_face.jpg")
img_list = [multi] * 8

for _bs in (1, 2, 4, 8):
    detector.detect(img_list, batch_size=_bs, data_type="image")  # warmup
    _t0 = time.perf_counter()
    detector.detect(img_list, batch_size=_bs, data_type="image")
    _dt = time.perf_counter() - _t0
    print(f"8 images, batch_size={_bs}: {_dt:.3f}s ({8 / _dt:.1f} img/s)")
```

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/8 [00:00&lt;?, ?it/s] 12%|█▎        | 1/8 [00:01&lt;00:10,  1.56s/it] 62%|██████▎   | 5/8 [00:01&lt;00:00,  3.86it/s]100%|██████████| 8/8 [00:01&lt;00:00,  4.50it/s]
  0%|          | 0/8 [00:00&lt;?, ?it/s] 50%|█████     | 4/8 [00:00&lt;00:00, 32.47it/s]100%|██████████| 8/8 [00:00&lt;00:00, 32.31it/s]100%|██████████| 8/8 [00:00&lt;00:00, 32.29it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=1: 0.252s (31.7 img/s)
</pre>

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/4 [00:00&lt;?, ?it/s] 25%|██▌       | 1/4 [00:03&lt;00:11,  3.76s/it]100%|██████████| 4/4 [00:03&lt;00:00,  1.35it/s]100%|██████████| 4/4 [00:03&lt;00:00,  1.03it/s]
  0%|          | 0/4 [00:00&lt;?, ?it/s] 75%|███████▌  | 3/4 [00:00&lt;00:00, 26.89it/s]100%|██████████| 4/4 [00:00&lt;00:00, 26.78it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=2: 0.153s (52.3 img/s)
</pre>

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/2 [00:00&lt;?, ?it/s] 50%|█████     | 1/2 [00:03&lt;00:03,  3.76s/it]100%|██████████| 2/2 [00:03&lt;00:00,  1.90s/it]
  0%|          | 0/2 [00:00&lt;?, ?it/s]100%|██████████| 2/2 [00:00&lt;00:00, 19.48it/s]100%|██████████| 2/2 [00:00&lt;00:00, 19.41it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=4: 0.107s (75.1 img/s)
</pre>

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:03&lt;00:00,  3.78s/it]100%|██████████| 1/1 [00:03&lt;00:00,  3.78s/it]
  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 10.46it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=8: 0.098s (81.3 img/s)
</pre>

## 6.4 Pin memory for faster CUDA transfers

Every batch is copied from host (CPU) RAM into GPU memory before the
network runs. On **CUDA**, passing `pin_memory=True` allocates that batch
in *page-locked* host memory, which lets the copy overlap with computation
— Py-Feat already issues the host→device transfer with `non_blocking=True`,
so the two halves pair up for a small, free win on GPU-bound batches.

`pin_memory` has no effect on **MPS** or **CPU** (there's no pinned-memory
fast path), so only set it when `device="cuda"`.

```python
if device == "cuda":
    for _pin in (False, True):
        detector.detect(
            img_list, batch_size=8, pin_memory=_pin, data_type="image"
        )  # warmup
        _t0 = time.perf_counter()
        detector.detect(
            img_list, batch_size=8, pin_memory=_pin, data_type="image"
        )
        print(f"8 images, batch_size=8, pin_memory={_pin}: {time.perf_counter() - _t0:.3f}s")
else:
    print(f"pin_memory only affects CUDA; current device is {device!r} — skipping.")
```

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 11.18it/s]
  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 11.38it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=8, pin_memory=False: 0.090s
</pre>

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 10.13it/s]
  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 11.42it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=8, pin_memory=True: 0.090s
</pre>

## 6.5 Leave `num_workers=0`

`detect()` accepts a DataLoader `num_workers` argument. **Keep the default
`num_workers=0`.** On Apple Silicon + Python 3.13 with Py-Feat's
`OMP_NUM_THREADS=1` default, `num_workers > 0` is consistently *slower*
(worst case ~33× for image batches) because worker processes contend for
the same single-threaded BLAS/OMP pool. If a DataLoader feels slow, this
is usually why.

## 6.6 Large datasets

- Pass `save="out.csv"` to write results incrementally instead of holding
  every frame in memory.
- Use `skip_frames` on long videos.
- Reuse one detector instance across many files — model weights load once.
