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

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:02&lt;00:00,  2.44s/it]100%|██████████| 1/1 [00:02&lt;00:00,  2.44s/it]
  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 36.06it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">single-image detect: 0.030s on cuda
</pre>

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

```python
multi = os.path.join(get_test_data_path(), "multi_face.jpg")
img_list = [multi] * 8

for _bs in (1, 8):
    detector.detect(img_list, batch_size=_bs, data_type="image")  # warmup
    _t0 = time.perf_counter()
    detector.detect(img_list, batch_size=_bs, data_type="image")
    print(f"8 images, batch_size={_bs}: {time.perf_counter() - _t0:.3f}s")
```

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/8 [00:00&lt;?, ?it/s] 12%|█▎        | 1/8 [00:01&lt;00:11,  1.61s/it] 62%|██████▎   | 5/8 [00:01&lt;00:00,  3.76it/s]100%|██████████| 8/8 [00:01&lt;00:00,  4.38it/s]
  0%|          | 0/8 [00:00&lt;?, ?it/s] 50%|█████     | 4/8 [00:00&lt;00:00, 31.98it/s]100%|██████████| 8/8 [00:00&lt;00:00, 31.81it/s]100%|██████████| 8/8 [00:00&lt;00:00, 31.79it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=1: 0.257s
</pre>

<pre class="marimo-book-output-text marimo-stream-stderr">  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:03&lt;00:00,  3.98s/it]100%|██████████| 1/1 [00:03&lt;00:00,  3.98s/it]
  0%|          | 0/1 [00:00&lt;?, ?it/s]100%|██████████| 1/1 [00:00&lt;00:00, 10.39it/s]
</pre>

<pre class="marimo-book-output-text marimo-stream-stdout">8 images, batch_size=8: 0.099s
</pre>

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
