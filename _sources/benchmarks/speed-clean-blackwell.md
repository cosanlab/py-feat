# py-feat detector benchmark — 2026-06-05 16:59:44

## Run metadata

- **Date:** 2026-06-05 16:59:44
- **py-feat version:** 0.7.0
- **Git commit:** 6645db3
- **Host:** liquidswords2 (x86_64, 128 CPUs)
- **Python:** 3.12.13
- **PyTorch:** 2.11.0+cu128
- **GPU:** CUDA 12.8, NVIDIA GeForce RTX 3090
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cuda']
- **Batch sizes:** [1, 16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 5.07 | 70.4 | 14.2 |
| cuda | 16 | 2.40 | 33.3 | 30.0 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 3.43 | 47.7 | 21.0 |
| cuda | 16 | 0.83 | 11.5 | 86.7 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 3.03 | 42.1 | 23.8 |
| cuda | 16 | 1.85 | 25.7 | 39.0 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 1.84 | 25.6 | 39.0 |
| cuda | 16 | 0.36 | 5.0 | 202.0 |


## Video: long (472 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 38.25 | 81.0 | 12.3 |
| cuda | 16 | 20.51 | 43.5 | 23.0 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 23.59 | 50.0 | 20.0 |
| cuda | 16 | 4.81 | 10.2 | 98.1 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 20.30 | 43.0 | 23.2 |
| cuda | 16 | 5.36 | 11.4 | 88.1 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 12.46 | 26.4 | 37.9 |
| cuda | 16 | 2.33 | 4.9 | 202.5 |


## Images: 16 x multi_face.jpg = 80 faces

### img2pose

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 1.91 | 119.1 | 80 |
| cuda | 16 | 1.28 | 80.0 | 80 |


### retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 1.17 | 73.2 | 80 |
| cuda | 16 | 0.81 | 50.9 | 80 |


### MPDetector retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 1.01 | 63.1 | 80 |
| cuda | 16 | 1.49 | 93.1 | 80 |


### Detectorv2 multitask

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 0.59 | 37.1 | 80 |
| cuda | 16 | 0.31 | 19.2 | 80 |

