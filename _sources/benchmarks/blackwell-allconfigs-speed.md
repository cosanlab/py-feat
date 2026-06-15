# py-feat detector benchmark — 2026-05-31 07:21:14

## Run metadata

- **Date:** 2026-05-31 07:21:14
- **py-feat version:** 0.7.0
- **Git commit:** 29a11e1
- **Host:** liquidswords2 (x86_64, 128 CPUs)
- **Python:** 3.12.13
- **PyTorch:** 2.11.0+cu128
- **GPU:** CUDA 12.8, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cuda']
- **Batch sizes:** [1, 4, 16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 5.25 | 72.9 | 13.7 |
| cuda | 4 | 2.58 | 35.9 | 27.9 |
| cuda | 16 | 1.74 | 24.1 | 41.4 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 3.66 | 50.8 | 19.7 |
| cuda | 4 | 1.05 | 14.6 | 68.4 |
| cuda | 16 | 0.54 | 7.6 | 132.1 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 3.05 | 42.3 | 23.6 |
| cuda | 4 | 0.85 | 11.8 | 85.0 |
| cuda | 16 | 0.71 | 9.9 | 101.5 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 2.34 | 32.5 | 30.8 |
| cuda | 4 | 0.68 | 9.5 | 105.7 |
| cuda | 16 | 0.32 | 4.5 | 223.1 |


## Video: long (472 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 35.60 | 75.4 | 13.3 |
| cuda | 4 | 18.76 | 39.7 | 25.2 |
| cuda | 16 | 15.34 | 32.5 | 30.8 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 23.10 | 48.9 | 20.4 |
| cuda | 4 | 7.34 | 15.5 | 64.3 |
| cuda | 16 | 2.92 | 6.2 | 161.5 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 20.30 | 43.0 | 23.3 |
| cuda | 4 | 5.85 | 12.4 | 80.7 |
| cuda | 16 | 2.43 | 5.2 | 194.0 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 15.86 | 33.6 | 29.8 |
| cuda | 4 | 4.70 | 10.0 | 100.3 |
| cuda | 16 | 2.19 | 4.6 | 215.5 |


## Images: 16 x multi_face.jpg = 80 faces

### img2pose

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 2.39 | 149.3 | 80 |
| cuda | 4 | 1.18 | 73.8 | 80 |
| cuda | 16 | 1.10 | 69.0 | 80 |


### retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 1.76 | 109.7 | 80 |
| cuda | 4 | 0.65 | 40.8 | 80 |
| cuda | 16 | 0.52 | 32.7 | 80 |


### MPDetector retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 0.80 | 50.0 | 80 |
| cuda | 4 | 0.32 | 20.1 | 80 |
| cuda | 16 | 1.52 | 94.8 | 80 |


### Detectorv2 multitask

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 0.65 | 40.7 | 80 |
| cuda | 4 | 0.32 | 19.9 | 80 |
| cuda | 16 | 0.37 | 23.4 | 80 |

