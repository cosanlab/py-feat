# py-feat detector benchmark — 2026-06-05 16:53:37

## Run metadata

- **Date:** 2026-06-05 16:53:37
- **py-feat version:** 0.7.0
- **Git commit:** 6645db3
- **Host:** liquidswords2 (x86_64, 128 CPUs)
- **Python:** 3.12.13
- **PyTorch:** 2.11.0+cu128
- **GPU:** CUDA 12.8, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cuda']
- **Batch sizes:** [1, 16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 6.24 | 86.7 | 11.5 |
| cuda | 16 | 1.84 | 25.6 | 39.1 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 4.22 | 58.6 | 17.1 |
| cuda | 16 | 0.67 | 9.3 | 107.1 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 3.08 | 42.8 | 23.4 |
| cuda | 16 | 1.67 | 23.2 | 43.0 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 1.89 | 26.3 | 38.1 |
| cuda | 16 | 0.27 | 3.7 | 270.6 |


## Video: long (472 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 40.20 | 85.2 | 11.7 |
| cuda | 16 | 14.87 | 31.5 | 31.7 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 23.38 | 49.5 | 20.2 |
| cuda | 16 | 2.82 | 6.0 | 167.2 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 20.31 | 43.0 | 23.2 |
| cuda | 16 | 3.33 | 7.1 | 141.8 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 12.54 | 26.6 | 37.6 |
| cuda | 16 | 1.33 | 2.8 | 355.5 |


## Images: 16 x multi_face.jpg = 80 faces

### img2pose

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 2.19 | 136.7 | 80 |
| cuda | 16 | 0.99 | 62.2 | 80 |


### retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 1.00 | 62.3 | 80 |
| cuda | 16 | 0.49 | 30.6 | 80 |


### MPDetector retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 0.81 | 50.6 | 80 |
| cuda | 16 | 1.35 | 84.5 | 80 |


### Detectorv2 multitask

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 0.51 | 32.2 | 80 |
| cuda | 16 | 0.18 | 11.0 | 80 |

