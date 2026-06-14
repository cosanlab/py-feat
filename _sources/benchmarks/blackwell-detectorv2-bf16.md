# py-feat detector benchmark — 2026-05-31 07:26:28

## Run metadata

- **Date:** 2026-05-31 07:26:28
- **py-feat version:** 0.7.0
- **Git commit:** dcd035c
- **Host:** liquidswords2 (x86_64, 128 CPUs)
- **Python:** 3.12.13
- **PyTorch:** 2.11.0+cu128
- **GPU:** CUDA 12.8, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cuda']
- **Batch sizes:** [16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 16 | 0.97 | 13.5 | 74.3 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 16 | 1.43 | 19.9 | 50.2 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 16 | 0.33 | 4.6 | 216.7 |


## Video: long (472 frames)

### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 16 | 4.11 | 8.7 | 115.0 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 16 | 3.13 | 6.6 | 150.6 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 16 | 1.98 | 4.2 | 239.0 |


## Images: 16 x multi_face.jpg = 80 faces

### retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 16 | 0.58 | 36.0 | 80 |


### MPDetector retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 16 | 1.40 | 87.7 | 80 |


### Detectorv2 multitask

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 16 | 0.32 | 20.1 | 80 |

