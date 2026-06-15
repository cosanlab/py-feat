# py-feat detector benchmark — 2026-06-05 17:37:46

## Run metadata

- **Date:** 2026-06-05 17:37:46
- **py-feat version:** 0.7.0
- **Git commit:** ed84e74
- **Host:** liquidswords2 (x86_64, 128 CPUs)
- **Python:** 3.12.13
- **PyTorch:** 2.11.0+cu128
- **GPU:** CUDA 12.8, NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cpu']
- **Batch sizes:** [1, 16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 45.84 | 636.7 | 1.6 |
| cpu | 16 | 40.23 | 558.7 | 1.8 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 16.91 | 234.8 | 4.3 |
| cpu | 16 | 6.85 | 95.2 | 10.5 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 17.31 | 240.4 | 4.2 |
| cpu | 16 | 6.63 | 92.2 | 10.9 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 7.99 | 110.9 | 9.0 |
| cpu | 16 | 5.60 | 77.8 | 12.9 |


## Video: long (472 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 321.25 | 680.6 | 1.5 |
| cpu | 16 | 266.55 | 564.7 | 1.8 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 111.17 | 235.5 | 4.2 |
| cpu | 16 | 45.68 | 96.8 | 10.3 |


### MPDetector retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 115.35 | 244.4 | 4.1 |
| cpu | 16 | 33.75 | 71.5 | 14.0 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 53.11 | 112.5 | 8.9 |
| cpu | 16 | 26.84 | 56.9 | 17.6 |

