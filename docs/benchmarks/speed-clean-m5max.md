# py-feat detector benchmark — 2026-06-22 16:49:39

## Run metadata

- **Date:** 2026-06-22 16:49:39
- **py-feat version:** 2.0.2
- **Git commit:** 788e6d8
- **Host:** m5max (arm64, 18 CPUs)
- **Python:** 3.13.12
- **PyTorch:** 2.11.0
- **GPU:** Apple M5 Max (MPS)
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cpu', 'mps']
- **Batch sizes:** [1, 4, 16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 27.52 | 382.2 | 2.6 |
| cpu | 4 | 26.91 | 373.8 | 2.7 |
| cpu | 16 | 107.46 | 1492.5 | 0.7 |
| mps | 1 | 7.90 | 109.7 | 9.1 |
| mps | 4 | 5.63 | 78.2 | 12.8 |
| mps | 16 | 5.34 | 74.1 | 13.5 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 13.91 | 193.2 | 5.2 |
| cpu | 4 | 13.24 | 183.9 | 5.4 |
| cpu | 16 | 74.70 | 1037.5 | 1.0 |
| mps | 1 | 3.73 | 51.9 | 19.3 |
| mps | 4 | 1.51 | 21.0 | 47.6 |
| mps | 16 | 1.00 | 13.9 | 71.9 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 31.91 | 443.2 | 2.3 |
| cpu | 4 | 29.45 | 409.0 | 2.4 |
| cpu | 16 | 36.05 | 500.7 | 2.0 |
| mps | 1 | 2.22 | 30.8 | 32.5 |
| mps | 4 | 0.77 | 10.7 | 93.5 |
| mps | 16 | 0.54 | 7.4 | 134.3 |


## Video: long (472 frames)

### img2pose

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| mps | 1 | 60.77 | 128.7 | 7.8 |
| mps | 4 | 44.73 | 94.8 | 10.6 |
| mps | 16 | 42.03 | 89.0 | 11.2 |


### retinaface

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 88.73 | 188.0 | 5.3 |
| cpu | 4 | 85.13 | 180.4 | 5.5 |
| cpu | 16 | 537.09 | 1137.9 | 0.9 |
| mps | 1 | 24.75 | 52.4 | 19.1 |
| mps | 4 | 10.61 | 22.5 | 44.5 |
| mps | 16 | 6.45 | 13.7 | 73.2 |


### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cpu | 1 | 205.84 | 436.1 | 2.3 |
| cpu | 4 | 193.92 | 410.8 | 2.4 |
| cpu | 16 | 238.38 | 505.0 | 2.0 |
| mps | 1 | 15.69 | 33.2 | 30.1 |
| mps | 4 | 5.44 | 11.5 | 86.7 |
| mps | 16 | 3.67 | 7.8 | 128.5 |


## Images: 16 x multi_face.jpg = 80 faces

### img2pose

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cpu | 1 | 13.26 | 829.0 | 80 |
| cpu | 4 | 67.18 | 4198.8 | 80 |
| cpu | 16 | 64.01 | 4000.4 | 80 |
| mps | 1 | 2.74 | 171.4 | 80 |
| mps | 4 | 2.20 | 137.5 | 80 |
| mps | 16 | 2.31 | 144.1 | 80 |


### retinaface

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cpu | 1 | 12.17 | 760.9 | 80 |
| cpu | 4 | 66.15 | 4134.2 | 80 |
| cpu | 16 | 64.35 | 4021.8 | 80 |
| mps | 1 | 1.34 | 83.8 | 80 |
| mps | 4 | 0.85 | 53.4 | 80 |
| mps | 16 | 0.96 | 60.3 | 80 |


### Detectorv2 multitask

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cpu | 1 | 29.61 | 1850.6 | 80 |
| cpu | 4 | 17.93 | 1120.8 | 80 |
| cpu | 16 | 24.38 | 1524.0 | 80 |
| mps | 1 | 0.75 | 46.8 | 80 |
| mps | 4 | 0.51 | 31.9 | 80 |
| mps | 16 | 0.52 | 32.2 | 80 |

