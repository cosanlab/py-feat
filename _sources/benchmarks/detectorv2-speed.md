# py-feat detector benchmark — 2026-05-31 07:13:22

## Run metadata

- **Date:** 2026-05-31 07:13:22
- **py-feat version:** 0.7.0
- **Git commit:** fcabbf0
- **Host:** liquidswords2 (x86_64, 128 CPUs)
- **Python:** 3.12.13
- **PyTorch:** 2.11.0+cu128
- **GPU:** CUDA 12.8, NVIDIA GeForce RTX 3090
- **OMP_NUM_THREADS:** `1`
- **Devices swept:** ['cuda']
- **Batch sizes:** [1, 4, 16]
- **DataLoader workers:** [0]

Each timed call is preceded by one untimed warmup; the timed-call wall time is reported.

## Video: short (72 frames)

### Detectorv2 multitask

| device | batch | sec | ms/frame | fps |
|---|---|---|---|---|
| cuda | 1 | 2.43 | 33.8 | 29.6 |
| cuda | 4 | 1.06 | 14.8 | 67.8 |
| cuda | 16 | 0.83 | 11.5 | 86.8 |


## Images: 16 x multi_face.jpg = 80 faces

### Detectorv2 multitask

| device | batch | sec | ms/img | rows |
|---|---|---|---|---|
| cuda | 1 | 0.99 | 62.0 | 80 |
| cuda | 4 | 0.77 | 48.1 | 80 |
| cuda | 16 | 0.71 | 44.2 | 80 |

