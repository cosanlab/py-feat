# Accuracy benchmarks

Tracks per-release accuracy of py-feat detectors against held-out labeled datasets. Each entry is a single run produced by `python scripts/bench_regression.py --markdown`. Throughput benchmarks live in [throughput.md](throughput.md).

## Latest

See [2026-05-15-67cd7d9-accuracy.md](2026-05-15-67cd7d9-accuracy.md).

## Methodology

- **DISFA** P3 fold, ArcFace-aligned crops, AU intensity binarized at >=2 for F1; ICC(3,1) on continuous intensity vs. py-feat probability.
- **AffectNet** validation set, classes 0..6 mapped to the 7 py-feat emotion columns; top-1 emotion accuracy and macro F1.
- **CALFW / CPLFW** 6000 pairs, LFW 10-fold CV protocol, InsightFace 5-landmark template alignment before ArcFace embedding.
- **TinyFace** closed-set + open-set rank-K identification with the Gallery_Distractor set (153k images) when not disabled.

## History

| date | run |
|---|---|
| 2026-05-15-67cd7d9 | [2026-05-15-67cd7d9-accuracy.md](2026-05-15-67cd7d9-accuracy.md) |
