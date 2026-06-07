#!/usr/bin/env bash
# Benchmark py-feat (and optionally LibreFace) on an Apple Silicon Mac (M-series)
# for the cross-tool speed comparison. Run from the py-feat repo root on the Mac.
#
#   bash scripts/benchmark_mac.sh
#
# Produces docs/benchmarks/m5-<date>.md (py-feat throughput on MPS + CPU, batch
# 1 and 16, on the shared test fixtures). Send that file back to add the "M5"
# column to the speed matrix.
#
# Notes on the other tools on Apple Silicon:
#   - LibreFace: its API exposes only device="cpu"/"cuda" (no MPS), so on a Mac
#     it runs CPU-only; this script times that if a libreface venv is provided
#     via $LIBREFACE_PY (path to a python with `libreface` installed).
#   - OpenFace 3.0: torch MPS may work; not wired here yet.
#   - PyAFAR: GPU is Ubuntu/WSL2-only and the install is broken (pysimplegui /
#     dlib) — it does not run on macOS. That blank is part of the comparison.
set -euo pipefail

DATE=$(date +%Y-%m-%d)
OUT="docs/benchmarks/m5-${DATE}.md"

echo "== py-feat: detecting device support =="
python - <<'PY'
import torch
print("torch:", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
PY

echo
echo "== py-feat throughput sweep (MPS + CPU, batch 1 & 16) -> ${OUT} =="
# bench_detectors.py records the real device/host in the dump metadata.
python scripts/bench_detectors.py --devices mps cpu --batches 1 16 --markdown "${OUT}"

echo
echo "Done. py-feat M5 numbers are in: ${OUT}"
echo "Send that file back to fill the M5 column."

# --- Optional: LibreFace on the Mac (CPU only) -------------------------------
if [[ -n "${LIBREFACE_PY:-}" ]]; then
  echo
  echo "== LibreFace (CPU) on single_face.mp4 =="
  "${LIBREFACE_PY}" - <<'PY'
import time, libreface
VID = "feat/tests/data/single_face.mp4"; NF = 72
libreface.get_facial_attributes_video(VID, device="cpu", batch_size=16)  # warmup
for bs in (1, 16):
    t0 = time.time()
    libreface.get_facial_attributes_video(VID, device="cpu", batch_size=bs)
    dt = time.time() - t0
    print(f"LibreFace  device=cpu(Apple)  batch={bs}  {NF/dt:.1f} fps  ({dt:.1f}s)")
PY
else
  echo
  echo "(Skip LibreFace: set LIBREFACE_PY=/path/to/venv/bin/python with libreface installed to also time it CPU-only.)"
fi
