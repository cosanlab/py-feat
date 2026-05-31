"""Top-level package for FEAT."""

# --- OpenMP runtime conflict mitigation (must run before any other import) ---
#
# torch and xgboost both link against OpenMP, but PyPI wheels of each link
# against DIFFERENT libomp/libiomp5 dylibs on macOS. When both runtimes
# initialize independently in the same process, their thread-pool TLS state
# diverges and one of them later segfaults inside `__kmp_*` routines.
# Symptom on the affected configs: SIGSEGV (exit 139) during the FIRST op
# from whichever runtime initialized second.
#
# Forcing OMP_NUM_THREADS=1 at process start makes both runtimes use a
# single thread, which sidesteps the divergent thread-pool state and is
# the workaround documented in:
#   https://github.com/dmlc/xgboost/issues/11500
#   https://github.com/pytorch/pytorch/issues/44282
#   https://github.com/microsoft/LightGBM/issues/6595
#
# Performance impact is minor for typical py-feat use cases:
# - xgboost AU classifiers are small (n_estimators=100, depth=3), so
#   single-threaded scoring is essentially as fast as multi-threaded.
# - torch's heavy inference runs on MPS / CUDA / CPU-BLAS, which Apple's
#   Accelerate or MKL parallelize independently of OMP_NUM_THREADS.
# - numpy via OpenBLAS does respect OMP_NUM_THREADS, so dot products run
#   single-threaded.
# Users who need multi-threaded OMP can set OMP_NUM_THREADS=N in their
# environment BEFORE importing feat; this block only sets a default when
# the variable is unset.
#
# `KMP_DUPLICATE_LIB_OK=TRUE` was a tempting alternative but PyTorch
# explicitly warns it "may cause crashes or silently produce incorrect
# results" (#44282) — it suppresses the startup check without fixing the
# underlying divergent state. Don't use it.
import os as _os

if "OMP_NUM_THREADS" not in _os.environ:
    _os.environ["OMP_NUM_THREADS"] = "1"

# Import xgboost early too as defense in depth so its runtime
# (and the libomp it links against) is loaded before any torch-using
# import. With OMP_NUM_THREADS=1 above this is belt-and-braces, but
# cheap and harmless.
import xgboost  # noqa: F401, E402

__author__ = """Jin Hyun Cheong, Tiankang Xie, Sophie Byrne, Eshin Jolly, Luke Chang """
__email__ = "jcheong0428@gmail.com, eshin.jolly@gmail.com, luke.j.chang@dartmouth.edu"
__all__ = ["detector", "data", "utils", "plotting", "transforms", "__version__"]

from .data import Fex  # noqa: F401, E402
from .detector import Detector  # noqa: F401, E402
from .detector_v2 import Detectorv2  # noqa: F401, E402

# Explicit version alias: Detectorv1 == the original Detector (XGB/SVM AU +
# ResMaskNet emotion + L2CS gaze). Detector is kept for backward compat.
Detectorv1 = Detector

from .version import __version__  # noqa: E402
