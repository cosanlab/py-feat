"""Maintainer tool: re-save AU classifier .skops files in modern format.

Why this exists
---------------
The published ``xgb_au_classifier.skops`` and ``svm_au_classifier.skops``
on the py-feat HuggingFace org each have format issues that surface on
modern toolchains:

1. Both files reference their outer wrapper class as ``__main__.SVMClassifier``
   or ``__main__.XGBClassifier`` (a training-script artifact). Loading
   requires ``sys.modules['__main__'].XGBClassifier = ...`` which py-feat
   ships as a workaround.

2. The xgb file's embedded Booster buffers are in xgboost's pre-3.x
   serialization format. Current xgboost (3.x) parses them via the
   deprecated ``XGBoosterUnserializeFromBuffer`` path with a one-time
   warning. On Python 3.13 that path also has an intermittent
   bytearray-lifetime SIGSEGV during ``Booster.__setstate__``; copying
   the buffer mitigates but does not fully eliminate it.

This script re-saves each file. The output:

- references the real wrapper class path
  (``feat.au_detectors.StatLearning.SL_test.{XGB,SVM}Classifier``)
  rather than ``__main__``,
- and for xgb, embeds Booster buffers in xgboost's modern UBJ format.

The migration is **lossless**: predictions match bit-exactly between
old and new file across all 20 AU sub-classifiers (verified at
migration time).

Usage
-----
    # xgb
    python scripts/migrate_au_skops.py --model xgb \\
        --in feat/resources/models--py-feat--xgb_au/.../xgb_au_classifier.skops \\
        --out xgb_au_classifier_v2.skops

    # svm
    python scripts/migrate_au_skops.py --model svm \\
        --in feat/resources/models--py-feat--svm_au/.../svm_au_classifier.skops \\
        --out svm_au_classifier_v2.skops

Then upload the output files to ``py-feat/xgb_au`` and ``py-feat/svm_au``
on the HuggingFace Hub alongside the originals (do not replace), and
update the ``filename=`` in ``feat/detector.py`` to point at the v2 files.
Old py-feat releases on PyPI continue to download the original files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xgboost.core


def _patched_setstate_factory(orig):
    """Make a Booster.__setstate__ that copies the buffer first.

    The bytearray skops hands xgboost can be reallocated mid-call on
    Python 3.13. Copying defends against that lifetime issue."""

    def patched(self, state):
        if state.get("handle") is not None:
            state = {**state, "handle": bytearray(state["handle"])}
        return orig(self, state)

    return patched


def _verify_xgb_predictions(old, new, *, n_samples: int = 10, seed: int = 0) -> bool:
    """Compare xgb predict_proba on random features, per-classifier feature width."""

    rng = np.random.default_rng(seed)
    ok = True
    for au, old_clf in old.classifiers.items():
        n_feat = old_clf.get_booster().num_features()
        features = rng.normal(size=(n_samples, n_feat)).astype(np.float32)
        old_probs = old_clf.predict_proba(features)
        new_probs = new.classifiers[au].predict_proba(features)
        if not np.array_equal(old_probs, new_probs):
            ok = False
            diff = np.abs(old_probs - new_probs).max()
            print(f"  AU {au}: MISMATCH (max diff {diff:.2e}, n_feat={n_feat})")
    return ok


def _verify_svm_predictions(old, new, *, n_samples: int = 10, seed: int = 0) -> bool:
    """Compare svm decision_function on random features.

    sklearn's LinearSVC takes a fixed feature width per classifier. We
    probe each one at the width its trained coefficients expect."""

    rng = np.random.default_rng(seed)
    ok = True
    for au, old_clf in old.classifiers.items():
        n_feat = old_clf.coef_.shape[1]
        features = rng.normal(size=(n_samples, n_feat)).astype(np.float32)
        old_dec = old_clf.decision_function(features)
        new_dec = new.classifiers[au].decision_function(features)
        if not np.array_equal(old_dec, new_dec):
            ok = False
            diff = np.abs(old_dec - new_dec).max()
            print(f"  AU {au}: MISMATCH (max diff {diff:.2e}, n_feat={n_feat})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument(
        "--model", choices=["xgb", "svm"], required=True,
        help="which classifier file to migrate"
    )
    parser.add_argument("--in", dest="src", required=True, help="path to old .skops")
    parser.add_argument("--out", dest="dst", required=True, help="path for new .skops")
    parser.add_argument("--no-verify", action="store_true", help="skip prediction parity check")
    args = parser.parse_args()

    # The original .skops files reference the wrapper class as __main__.{XGB,SVM}Classifier
    # instead of feat.au_detectors.StatLearning.SL_test.{XGB,SVM}Classifier. Inject the
    # real classes into __main__ so skops can resolve them at load time.
    from feat.au_detectors.StatLearning.SL_test import SVMClassifier, XGBClassifier
    sys.modules["__main__"].XGBClassifier = XGBClassifier
    sys.modules["__main__"].SVMClassifier = SVMClassifier

    # Patch xgboost.Booster.__setstate__ to copy the buffer (lets the old xgb file
    # load on Python 3.13 without segfault). Only needed for xgb.
    orig_setstate = xgboost.core.Booster.__setstate__
    if args.model == "xgb":
        xgboost.core.Booster.__setstate__ = _patched_setstate_factory(orig_setstate)

    from skops.io import dump, get_untrusted_types, load

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        parser.error(f"input file does not exist: {src}")

    print(f"loading {src}...")
    old = load(src, trusted=get_untrusted_types(file=src))
    print(f"  classifiers: {len(old.classifiers)}")

    print(f"saving {dst} (modern format, real module path)...")
    dump(old, dst)

    print(f"loading {dst}...")
    new = load(dst, trusted=get_untrusted_types(file=dst))

    if not args.no_verify:
        print("verifying bit-identical predictions across all classifiers...")
        verify = _verify_xgb_predictions if args.model == "xgb" else _verify_svm_predictions
        if verify(old, new):
            print("  all classifiers produce bit-identical outputs")
        else:
            print("ERROR: predictions diverged after migration", file=sys.stderr)
            return 2

    # Restore original Booster.__setstate__ for hygiene
    xgboost.core.Booster.__setstate__ = orig_setstate

    # Confirm the new file's untrusted-types list no longer references __main__
    new_ut = get_untrusted_types(file=dst)
    has_main_ref = any(t.startswith("__main__.") for t in new_ut)
    if has_main_ref:
        print(f"WARNING: new file still references __main__: {new_ut}", file=sys.stderr)
    else:
        print(f"  new file untrusted types: {new_ut}")

    print(f"done: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
