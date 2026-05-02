"""Maintainer tool: migrate the xgb_au .skops model file to a modern format.

Why this exists
---------------
The original ``xgb_au_classifier.skops`` on HuggingFace was saved with an
older xgboost serialization that current xgboost (3.x) parses via a
deprecated path (``XGBoosterUnserializeFromBuffer``). On Python 3.13
that path crashes with a SIGSEGV during ``Booster.__setstate__`` because
the bytearray buffer skops hands xgboost is freed/relocated mid-call.

The fix is structural: re-save the model. Loading the old file once via
the deprecated path (with a defensive bytearray-copy patch) and then
calling ``skops.io.dump`` on the result produces a new .skops whose
embedded Booster buffers are in xgboost's modern UBJ format. The
modern format goes through ``XGBoosterLoadModel`` instead of
``XGBoosterUnserializeFromBuffer`` and works without any patches.

The migration is **lossless**: predictions on synthetic features match
bit-exactly across all 20 AU sub-classifiers (verified at migration
time).

A second, free benefit: the original .skops attributed the outer
wrapper class as ``__main__.XGBClassifier`` (a training-script
artifact). Re-saving with the class imported from its real location
fixes that to ``feat.au_detectors.StatLearning.SL_test.XGBClassifier``,
removing the need for the ``sys.modules['__main__'].XGBClassifier =``
shim at load time.

Usage
-----
    python scripts/migrate_xgb_au_skops.py \\
        --in feat/resources/models--py-feat--xgb_au/.../xgb_au_classifier.skops \\
        --out xgb_au_classifier_v2.skops

Then upload the output file to ``py-feat/xgb_au`` on the HuggingFace Hub
and update the py-feat ``Detector`` filename if the maintainer chooses
to publish the migrated file under a new name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xgboost.core


def _patched_setstate_factory(orig):
    """Make a Booster.__setstate__ that copies the buffer first.

    Some xgboost+skops+Python combos free the bytearray skops hands the
    Booster before xgboost finishes reading it. Copying defends against
    that lifetime issue on the read side."""

    def patched(self, state):
        if "handle" in state and state["handle"] is not None:
            state = {**state, "handle": bytearray(state["handle"])}
        return orig(self, state)

    return patched


def _verify_predictions_match(old, new, *, n_samples: int = 10, seed: int = 0) -> bool:
    """Compare predict_proba on random features across all classifiers.

    Each Booster is trained on a different feature count (912, 1195, or
    1379 depending on which face region it covers). We probe each one
    at its native feature width."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument("--in", dest="src", required=True, help="path to old .skops")
    parser.add_argument("--out", dest="dst", required=True, help="path for new .skops")
    parser.add_argument("--no-verify", action="store_true", help="skip prediction parity check")
    args = parser.parse_args()

    # Patch xgboost.Booster.__setstate__ to copy the buffer (lets the
    # old file load without segfault). Reverted before loading the new
    # file so the verification proves the new file works without patches.
    orig_setstate = xgboost.core.Booster.__setstate__
    xgboost.core.Booster.__setstate__ = _patched_setstate_factory(orig_setstate)

    # The original .skops references __main__.XGBClassifier instead of
    # feat.au_detectors.StatLearning.SL_test.XGBClassifier. Inject the
    # real class into __main__ so skops can resolve it at load time.
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier
    sys.modules["__main__"].XGBClassifier = XGBClassifier

    from skops.io import dump, get_untrusted_types, load

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        parser.error(f"input file does not exist: {src}")

    print(f"loading {src} (old format, with patch)...")
    old = load(src, trusted=get_untrusted_types(file=src))
    print(f"  classifiers: {len(old.classifiers)}")

    print(f"saving {dst} (modern format)...")
    dump(old, dst)

    # Verify the new file loads cleanly without the patch
    xgboost.core.Booster.__setstate__ = orig_setstate
    print(f"loading {dst} without patch (should work)...")
    new = load(dst, trusted=get_untrusted_types(file=dst))

    if not args.no_verify:
        print("verifying bit-identical predictions across all classifiers...")
        if _verify_predictions_match(old, new):
            print("  all classifiers produce bit-identical outputs")
        else:
            print("ERROR: predictions diverged after migration", file=sys.stderr)
            return 2

    print(f"done: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
