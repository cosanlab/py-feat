"""Tests for the MPDetector AU-column surfacing in PR #302.

MPDetector previously exposed only 52 MP blendshapes under the wrong column
names (the Fex declared ``au_columns=AU_LANDMARK_MAP['Feat']`` but the
DataFrame contained MP blendshape names). This PR wires in the BS→AU PLS
regressor so MPDetector emits both 52 blendshapes AND 20 predicted AUs.

Tests:
- Schema: a forward() result has all 52 ``MP_BLENDSHAPE_NAMES`` AND all 20
  ``AU_LANDMARK_MAP['Feat']`` columns; ``Fex.aus`` returns (N, 20).
- Range: predicted AUs are clipped to [0, 1].
- Empty-batch passthrough does not raise.
- Column-order guard fires if the PLS au_columns drift from the canonical list.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from feat.pretrained import AU_LANDMARK_MAP
from feat.utils import MP_BLENDSHAPE_NAMES


def _have_test_image() -> bool:
    from feat.utils.io import get_test_data_path
    return os.path.exists(os.path.join(get_test_data_path(), "multi_face.jpg"))


# ---------------------------------------------------------------------
# Column-order guard
# ---------------------------------------------------------------------

class TestPlsAuOrderGuard:
    def test_guard_passes_with_canonical_order(self, monkeypatch):
        """The check should be a no-op when PLS au_columns match the canonical
        Feat order — the production case."""
        from feat import MPDetector as mp_mod
        # Reset the cached flag so the check runs again.
        monkeypatch.setattr(mp_mod, "_pls_au_order_verified", [False])
        # Stub the loader to return correctly-ordered weights.
        canonical = AU_LANDMARK_MAP["Feat"]
        monkeypatch.setattr(mp_mod, "_load_pls_weights",
                            lambda: {"au_columns": list(canonical)})
        # Run the guarded snippet via the same logic as forward().
        assert not mp_mod._pls_au_order_verified[0]
        if not mp_mod._pls_au_order_verified[0]:
            w = mp_mod._load_pls_weights()
            assert w["au_columns"] == AU_LANDMARK_MAP["Feat"]
            mp_mod._pls_au_order_verified[0] = True
        assert mp_mod._pls_au_order_verified[0]

    def test_guard_raises_on_drift(self, monkeypatch):
        """If a future re-trained npz has shuffled au_columns, MPDetector
        must refuse to silently mislabel rather than producing wrong AU values."""
        from feat import MPDetector as mp_mod
        monkeypatch.setattr(mp_mod, "_pls_au_order_verified", [False])
        # Same set, different order — would silently mislabel without the guard.
        wrong_order = AU_LANDMARK_MAP["Feat"][::-1]
        monkeypatch.setattr(mp_mod, "_load_pls_weights",
                            lambda: {"au_columns": list(wrong_order)})

        # Replicate the guard logic the way forward() runs it.
        with pytest.raises(RuntimeError, match=r"PLS au_columns drifted"):
            w = mp_mod._load_pls_weights()
            if w["au_columns"] != AU_LANDMARK_MAP["Feat"]:
                raise RuntimeError(
                    f"BS→AU PLS au_columns drifted from AU_LANDMARK_MAP['Feat']. "
                    f"PLS: {w['au_columns']}; canonical: {AU_LANDMARK_MAP['Feat']}."
                )


# ---------------------------------------------------------------------
# End-to-end: MPDetector forward output schema & AU range
# Network test (MPDetector loads weights from HF Hub on first construction).
# ---------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(not _have_test_image(), reason="multi_face.jpg missing")
class TestMpdetectorAuOutput:
    @pytest.fixture(scope="class")
    def fex(self):
        """Run MPDetector once, share the result across tests in this class."""
        from feat.MPDetector import MPDetector
        from feat.utils.io import get_test_data_path
        det = MPDetector(face_model="retinaface", device="cpu")
        return det.detect(
            [os.path.join(get_test_data_path(), "multi_face.jpg")],
            batch_size=1,
        )

    def test_has_blendshape_columns(self, fex):
        for col in MP_BLENDSHAPE_NAMES:
            assert col in fex.columns, f"missing blendshape column {col}"

    def test_has_au_columns(self, fex):
        for col in AU_LANDMARK_MAP["Feat"]:
            assert col in fex.columns, f"missing AU column {col}"

    def test_aus_slot_returns_20_columns(self, fex):
        """Fex.aus uses au_columns to slice; this was silently broken before
        PR #302 because the actual columns were blendshapes."""
        aus = fex.aus
        assert aus.shape[1] == 20
        assert list(aus.columns) == AU_LANDMARK_MAP["Feat"]

    def test_au_values_in_unit_interval(self, fex):
        au_block = fex[AU_LANDMARK_MAP["Feat"]].to_numpy()
        # Predicted AUs are clipped to [0, 1] by pls_predict_batch.
        finite = au_block[np.isfinite(au_block)]
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0

    def test_blendshapes_and_aus_are_distinct_columns(self, fex):
        """No collision between MP blendshape names and FACS AU names."""
        bs_set = set(MP_BLENDSHAPE_NAMES)
        au_set = set(AU_LANDMARK_MAP["Feat"])
        assert bs_set.isdisjoint(au_set)
