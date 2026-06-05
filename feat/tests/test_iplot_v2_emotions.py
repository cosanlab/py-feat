"""Regression test: ``iplot_detections`` honors the Fex's own emotion columns.

``Fex.plot_singleframe_detections`` (used by ``iplot_detections``) hardcoded the
v1 emotion names ``["anger","disgust","fear","happiness","sadness","surprise",
"neutral"]`` when building the per-face emotion annotation. ``Detectorv2`` emits
capitalized names (``Neutral``/``Happy``/...), so the hardcoded lookup raised
``KeyError`` on any v2 detection — even with ``emotions=False`` — because the
annotation block runs unconditionally. The fix reads ``row[self.emotion_columns]``.
"""

import os

import plotly.graph_objects as go
import pytest

from feat.detector_v2 import Detectorv2
from feat.utils.io import get_test_data_path


@pytest.mark.network
def test_iplot_detections_uses_v2_emotion_columns():
    img = os.path.join(get_test_data_path(), "single_face.jpg")
    fex = Detectorv2(device="cpu", identity_model="arcface").detect(
        img, data_type="image"
    )
    # Detectorv2 emotions are capitalized — the old hardcoded v1 list would KeyError.
    assert "Happy" in fex.emotion_columns
    fig = fex.iplot_detections(bounding_boxes=True, emotions=True)
    assert isinstance(fig, go.Figure)
