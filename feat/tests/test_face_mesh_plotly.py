"""Tests for ``feat.plotting.plot_face_mesh_plotly`` (PR7).

Interactive 3D Plotly backend for the face mesh viz. Mirrors PR #304's
matplotlib path but emits a ``go.Figure``. All tests are offline — synthetic
stub model + edge-count checks instead of pixel diffs (Plotly figures are
inspected via their data attribute, not rendered).
"""
from __future__ import annotations

import numpy as np
import pytest

from feat import plotting as plt_mod
from feat.plotting import (
    PLSAUMeshModel,
    plot_face_mesh_plotly,
)
from feat.utils.mp_plotting import FaceLandmarksConnections


N_TESSELATION_EDGES = len(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
N_CONTOURS_EDGES = len(FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)


@pytest.fixture
def stub_mesh_model(monkeypatch):
    """Synthetic PLSAUMeshModel with realistic-magnitude intercept ranges
    so the equal-aspect bounds and segment math don't NaN."""
    rng = np.random.default_rng(0)
    coef = rng.standard_normal((23, 1434)).astype(np.float32) * 0.05
    intercept = np.empty(1434, dtype=np.float32)
    intercept[:478] = rng.uniform(-7, 7, 478)
    intercept[478:956] = rng.uniform(-9, 8, 478)
    intercept[956:] = rng.uniform(-1, 7, 478)
    au_cols = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43",
    ]
    mean_mesh = np.column_stack([
        intercept[:478], intercept[478:956], intercept[956:],
    ])
    model = PLSAUMeshModel(
        coef=coef, intercept=intercept,
        au_columns=au_cols, pose_columns=["Pitch", "Yaw", "Roll"],
        mean_aligned_mesh=mean_mesh,
    )
    # Loader caches by version in a dict; inject under the default version
    # ("v4") so the bare plot_face_mesh_plotly() uses this stub offline.
    monkeypatch.setattr(plt_mod, "_PLS_MESH_MODELS", {"v4": model})
    return model


# ---------------------------------------------------------------------
# Returns a usable plotly Figure
# ---------------------------------------------------------------------

class TestReturnType:
    def test_returns_plotly_figure(self, stub_mesh_model):
        import plotly.graph_objects as go
        fig = plot_face_mesh_plotly()
        assert isinstance(fig, go.Figure)

    def test_has_mesh_and_eye_traces(self, stub_mesh_model):
        # Returns: 1 Scatter3d (mesh wireframe) + 4 Mesh3d (left iris,
        # left pupil, right iris, right pupil). Iris/pupil were added in
        # the v0.7 plotting overhaul to give the mesh visible eyes.
        fig = plot_face_mesh_plotly()
        types = [t.type for t in fig.data]
        assert types.count("scatter3d") == 1
        assert types.count("mesh3d") == 4
        wireframe = fig.data[0]
        assert wireframe.type == "scatter3d"
        assert wireframe.mode == "lines"


# ---------------------------------------------------------------------
# Mode dispatch — number of segments must match the chosen connection set
# ---------------------------------------------------------------------

class TestModeDispatch:
    def test_default_mode_is_tesselation(self, stub_mesh_model):
        """Default ``mode='tesselation'`` should emit 3 × 2556 = 7668 points
        (each edge contributes start, end, NaN separator)."""
        fig = plot_face_mesh_plotly()
        assert len(fig.data[0].x) == 3 * N_TESSELATION_EDGES

    def test_contours_mode_uses_canonical_subset(self, stub_mesh_model):
        """``mode='contours'`` should emit 3 × 124 = 372 points."""
        fig = plot_face_mesh_plotly(mode="contours")
        assert len(fig.data[0].x) == 3 * N_CONTOURS_EDGES

    def test_invalid_mode_raises(self, stub_mesh_model):
        with pytest.raises(ValueError, match=r"'tesselation'"):
            plot_face_mesh_plotly(mode="bogus")

    def test_tessellation_double_l_spelling_accepted(self, stub_mesh_model):
        """``'tessellation'`` (standard English, double-l) is an alias of
        the MediaPipe-compatible ``'tesselation'`` (single-l)."""
        fig_single = plot_face_mesh_plotly(mode="tesselation")
        fig_double = plot_face_mesh_plotly(mode="tessellation")
        # Same edge count → same number of points emitted
        assert len(fig_single.data[0].x) == len(fig_double.data[0].x)
        assert len(fig_double.data[0].x) == 3 * N_TESSELATION_EDGES

    def test_nan_separator_between_segments(self, stub_mesh_model):
        """Every 3rd point must be NaN — that's the line-segment terminator
        Plotly uses to break the polyline between disconnected edges."""
        fig = plot_face_mesh_plotly(mode="contours")
        seps = np.asarray(fig.data[0].x[2::3])
        assert np.all(np.isnan(seps)), "every 3rd x value should be NaN"


# ---------------------------------------------------------------------
# Source dispatch — au= / mesh= / neither
# ---------------------------------------------------------------------

class TestSourceDispatch:
    def test_au_drives_prediction(self, stub_mesh_model):
        """Passing AU should call into PLSAUMeshModel.predict and the
        resulting figure should differ from the rest-mesh figure."""
        rest_fig = plot_face_mesh_plotly()
        au = np.zeros(20, dtype=np.float32)
        au[stub_mesh_model.au_columns.index("AU12")] = 3.0
        au_fig = plot_face_mesh_plotly(au=au)
        # First non-NaN x of trace should differ between rest and AU=3
        rest_x = np.asarray(rest_fig.data[0].x)
        au_x = np.asarray(au_fig.data[0].x)
        # Both have the same shape; at least one finite point must differ.
        finite = ~(np.isnan(rest_x) | np.isnan(au_x))
        assert (rest_x[finite] != au_x[finite]).any()

    def test_mesh_kwarg_uses_provided_array(self, stub_mesh_model):
        """A precomputed mesh should be drawn directly without consulting
        the AU model."""
        custom_mesh = np.zeros((478, 3), dtype=np.float32)
        custom_mesh[:, 0] = np.linspace(-5, 5, 478)
        custom_mesh[:, 1] = np.linspace(-8, 8, 478)
        custom_mesh[:, 2] = np.linspace(0, 6, 478)
        fig = plot_face_mesh_plotly(mesh=custom_mesh, mode="contours")
        # The X axis (data X) should show the linspace range
        xs = np.asarray(fig.data[0].x)
        finite_x = xs[~np.isnan(xs)]
        assert finite_x.min() >= -5.001 and finite_x.max() <= 5.001

    def test_rejects_au_and_mesh_together(self, stub_mesh_model):
        with pytest.raises(ValueError, match=r"either `au` or `mesh`"):
            plot_face_mesh_plotly(
                au=np.zeros(20),
                mesh=np.zeros((478, 3), dtype=np.float32),
            )

    def test_rejects_wrong_mesh_shape(self, stub_mesh_model):
        with pytest.raises(ValueError, match=r"shape \(478, 3\)"):
            plot_face_mesh_plotly(mesh=np.zeros((100, 3), dtype=np.float32))

    def test_mesh_keyword_only(self, stub_mesh_model):
        """`mesh` must be keyword-only — positional sixth arg shouldn't
        silently bind to it (catches a future signature regression)."""
        with pytest.raises(TypeError):
            plot_face_mesh_plotly(
                None, None, "black", 1.5, 0.85, "white",
                np.zeros((478, 3), dtype=np.float32),
            )


# ---------------------------------------------------------------------
# Coordinate frame — data Y maps to plotly Z so the face is upright
# ---------------------------------------------------------------------

class TestAxisSwap:
    def test_data_y_maps_to_plotly_z(self, stub_mesh_model):
        """Data has Y up (forehead +, chin -). Plotly's default camera puts
        +Z up, so the function maps data Y → plotly Z. If the swap is
        dropped, the face renders lying down."""
        custom = np.zeros((478, 3), dtype=np.float32)
        custom[:, 1] = np.linspace(-9, 8, 478)   # data Y span = 17 (largest)
        custom[:, 2] = np.linspace(0, 5, 478)    # data Z span = 5 (smaller)
        fig = plot_face_mesh_plotly(mesh=custom, mode="contours")
        z_range = fig.layout.scene.zaxis.range
        # Plotly's Z (vertical) range must contain the data-Y span (~17),
        # not the data-Z span (~5).
        z_span = z_range[1] - z_range[0]
        assert z_span >= 17 - 1e-2, (
            f"Plotly Z span ({z_span:.3f}) should contain data-Y range (17)"
        )


# ---------------------------------------------------------------------
# Layout sanity
# ---------------------------------------------------------------------

class TestLayout:
    def test_axes_hidden(self, stub_mesh_model):
        fig = plot_face_mesh_plotly()
        scene = fig.layout.scene
        assert scene.xaxis.visible is False
        assert scene.yaxis.visible is False
        assert scene.zaxis.visible is False

    def test_aspectmode_cube(self, stub_mesh_model):
        """Cube aspect mode prevents Plotly from auto-stretching axes —
        face should render with correct proportions."""
        fig = plot_face_mesh_plotly()
        assert fig.layout.scene.aspectmode == "cube"

    def test_background_param(self, stub_mesh_model):
        fig = plot_face_mesh_plotly(background="lightgray")
        assert fig.layout.scene.bgcolor == "lightgray"
        assert fig.layout.paper_bgcolor == "lightgray"

    def test_color_and_line_width_passthrough(self, stub_mesh_model):
        """Style kwargs must reach the trace — catches a future refactor
        that drops them when building the Scatter3d."""
        fig = plot_face_mesh_plotly(color="crimson", line_width=4.0)
        line = fig.data[0].line
        assert line.color == "crimson"
        assert line.width == 4.0


# ---------------------------------------------------------------------
# Real HF Hub fetch — runs once, then hits cache.
# ---------------------------------------------------------------------

@pytest.mark.network
class TestRealModelSmoke:
    def test_renders_with_real_au_mesh_model(self, monkeypatch):
        """End-to-end: clear the cached PLS model so load_face_mesh_viz_model
        actually loads the real model (bundled v4, or HF Hub), then render a
        figure with a non-trivial AU activation. Verifies real predict + figure
        construction round-trips."""
        monkeypatch.setattr(plt_mod, "_PLS_MESH_MODELS", {})
        au = np.zeros(20, dtype=np.float32)
        # AU12 (smile) — index 9 in AU_LANDMARK_MAP['Feat']
        au[9] = 3.0
        fig = plot_face_mesh_plotly(au=au)
        # 5 traces: 1 mesh wireframe (dense tessellation by default) +
        # 4 iris/pupil Mesh3d disks. wireframe is data[0].
        assert len(fig.data) == 5
        assert fig.data[0].type == "scatter3d"
        assert len(fig.data[0].x) == 3 * N_TESSELATION_EDGES
        # Sanity: scene has finite axis ranges (no NaN propagation)
        z_range = fig.layout.scene.zaxis.range
        assert all(np.isfinite(z_range))
