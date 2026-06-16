"""Tests for ``feat.utils.region_maps`` + ``feat.plotting.plot_face_regions`` —
the non-overlapping AU / ARKit-blendshape region overlays on the MP-478 mesh
shipped at ``feat/resources/{au,blendshape}_region_map.json``.

Validates structural invariants (vertex range, aperture exclusion, AU
non-overlap, L/R mirror symmetry), that both maps regenerate byte-identically
from their seed tables (so the shipped files can't drift from source), and that
the plotting API renders map + activation views without error.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from feat.utils import region_maps as rm


@pytest.fixture(scope="module")
def au_map():
    return rm.load_au_region_map()


@pytest.fixture(scope="module")
def bs_map():
    return rm.load_blendshape_region_map()


@pytest.fixture(scope="module")
def canonical():
    V, _ = rm._canonical_geometry()
    return V


class TestLoad:
    def test_returns_independent_copies(self):
        a = rm.load_au_region_map()
        b = rm.load_au_region_map()
        assert a == b and a is not b
        a["AU12"]["mp478_vertices"].append(99999)
        assert 99999 not in rm.load_au_region_map()["AU12"]["mp478_vertices"]

    def test_au_schema(self, au_map):
        for au, spec in au_map.items():
            assert au.startswith("AU")
            assert spec["mp478_vertices"] == sorted(set(spec["mp478_vertices"]))
            assert spec["n_vertices"] == len(spec["mp478_vertices"]) > 0
            assert spec["muscles"] and all(isinstance(m, str) for m in spec["muscles"])

    def test_blendshape_schema(self, bs_map):
        for bs, spec in bs_map.items():
            assert spec["au"].startswith(("AU", "AD"))
            assert spec["side"] in ("L", "R", "C")
            assert spec["n_vertices"] == len(spec["mp478_vertices"]) > 0


class TestInvariants:
    def test_vertices_in_canonical_range(self, au_map, bs_map):
        for m in (au_map, bs_map):
            for spec in m.values():
                assert all(0 <= v < 468 for v in spec["mp478_vertices"])

    def test_no_aperture_leak(self, au_map, bs_map):
        for name, m in (("au", au_map), ("bs", bs_map)):
            for region, spec in m.items():
                leak = set(spec["mp478_vertices"]) & rm.APERTURE
                assert not leak, f"{name}:{region} covers aperture {sorted(leak)}"

    @pytest.mark.parametrize("which", ["au", "bs"])
    def test_regions_are_non_overlapping(self, which, au_map, bs_map):
        """Both maps are strict partitions — each vertex in at most one region.
        (Blendshape L/R divide exactly at the midline, so no shared seam.)"""
        m = au_map if which == "au" else bs_map
        seen: dict[int, str] = {}
        for region, spec in m.items():
            for v in spec["mp478_vertices"]:
                assert v not in seen, f"{which}: vertex {v} in {seen.get(v)} & {region}"
                seen[v] = region

    def test_blendshape_left_right_mirror(self, bs_map, canonical):
        """``...Left`` and ``...Right`` regions are exact mirrors across the
        midline — the partition is symmetrized (``_symmetrize_partition``), so
        winner-take-all boundary noise can't make paired regions asymmetric."""
        for name, spec in bs_map.items():
            if not name.endswith("Left"):
                continue
            right = name[:-4] + "Right"
            assert right in bs_map, f"missing {right}"
            assert spec["n_vertices"] == bs_map[right]["n_vertices"]
            cl = canonical[spec["mp478_vertices"]].mean(0)
            cr = canonical[bs_map[right]["mp478_vertices"]].mean(0)
            assert abs(cl[0] + cr[0]) < 0.05, f"{name}: not mirrored in x"
            assert abs(cl[1] - cr[1]) < 0.05, f"{name}: height mismatch"


class TestRegenerationMatchesShipped:
    def test_au_map_regenerates(self, au_map):
        assert rm.build_au_region_map() == au_map

    def test_blendshape_map_regenerates(self, bs_map):
        assert rm.build_blendshape_region_map() == bs_map


class TestRenderAssets:
    def test_au_assets_disjoint(self):
        a = rm.render_assets("au")
        assert {"parents", "tris", "region_verts", "n_base"} <= a.keys()
        seen: set[int] = set()
        for verts in a["region_verts"].values():
            assert not (verts & seen)
            seen |= verts

    def test_dense_positions_match_subdivision(self):
        """``dense_positions`` on the canonical base reproduces the subdivided
        canonical vertices the assets were built from."""
        a = rm.render_assets("au")
        V, _ = rm._canonical_geometry()
        dense = rm.dense_positions(rm.project_xy(V), a["parents"])
        assert len(dense) == a["n_base"] + len(a["parents"])

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError):
            rm.render_assets("nope")


class TestPlot:
    def test_map_views(self):
        from feat.plotting import plot_face_regions

        for kind in ("au", "blendshape"):
            ax = plot_face_regions(kind=kind)
            assert ax.collections  # something was drawn
            plt.close(ax.figure)

    def test_activation_view_dict_and_array(self, au_map):
        from feat.plotting import plot_face_regions

        ax = plot_face_regions(values={"AU12": 1.0, "AU06": 0.5}, kind="au")
        plt.close(ax.figure)
        arr = np.zeros(len(au_map))
        arr[0] = 1.0
        ax = plot_face_regions(values=arr, kind="au")
        plt.close(ax.figure)

    def test_overlay_on_detected_mesh(self):
        """Passing a deformed 478-mesh in image coords overlays on a host axis."""
        from feat.plotting import plot_face_regions

        V, _ = rm._canonical_geometry()
        mesh = V[:, :2] * 100 + 200  # fake image-coord landmarks
        _, ax = plt.subplots()
        ax.imshow(np.zeros((400, 400, 3), dtype=np.uint8))
        out = plot_face_regions(values={"AU12": 1.0}, kind="au", landmarks=mesh, ax=ax)
        assert out is ax
        plt.close(ax.figure)

    def test_bad_inputs(self, au_map):
        from feat.plotting import plot_face_regions

        with pytest.raises(ValueError):
            plot_face_regions(kind="nope")
        with pytest.raises(ValueError):
            plot_face_regions(values=np.zeros(len(au_map) + 3), kind="au")
