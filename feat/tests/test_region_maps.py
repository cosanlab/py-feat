"""Tests for ``feat.utils.region_maps`` + ``feat.plotting.plot_face_regions`` —
the non-overlapping AU / ARKit-blendshape region overlays on the MP-478 mesh
shipped at ``feat/resources/{au,blendshape}_region_map.json``.

The maps are a **per-triangle** partition of the bundled canonical tessellation
(geodesic-Voronoi seed partition, cleaned with boundary-length descent + L/R
symmetrization + speck removal). Validates structural invariants (vertex range,
aperture-hole exclusion, triangle non-overlap, L/R mirror symmetry), that both
maps regenerate byte-identically from their seed tables (so the shipped files
can't drift from source), and that the plotting API renders map + activation
views without error.
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


@pytest.fixture(scope="module")
def holes():
    V, tris = rm._canonical_geometry()
    aperture = {a for a in rm.APERTURE if a < len(V)}
    return rm.aperture_triangles(tris, aperture)


class TestLoad:
    def test_returns_independent_copies(self):
        a = rm.load_au_region_map()
        b = rm.load_au_region_map()
        assert a == b and a is not b
        a["AU12"]["triangles"].append(99999)
        assert 99999 not in rm.load_au_region_map()["AU12"]["triangles"]

    def test_au_schema(self, au_map):
        for au, spec in au_map.items():
            assert au.startswith("AU")
            assert spec["triangles"] == sorted(set(spec["triangles"]))
            assert spec["n_triangles"] == len(spec["triangles"]) > 0
            assert spec["mp478_vertices"] == sorted(set(spec["mp478_vertices"]))
            assert spec["n_vertices"] == len(spec["mp478_vertices"]) > 0
            assert spec["muscles"] and all(isinstance(m, str) for m in spec["muscles"])

    def test_blendshape_schema(self, bs_map):
        for bs, spec in bs_map.items():
            assert spec["au"].startswith(("AU", "AD"))
            assert spec["side"] in ("L", "R", "C")
            assert spec["n_triangles"] == len(spec["triangles"]) > 0
            assert spec["n_vertices"] == len(spec["mp478_vertices"]) > 0


class TestInvariants:
    def test_vertices_in_canonical_range(self, au_map, bs_map):
        for m in (au_map, bs_map):
            for spec in m.values():
                assert all(0 <= v < 468 for v in spec["mp478_vertices"])

    def test_triangles_in_range(self, au_map, bs_map):
        _, tris = rm._canonical_geometry()
        for m in (au_map, bs_map):
            for spec in m.values():
                assert all(0 <= t < len(tris) for t in spec["triangles"])

    def test_no_aperture_hole_coverage(self, au_map, bs_map, holes):
        """No region may cover an eye/mouth *opening* (a triangle whose 3 verts
        are all on an aperture ring). Boundary triangles touching a ring are
        fine; only the holes are forbidden."""
        for name, m in (("au", au_map), ("bs", bs_map)):
            for region, spec in m.items():
                leak = set(spec["triangles"]) & holes
                assert not leak, f"{name}:{region} covers aperture hole {sorted(leak)}"

    @pytest.mark.parametrize("which", ["au", "bs"])
    def test_regions_are_non_overlapping(self, which, au_map, bs_map):
        """Both maps are strict per-triangle partitions — each tessellation
        triangle belongs to at most one region. (Boundary *vertices* are shared
        between adjacent regions; the triangles are not.)"""
        m = au_map if which == "au" else bs_map
        seen: dict[int, str] = {}
        for region, spec in m.items():
            for t in spec["triangles"]:
                assert t not in seen, f"{which}: triangle {t} in {seen.get(t)} & {region}"
                seen[t] = region

    def test_blendshape_left_right_mirror(self, bs_map, canonical):
        """``...Left`` and ``...Right`` regions are exact mirrors across the
        midline — the per-triangle partition is symmetrized, so winner-take-all
        boundary noise can't make paired regions asymmetric."""
        for name, spec in bs_map.items():
            if not name.endswith("Left"):
                continue
            right = name[:-4] + "Right"
            assert right in bs_map, f"missing {right}"
            assert spec["n_triangles"] == bs_map[right]["n_triangles"]
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
    def test_au_assets_triangles_disjoint(self):
        a = rm.render_assets("au")
        assert {"tris", "region_tris", "n_base"} <= a.keys()
        seen: set[int] = set()
        for idx in a["region_tris"].values():
            s = set(idx)
            assert not (s & seen)
            seen |= s

    def test_assets_match_loaded_map(self):
        a = rm.render_assets("blendshape")
        m = rm.load_blendshape_region_map()
        for region, spec in m.items():
            assert sorted(a["region_tris"].get(region, [])) == spec["triangles"]

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
        """Passing a deformed mesh in image coords overlays on a host axis."""
        from feat.plotting import plot_face_regions

        V, _ = rm._canonical_geometry()
        mesh = V[:, :2] * 100 + 200  # fake image-coord landmarks
        _, ax = plt.subplots()
        ax.imshow(np.zeros((400, 400, 3), dtype=np.uint8))
        out = plot_face_regions(values={"AU12": 1.0}, kind="au", landmarks=mesh, ax=ax)
        assert out is ax
        plt.close(ax.figure)

    def test_overlay_on_478_mesh_matches_468(self):
        """A real 478-vertex MediaPipe mesh (10 trailing iris verts) must render
        the same overlay as its 468-vertex face slice — the tessellation indexes
        the 468 face verts, so the extra iris rows must be trimmed, not shift the
        triangle vertex lookups."""
        from feat.plotting import plot_face_regions
        from matplotlib.collections import PolyCollection

        V, _ = rm._canonical_geometry()
        mesh468 = V[:, :2] * 100 + 200
        mesh478 = np.vstack([mesh468, mesh468[:10] + 1234.0])  # bogus iris rows

        def polys(mesh):
            _, ax = plt.subplots()
            plot_face_regions(values={"AU12": 1.0, "AU06": 0.6}, kind="au",
                              landmarks=mesh, ax=ax)
            out = [p for c in ax.collections
                   if isinstance(c, PolyCollection) for p in c.get_paths()]
            plt.close(ax.figure)
            return out

        p468, p478 = polys(mesh468), polys(mesh478)
        assert len(p468) == len(p478) and len(p468) > 0
        for a, b in zip(p468, p478):
            assert np.allclose(a.vertices, b.vertices)

    def test_bad_inputs(self, au_map):
        from feat.plotting import plot_face_regions

        with pytest.raises(ValueError):
            plot_face_regions(kind="nope")
        with pytest.raises(ValueError):
            plot_face_regions(values=np.zeros(len(au_map) + 3), kind="au")
