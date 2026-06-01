"""Tests for ``feat.utils.muscle_to_landmark`` — the facial-muscle → MP-478
mesh / AU map shipped at ``feat/resources/muscle_to_mesh_map.json``.

Validates the bundled map's structural invariants (vertex range, aperture
exclusion, L/R symmetry) and that it regenerates byte-identically from the
generator script on py-feat's own canonical mesh — so the shipped file can't
silently drift from its source.
"""

from __future__ import annotations

import pytest

from feat.utils import muscle_to_landmark as m2l
from feat.utils.mp_plotting import FaceLandmarksConnections as F


def _group_verts(name):
    return {v for c in getattr(F, name) for v in (c.start, c.end)}


# Eye-lid + inner-lip aperture rings — no muscle region may cover these (the
# openings themselves). Eye rings are pulled from MediaPipe's canonical groups
# (the authoritative source the generator also uses); the inner-lip ring has no
# dedicated group so it's listed explicitly.
APERTURE = (
    _group_verts("FACE_LANDMARKS_RIGHT_EYE")
    | _group_verts("FACE_LANDMARKS_LEFT_EYE")
    | {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317,
       402, 318, 324}
)


@pytest.fixture(scope="module")
def muscle_map():
    return m2l.load_muscle_to_landmark_map()


class TestLoad:
    def test_loads_and_returns_independent_copies(self):
        a = m2l.load_muscle_to_landmark_map()
        b = m2l.load_muscle_to_landmark_map()
        assert a == b
        assert a is not b  # fresh copy each call — caller mutation can't leak
        a["procerus"]["mp478_vertices"].append(99999)
        assert 99999 not in m2l.load_muscle_to_landmark_map()["procerus"]["mp478_vertices"]
        assert len(a) == 37

    def test_entry_schema(self, muscle_map):
        for name, spec in muscle_map.items():
            assert spec["au"].startswith("AU")
            assert isinstance(spec["mp478_vertices"], list)
            assert spec["n_vertices"] == len(spec["mp478_vertices"])
            assert spec["n_vertices"] > 0, f"{name} has an empty region"
            assert spec["mp478_vertices"] == sorted(set(spec["mp478_vertices"]))


class TestInvariants:
    def test_vertices_in_canonical_range(self, muscle_map):
        for spec in muscle_map.values():
            for v in spec["mp478_vertices"]:
                assert 0 <= v < 468

    def test_no_aperture_leak(self, muscle_map):
        """No muscle region may cover an eye or mouth opening."""
        for name, spec in muscle_map.items():
            leak = set(spec["mp478_vertices"]) & APERTURE
            assert not leak, f"{name} covers aperture verts {sorted(leak)}"

    def test_left_right_symmetric(self, muscle_map):
        """Paired _L/_R muscles should be mirror images across the midline."""
        from feat.utils.face_pose import load_canonical_face_model

        V = load_canonical_face_model().detach().cpu().numpy()
        for name, spec in muscle_map.items():
            if not name.endswith("_L"):
                continue
            r = name[:-2] + "_R"
            assert r in muscle_map, f"missing right partner for {name}"
            cl = V[spec["mp478_vertices"]].mean(0)
            cr = V[muscle_map[r]["mp478_vertices"]].mean(0)
            # observed worst residual is x=0.09, y=0.23 on a 15.5-unit span;
            # 0.5 catches a >2x asymmetry without being flaky.
            assert abs(cl[0] + cr[0]) < 0.5   # x mirrors across midline
            assert abs(cl[1] - cr[1]) < 0.5   # y (height) matches


class TestViews:
    def test_muscle_vertices(self, muscle_map):
        v = m2l.muscle_vertices("zygomaticus_major_L")
        assert v == muscle_map["zygomaticus_major_L"]["mp478_vertices"]

    def test_au_to_muscle_vertices_merges_sides(self, muscle_map):
        by_au = m2l.au_to_muscle_vertices()
        # AU12 (smile) is carried by zygomaticus_major L+R — merged set should
        # be the union of both, deduped and sorted.
        expected = sorted(
            set(muscle_map["zygomaticus_major_L"]["mp478_vertices"])
            | set(muscle_map["zygomaticus_major_R"]["mp478_vertices"])
        )
        assert by_au["AU12"] == expected


class TestRegenerationMatchesShipped:
    def test_generator_reproduces_shipped_map(self, muscle_map):
        """The shipped JSON must equal what the generator produces from the
        bundled canonical mesh — guards against the file drifting from source.

        Source-checkout only: ``scripts/`` is not part of the installed package,
        so this is skipped (not failed) when running against a wheel.
        """
        import importlib.util
        import os

        script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                              "map_muscles_canonical_geodesic.py")
        if not os.path.exists(script):
            pytest.skip("generator script not present (installed package)")

        spec = importlib.util.spec_from_file_location("_gen", script)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        V, faces = gen.load_geometry()
        regenerated, _ = gen.build_map(V, faces)
        # full per-muscle equality (au, seed, vertices, n_vertices, hops) — the
        # generator's build_map output is exactly the shipped "muscles" payload.
        assert regenerated == muscle_map
