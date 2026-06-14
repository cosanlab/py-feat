# Detector / MPDetector Unification — Design

**Status**: draft, awaiting maintainer review.
**Target milestone**: v0.8 (will not ship in v0.7).
**Approach chosen**: Option A — template-method subclass.

## Problem

`feat/detector.py::Detector` (966 lines) and `feat/MPDetector.py::MPDetector`
(968 lines) are parallel implementations of the same multi-stage face
analysis pipeline. They share:

- Identical `nn.Module + PyTorchModelHubMixin` parent classes.
- Near-identical `__init__` skeleton: validate args → build `info` dict →
  `set_torch_device` → cache `HOGLayer` → load face/landmark/au/emotion/
  identity detectors → register output column metadata.
- Near-identical `forward(faces_data, batch_data)` shape: extract face
  crops → run landmark detector → run AU detector → run emotion detector
  → run identity detector → invert padding → assemble per-face DataFrame.
- 95% identical `detect()` public method (only difference: `Detector`
  exposes `save=None`).

The duplication has real costs:

1. Adding a new face detector or landmark model requires touching both
   classes and reasoning about whether the change applies to one or both
   pipelines.
2. Bug fixes drift between classes (e.g., the `MPDetector(emotion_model='svm')`
   path was broken pre-PR #292; was caught and patched only in MPDetector
   while the equivalent codepath in Detector worked).
3. Test surface doubles: `test_detector.py` + `test_mpdetector_retinaface.py`
   exercise overlapping ground.
4. Refactors (e.g., PR #292's HOG batching) need to land in two files for
   one logical change.

## Goal

A shared `BaseDetector` parent class that contains the common pipeline
orchestration. `Detector` and `MPDetector` become thin subclasses that
override only the stages where they differ, while the public API and
import surface remain unchanged.

## Non-goals

- **No public API breakage.** `from feat import Detector`, `from
  feat.MPDetector import MPDetector`, and the `detect()` signature stay
  the same.
- **No model behavior change.** Pipeline output is bit-identical to
  v0.7-dev for any input the existing classes support.
- **No new feature.** This is purely structural.
- **Not bundling the `feat/data.py` split** with this work. `data.py` is
  a separate refactor item.
- **Not bundling the `detect()` 11-param signature reduction** (separate
  v0.8 item; the parameter list moves verbatim into `BaseDetector`).

## Proposed structure

```
feat/
  detectors/
    __init__.py              # re-exports Detector, MPDetector
    base.py                  # BaseDetector — orchestration + hooks
    classic.py               # Detector(BaseDetector) — img2pose/retinaface
                             # + mobilefacenet/mobilenet + xgb/svm AU +
                             # resmasknet/svm emotion + arcface/facenet
                             # identity
    mediapipe.py             # MPDetector(BaseDetector) — retinaface +
                             # mp_facemesh_v2 + mp_blendshapes + ...
```

`feat/detector.py` and `feat/MPDetector.py` become 1-line shims that
re-export from the new module so existing import paths continue to work
(deprecated as locations but never removed in v0.x).

## BaseDetector responsibilities

```python
class BaseDetector(nn.Module, PyTorchModelHubMixin):
    # ---- structural attributes set by subclass __init__ ----
    info: dict                     # face_model, landmark_model, etc.
    device: torch.device
    face_size: int
    face_detector: nn.Module | None
    landmark_detector: nn.Module | None
    au_detector: nn.Module | None
    emotion_detector: nn.Module | None
    identity_detector: nn.Module | None
    facepose_detector: nn.Module | None
    _hog_layer: HOGLayer            # cached, may be unused depending on AU choice

    # ---- shared concrete methods ----
    def detect(self, inputs, data_type, batch_size, ...) -> Fex:
        # full public-API method, single implementation
    def forward(self, faces_data, batch_data) -> pd.DataFrame:
        # orchestrates the pipeline by calling hooks below
    def _setup_hog_layer(self): ...

    # ---- abstract / overridable hooks ----
    def _detect_landmarks(self, faces_dev, faces_data) -> torch.Tensor:
        raise NotImplementedError
    def _detect_aus(self, faces_data, landmarks, faces_dev) -> torch.Tensor:
        raise NotImplementedError
    def _detect_emotions(self, faces_data, landmarks, faces_dev) -> torch.Tensor:
        raise NotImplementedError
    def _detect_identities(self, faces_dev) -> torch.Tensor:
        raise NotImplementedError
    def _detect_facepose(self, faces_data, landmarks) -> torch.Tensor:
        raise NotImplementedError
    def _build_fex(self, ...) -> Fex:
        raise NotImplementedError
```

## Hook signatures and stage list

Concrete stages, in order, with the data shape each receives and emits:

| Stage | Input | Output | Differences across subclasses |
|---|---|---|---|
| `_detect_landmarks` | `faces_dev: [N, C, fs, fs]` | `[N, n_lm * 2]` (Detector) or `[N, 478, 3]` (MPDetector) | OpenFace 68 vs MediaPipe 478; mobilefacenet vs mp_facemesh_v2 |
| `_detect_aus` | crops + landmarks | `[N, n_aus]` | xgb/svm on HOG (Detector) vs mp_blendshapes on landmark subset (MPDetector) |
| `_detect_emotions` | crops + landmarks + faces_data | `[N, n_emotions]` | resmasknet/svm path uniform; emotion is fully shareable |
| `_detect_identities` | crops | `[N, 512]` | arcface or facenet (uniform); largely shareable |
| `_detect_facepose` | landmarks (+ optional faces_data for img2pose's regressed pose) | `[N, 6]` (6DoF) | img2pose regresses; retinaface uses pnp_dlt; MPDetector uses mesh alignment via Umeyama |
| `_build_fex` | per-stage outputs | `pd.DataFrame` | Different column sets (FEAT_FACEPOSE_COLUMNS_6D shared, but landmark column count differs and `gaze_columns` is MPDetector-only) |

## Differences between Detector and MPDetector at each stage

| Stage | Detector | MPDetector |
|---|---|---|
| Face detection | `img2pose` or `retinaface` | `retinaface` only |
| Landmark detection | `mobilefacenet` (default) or `mobilenet` → 68 landmarks normalized [0, 1] flattened to `[N, 136]` | `mp_facemesh_v2` → 478 landmarks 3D in face-crop pixel space `[N, 478, 3]` |
| Landmark scale buffer | not needed | `_landmark_scale: [1, 1, 2]` registered buffer (added in PR #292) |
| AU classifier | xgb or svm on HOG features (via `extract_hog_features_batched`) | mp_blendshapes on a 146-landmark subset of mediapipe mesh |
| Emotion classifier | resmasknet (default) or svm-on-HOG | resmasknet (default), svm path is `NotImplementedError` per #294 |
| Identity classifier | arcface (default) or facenet, returns 512-d embedding | arcface (default), same |
| Facepose | img2pose's native head — or DLT-PnP from 68 landmarks for retinaface face_model | Umeyama alignment of 478 landmarks against canonical face model |
| Output columns | `openface_2d_landmark_columns` (136 columns) | `MP_LANDMARK_COLUMNS` (1434 columns), plus `FEAT_GAZE_COLUMNS` |
| Output Fex `_metadata` | no `gaze_columns` set (None) | `gaze_columns=FEAT_GAZE_COLUMNS` set |

## What lives in BaseDetector vs. subclass

**Concrete in BaseDetector** (single implementation):
- Argument validation infrastructure (`_validate_face_model`, etc. as helpers).
- `set_torch_device` + `_hog_layer` caching.
- `forward()` orchestration (the call sequence + per-frame padding inversion).
- `detect()` end-to-end (data loader, batch loop, save/CSV path, identity threshold, frame skipping, progress bar).
- `_apply_padding_inversion` helper (already shared logic via `per_face_padding_inversion_terms`).
- Identity-clustering call after detection (`compute_identities`).

**Hook overrides in subclass** (where they differ):
- `_detect_landmarks` (different output shape + different model wrapper).
- `_detect_aus` (different model + different input requirements — HOG vs landmark subset).
- `_detect_facepose` (img2pose-native vs PnP vs Umeyama).
- `_build_fex` (different column sets).

**Same in subclass but worth a default in base**:
- `_detect_emotions` — identical resmasknet path; provide it in base, MPDetector doesn't override.
- `_detect_identities` — identical arcface/facenet path; provide in base.

## Migration plan

### Step 1 — preparation (lands first, on its own)

- Move `extract_hog_features_batched` consumers to a shared mixin or
  module-level helper, no API change. Already mostly done in PR #292.
- Lock down `feat/MPDetector.py::forward()` and `feat/detector.py::forward()`
  with end-to-end snapshot tests on `single_face.jpg` and
  `multi_face.jpg`. Inputs go in, output Fex columns + AU values + pose
  + identity get pickled and committed as fixtures. Any post-unification
  forward() must reproduce these.
- Adds confidence that the structural change doesn't perturb numerics.

### Step 2 — base extraction (own PR)

- Create `feat/detectors/base.py` with `BaseDetector` skeleton
  containing only the abstract hooks + `__init__` skeleton helpers +
  `detect()` (verbatim move from `Detector.detect()`).
- `Detector` and `MPDetector` continue to live in their existing files
  but each becomes a `BaseDetector` subclass that overrides every hook
  with its current inline code (no logic change yet — just structural).
- `forward()` becomes the orchestrator in `BaseDetector`; subclasses
  drop their custom `forward()`.
- Snapshot tests from Step 1 must pass unchanged.

### Step 3 — hoist shared hooks (own PR)

- Identify hooks where Detector and MPDetector implementations are now
  identical (post-step-2 structural extraction): emotion, identity,
  Fex assembly minus column choice.
- Hoist those into `BaseDetector` as default impls.
- Subclasses override only the truly-different hooks.

### Step 4 — relocate to new module (optional, deferred to v0.8 if Step 3 isn't enough)

- Move `Detector` → `feat/detectors/classic.py` and `MPDetector` →
  `feat/detectors/mediapipe.py`; old paths become 1-line shims.
- Marks the cleanup as complete.

Each step is a separately reviewable PR. If a step fails snapshot
parity, it gets reverted without touching the others.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subtle numeric drift introduced by routing through new method dispatch | Snapshot tests in Step 1 catch any output change; CI gate at Step 2/3 |
| `PyTorchModelHubMixin` may not handle hierarchy cleanly | Verify in Step 2 that `Detector.from_pretrained` still works; if not, register both subclasses with the mixin |
| Users subclassing `Detector` for custom pipelines | Inheritance still works — subclasses can override the same hooks. Document the new extension story. |
| `_hog_layer` caching for MPDetector that doesn't use HOG | Already lazy-built in PR #292. Move construction to `BaseDetector.__init__` and let MPDetector hold a reference it never uses; or build it lazily on first AU call. |
| Tests in `test_detector.py` and `test_mpdetector_retinaface.py` overlap | Keep both files; they exercise different model stacks. Consider a shared `test_base_detector.py` for the new base in v0.8.x. |

## Open questions

1. Should `_detect_facepose` be a unified hook with three implementations
   (img2pose-native / pnp-dlt / mediapipe-umeyama), or live as separate
   subclass-specific helpers? Leaning unified — same return shape `[N, 6]`,
   different math.
2. Should `_build_fex` return a `dict` of per-stage tensors that
   `BaseDetector` assembles into a Fex, or directly return a Fex? Leaning
   the former — it lets BaseDetector centralize the column-binding logic
   and just take a `landmark_columns` kwarg from the subclass.
3. Does `compute_identities` belong in `detect()` (where it currently is)
   or should it become a post-processing hook? Leaning leave-as-is; the
   identity threshold is a runtime parameter, not an architectural one.
4. New module layout (`feat/detectors/`) vs keeping the existing
   `feat/detector.py` + `feat/MPDetector.py` filenames with shared code in
   a separate file (`feat/_detector_base.py`)? Leaning the new module —
   cleaner long-term — but adds a small amount of import-path churn that
   the v0.8 release notes need to call out.

## Out of scope (separate workstreams)

- `feat/data.py` 2648-line split.
- `detect()` 11-parameter signature reduction (probably wrap in a
  `DetectConfig` dataclass; that's a v0.8 deprecation cycle of its own).
- MediaPipe 478 → OpenFace 68 landmark translator (issue #294).
- Stage 2 of HOG (issue #293).
- Identity-clustering rewrite (separate roadmap doc).

## What this unlocks

- Adding a new pipeline (e.g., a future `RealtimeDetector` for video
  streams) becomes one new subclass file overriding the hooks that
  differ from the closest existing subclass.
- Bug fixes land in one place when they apply to both pipelines.
- Test infrastructure for the orchestration logic can live in one place.
- The `detect()` signature reduction work becomes a one-file change.

## Estimated effort

| Step | Estimate |
|---|---|
| 1 — snapshot tests | half day |
| 2 — base extraction (no logic change) | full day |
| 3 — hoist shared hooks | half day |
| 4 — relocate (optional) | quarter day |
| Total | 2–3 days of focused work, spread across 4 PRs |
