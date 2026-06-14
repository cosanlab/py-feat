# py-feat — agent / contributor notes

Project-specific conventions and gotchas that aren't obvious from the code or
the README. Read this before starting work in the repo. Update it when you
discover something durable; don't repeat what's already in `README.md` or
`docs/`.

## Branch state (as of v0.7.0 prep)

- **`main`** intentionally points at the **v0.6.x state** (`c4f6364`). Do NOT
  push v0.7-era changes to main; we hold main back so existing users on
  `pip install py-feat` don't get the breaking changes until we explicitly
  ship to PyPI.
- **`v0.7-dev`** is the active dev branch. Open PRs against `v0.7-dev`.
- **`v0.7.0` tag** exists at `b8ac8b6` (the merge commit produced when
  v0.7-dev was briefly merged to main during pre-release testing, then
  rolled back). The tag stays in place so testers can use
  `pip install git+https://github.com/cosanlab/py-feat.git@v0.7.0`.
  **Do NOT re-tag v0.7.0** — it would break those testers' URLs.

## Load-bearing things you'd be tempted to remove

### `OMP_NUM_THREADS=1` in `feat/__init__.py`

This block is REQUIRED on Python 3.13 + macOS. xgboost and torch link
against different `libomp`/`libiomp5` dylibs on PyPI; both get `dlopen`'d at
different addresses, their thread-pool TLS state diverges, and one runtime
segfaults inside `__kmp_*` routines once both are exercised. The symptom is
`Detectorv1(au_model='xgb')` exiting 139 with no Python traceback.

Setting `OMP_NUM_THREADS=1` (only when not already set) sidesteps the
divergent state. Performance impact is minor (xgboost AU classifiers are
small; torch heavy ops use MPS/CUDA/Accelerate which parallelize
independently). Bibliography:

- https://github.com/dmlc/xgboost/issues/11500
- https://github.com/pytorch/pytorch/issues/44282
- https://github.com/microsoft/LightGBM/issues/6595

`KMP_DUPLICATE_LIB_OK=TRUE` was tried — it does NOT fix the segfault
(only suppresses a startup check; PyTorch warns it can corrupt results).

If you remove the OMP block, verify `Detectorv1(au_model='xgb').detect(...)`
runs end-to-end on Python 3.13 + macOS before merging.

### `_patch_xgboost_setstate_for_skops` in `feat/detector.py`

A defensive bytearray-lifetime fix for a *separate* Python 3.13 race in
`xgboost.Booster.__setstate__`. Best-effort; the upstream race isn't fully
closed. Leave it in.

### `Fex.__init__` no longer has the per-column metadata loop

PR #283 removed the "Kludgy solution" loop that set `sampling_freq` /
`sessions` on every Series. `_constructor_sliced` + `__finalize__` already
propagates `_metadata` (see `feat/data.py`). Don't reintroduce the loop —
it's a 1000× speedup on construction.

## API conventions

### `face_model` kwarg

`'retinaface_r34'` was dropped in v0.7.0. Only `'retinaface'` is accepted.
`Detectorv1` / `MPDetector` validate the `face_model` kwarg; `Detectorv2` takes
no `face_model` kwarg at all (it hardcodes `'retinaface'`). The HuggingFace repo
is still named `py-feat/retinaface_r34` (we kept the backbone-tagged storage
name) — the two are intentionally different.

### `identity_model` default

`'arcface'` (was `'facenet'` through v0.6). FaceNet is still available via
explicit `identity_model='facenet'`. The ArcFace weights are
non-commercial-research only — flag this if a user asks about commercial
deployment. Model card at `feat/identity_detectors/arcface/MODEL_CARD.md`.

### `num_workers=0` default

Per `docs/benchmarks/2026-05-03-437b651.md`, `num_workers > 0` is uniformly
slower than `num_workers=0` on Apple Silicon + Python 3.13 with our
`OMP_NUM_THREADS=1` (worst case 33× slower for image batches). Don't bump
the default. If a user reports "DataLoader is slow," it's typically because
they've set `num_workers=N` themselves.

## Workflow conventions

### Commit messages — no AI attribution

**Do NOT add `Co-Authored-By: Claude ...` (or any AI attribution) to commit
messages, PR descriptions, or release notes.** The commit author is the
human who reviewed and pushed it. Don't add `🤖 Generated with...` footers
either. If you must reference tooling, do it in a code comment in the
relevant file, not in the git history.

### Code-review then merge

The user typically wants `superpowers:code-reviewer` run on each PR before
merging. Don't merge straight from "looks good to me" — spawn the reviewer
agent and address blockers first.

### Merge commits, not squash

`gh pr merge --merge --delete-branch` is the standard. See
`git log --merges` for the established style. Don't `--squash`.

### Benchmark tracking

`docs/benchmarks/<YYYY-MM-DD>-<git-sha>.md` per run. Generate with
`python scripts/bench_detectors.py --markdown` (defaults handle path).
Format is fixed by the script — don't hand-write deviating formats; if you
need narrative on top of a generated table, append it after a `---`
separator and document the dual-source pattern (see `docs/benchmarks/README.md`).

### Code style

- Prefer no comments unless the WHY is non-obvious. Variable names + tests
  carry the WHAT.
- No multi-paragraph docstrings on internal methods. Top-level public API
  gets fuller docs.
- Fix bugs by understanding root cause, not by adding fallback try/except
  scaffolding around the symptom.
- See `feat/__init__.py`'s `OMP_NUM_THREADS` block for the rough comment
  density that's appropriate when there's real load-bearing context to
  preserve.

## Detectorv2 chip extraction & head pose (v0.8 redesign in flight)

### Head-pose convention (all detectors, RADIANS)

Canonical convention across every detector: **+pitch = look up, +yaw = turn to
the subject's right, +roll = tilt to the subject's right**, in **radians** (Fex
columns are radians everywhere — img2pose dofs, Pose-MLP, mesh Umeyama, OpenFace
imports). Consumers convert to degrees for display only.

- **Classic `Detectorv1`** (img2pose + Pose-MLP, which share img2pose's frame):
  normalized at the `feat_poses` assembly — `Pitch=-Pitch_raw, Yaw=+Roll_raw,
  Roll=-Yaw_raw`.
- **`Detectorv2`** (multitask head): mapped at the Fex assembly —
  `Pitch=-head[0], Roll=-head[2], Yaw=+head[1]`. NOTE the head's index 0 tracks
  PITCH and index 1 tracks YAW (the inference docstring's `[yaw,pitch,…]` order
  is mislabeled) — verified on-camera, don't trust the docstring order.
- **PnP-DLT removed** (geometrically unreliable, heavy cross-axis bleed).
  retinaface always uses the Pose-MLP; `facepose_model` options are
  `["pose_mlp", "img2pose"]` only; pose stays NaN if MLP weights are missing.
- Calibrate pose signs by reading **raw numeric Fex values**, never by eyeballing
  an overlay against a selfie-mirrored video (the mirror flips left/right and
  will send you in circles).

### v2 chip geometry — RESOLVED for v2.5 (isotropic square-pad, inference fix)

History (kept because it explains the code): early `Detectorv2` pitch read
~0.4–0.6× of img2pose and its 478-mesh z was mis-scaled, because
**`extract_face_from_bbox_torch`** squashes the rectangular RetinaFace box into a
*square* chip with independent x/y scales (plus an edge `clamp` instead of pad).
That's invisible to v1 (2D landmarks un-squish exactly) but corrupts v2's
chip-space 3D/angle inference. Design spec:
`docs/superpowers/specs/2026-06-06-v2-chip-extraction-redesign-design.md`.

**This is now fixed.** The shipped v2.5 weights (`py-feat/face_multitask_v2`)
were already **trained on the isotropic square-pad chip** (au_deep
`extract_chips_unified.py --v25` → `extract_face_square_pad_torch`,
`side=max(W,H)·1.2`, reflection-pad, no clamp; mesh targets `[0,1]` over that
square chip). So no retrain is needed — only the inference path had to match. As
of the v25-square-pad inference fix, `Detectorv2.detect_faces` /
`crop_faces_from_boxes` use `extract_face_square_pad_torch` and store the square
crop region as `new_boxes`, so `_mesh_to_original_frame`'s `[0,1]→box` decode is
isotropic (`w==h==side`). This removed the 2D mesh-overlay shift (eyes-high /
mouth-low) and the pitch compression in one change.

`extract_face_from_bbox_torch` is unchanged and **still used by `Detectorv1` and
`MPDetector`** — their landmark heads invert that squish exactly, so do NOT
switch them to square-pad. The square-pad crop is v2-only.

## Repo-specific gotchas

- **`mp_facemesh_v2` legacy file**: `face_landmarks_detector_Nx3x256x256_onnx.pth`
  on HuggingFace is an `onnx2torch`-converted FX GraphModule that requires
  `weights_only=False` (arbitrary code execution path). py-feat 0.7+ tries
  the safe TorchScript file first and only falls back to the legacy with a
  warning. **The legacy file is slated for deletion from HF shortly after
  v0.7.0 ships** (announced in changelog).
- **xgb AU classifier files**: v1 + v2 `.skops` both shipped to HF. py-feat
  tries v2 first (modern UBJ format, dodges a Python 3.13 segfault path)
  and falls back to v1. Don't remove the fallback until v0.7's at-or-above
  is the floor for users.
- **HF write tokens**: live in a gitignored `.env` (see `.env.example`-style
  format inline). Agents typically don't have write access; the user runs
  `huggingface-cli upload` themselves or pastes a temporary token into
  `.env` for one-session use.
- **Identity clustering quality**: known to be mediocre — connected-components
  on cosine-thresholded embeddings, brittle to threshold choice. The
  embedding upgrade to ArcFace helped, but the clustering algorithm is
  still v0.6-era. Roadmap at
  `docs/superpowers/specs/2026-05-02-identity-detection-roadmap.md`
  (HDBSCAN, gallery-aware, tracker-based).

## Roadmap (post-v0.7.0)

- Identity-clustering rewrite (v0.8) — see roadmap doc above
- `invert_padding_to_results` formal deprecation (warn in 0.8, remove in 0.9)
- Labeled-video benchmark for identity F1 (PR-0 of the v0.8 identity track)

## Pre-existing project conventions

- Python 3.11+ required (PR #262)
- Test data lives in `feat/tests/data/`; long-video fixture is
  `WolfgangLanger_Pexels.mp4` (472 frames, 1 face/frame); image fixture is
  `multi_face.jpg` (5 faces).
- All HF models live under the `py-feat/` org. Lazy-download to
  `feat/resources/` cache.
