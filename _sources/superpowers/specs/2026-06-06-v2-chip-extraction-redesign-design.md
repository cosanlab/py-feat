# Detectorv2 Chip-Extraction Redesign — Design

**Date:** 2026-06-06
**Status:** Draft — investigation complete, open decisions pending (see §8)
**Scope:** `Detectorv2` (the ConvNeXt-V2 multitask model) only. v1 `Detector` and
`MPDetector` are out of scope and keep the legacy crop.

> This is a retrain-from-scratch change. It is written to be pulled onto the
> training server and reconciled against the actual training/extraction code —
> the open decisions in §8 need answers that only the training pipeline has.

## 1. Goal

RetinaFace finds faces (multiple per image allowed) and emits a batch of face
chips; the multitask ConvNeXt-V2 model consumes that batch and predicts every
head we ship (AU, emotion, valence/arousal, gaze, head pose, 478-mesh,
landmarks, identity). The change: replace the **anisotropic** chip crop with an
**aspect-preserving** one so the model never sees a geometrically distorted
face — fixing head-pose pitch and the 478-mesh depth axis, and (expected)
improving every head, especially on non-frontal and frame-edge faces.

## 2. The bug

`feat/utils/image_operations.py:extract_face_from_bbox_torch` resamples the
RetinaFace box into a square chip with **independent x and y scales**:

```python
width  = (x2 - x1) * expand_bbox      # rectangular RetinaFace box (h ≠ w)
height = (y2 - y1) * expand_bbox
new_x1 = (cx - width/2).clamp(min=0)  # edge CLAMP, not pad
new_y1 = (cy - height/2).clamp(min=0)
...
grid_x spans new_x1..new_x2           # → 256 square, x scaled by 256/width
grid_y spans new_y1..new_y2           # → 256 square, y scaled by 256/height
```

Two distinct distortions, both confirmed (there is **no** squaring/aspect logic
anywhere in the crop path):

1. **Anisotropic resize.** A rectangular box → square chip with two different
   scale factors. The face is stretched/compressed vertically by `height/width`.
2. **Edge clamp.** Off-frame boxes get `clamp`'d on one side only, shifting the
   face off-center and adding *more* aspect distortion near frame edges (the
   "mesh compresses off-frame" symptom).

### Why it's invisible to v1 but breaks v2

The crop util was written for v1's **2D landmark** detector, where the squish is
*invertible*: `inverse_transform_landmarks_torch` un-squishes x and y back to
frame coordinates exactly, and v1's pose comes from those correct 2D landmarks
(Pose-MLP). v2 instead infers **3D structure and angle directly from the chip**
(pose head, 478-mesh with depth), so the erased vertical foreshortening cannot
be recovered.

### Measured impact (on-camera, v2 vs img2pose/pose_mlp reference)

| gesture | v2 Pitch | v1 (pose_mlp) Pitch | ratio |
|---|---|---|---|
| chin up   | 14  | 38  | 0.37× |
| chin down | −15 | −24 | 0.63× |

Pitch is compressed to ~0.4–0.6×; yaw and roll are *not* (they're horizontal /
in-plane, far less sensitive to the vertical squish). The compression is
nonlinear (worse at extremes) and varies with box aspect, so no constant gain
corrects it.

### Compounding mesh-z bug

`feat/detector_v2.py:_mesh_to_original_frame` maps the 478-mesh from chip space
to frame coords but maps **x by box width, y by box height, and passes z through
raw** (in 224-chip units):

```python
x = (xy01[:,:,0] * w + left - pad_left) / scale
y = (xy01[:,:,1] * h + top  - pad_top ) / scale
return torch.stack([x, y, mesh[:,:,2]], dim=-1)   # z NOT rescaled
```

So the output mesh has x/y in frame pixels but z in a different (and per-face
variable) scale — an anisotropic x,y-vs-z mismatch a similarity/Umeyama fit (or
a downstream AU→landmark Procrustes/PLS) cannot absorb, distorting pitch. This
bug is **subsumed** by the redesign: with an isotropic chip and a single crop
scale, z becomes consistent with x/y for free.

## 3. Key enabling fact — ConvNeXt V2 is size-agnostic

The square 224² input is a **convention, not a requirement**. Confirmed in
`feat/multitask/model_v2.py`:

- Backbone `convnextv2_tiny.fcmae_ft_in22k_in1k` — fully convolutional, no
  positional embeddings.
- **Every** head consumes a global-average-pooled feature (`mean(dim=(-2,-1))`):
  backbone fusion (`v = U.mean(dim=(-2,-1))`), `PoseHead`, `LandmarkHead`,
  `EmotionVAHead` ("fuses with the global backbone GAP"); CrossAttention blocks
  operate over `H*W` tokens (token-count-agnostic). No flatten-to-fixed-length
  anywhere.

⇒ The model accepts any H×W whose dims are divisible by **32** (ConvNeXt total
stride: stem ÷4 + three ÷2 stages → 224/32 = 7×7). The fixed-square chip is
imposed purely by the extraction code, not by the network. We are free to feed
an aspect-preserving crop.

## 4. Requirements (irreducible)

1. **Isotropic** — preserve the face aspect ratio (single resize scale). The
   one non-negotiable; this is what fixes pitch and mesh-z.
2. **Consistent scale** — every face normalized to the same canonical pixel
   size (GAP pools globally, so face scale must be stable across the batch).
3. **Pad, don't clamp** off-frame regions (keeps the face centered + correct
   scale at frame edges).
4. **Dims ÷ 32** for ConvNeXt.
5. **Minimize padding, or mask it** — padded pixels dilute the global-average
   pool; either keep padding small (canvas ≈ face aspect) or use masked GAP so
   padding is ignored.
6. **One canonical chip function** used for BOTH training re-extraction and
   inference. Train/inference chips must be pixel-identical.

## 5. Proposed approach

Replace the v2 crop with an aspect-preserving square-pad chip:

- Take the RetinaFace box, apply the expand margin, **square it** by the larger
  side (`side = max(w, h) * expand`), centered on the box center.
- Resize **isotropically** to the canonical chip size (square is the natural
  fixed shape: faces are ≈square ⇒ least average padding, and the canonical 3D
  face model assumes it).
- Off-frame region of the square → **pad** (mode TBD §8), not clamp.
- Keep a single `(scale, pad_left, pad_top)` per face so the inverse map to
  frame coords is one isotropic affine; the 478-mesh z scales by the same factor
  as x/y ⇒ consistent depth, no special-casing.

This is the standard face-chip convention (img2pose / ArcFace / MediaPipe all do
square+pad). Because ConvNeXt is size-agnostic, the canvas could alternatively
be a fixed near-face rectangle — but square is simplest and we have no reason to
deviate.

### Retrain premise

Changing the chip geometry changes the input distribution, so the existing
face-trained weights do not transfer. **Retrain from scratch**: initialize from
the public `convnextv2_tiny.fcmae_ft_in22k_in1k` backbone (ImageNet pretraining
is robust to the change), re-extract **every** label source's chips with the new
canonical function, and retrain all heads. Targets that were precomputed in the
old squished-chip space (mesh, pose distillation) must be **regenerated**, not
just re-cropped.

## 6. Coordinate-mapping rewrite

With one isotropic crop scale:
- `_mesh_to_original_frame` collapses to a single affine; z scales like x/y
  (delete the raw-z passthrough). The earlier "scale z by w/MODEL_INPUT/scale"
  patch is unnecessary under the new chip.
- Landmark inverse transform and the bbox echo (`convert_bbox_output`,
  `per_face_padding_inversion_terms`) updated to the single square scale.
- Pose post-processing: the canonical sign/label convention stays
  (+pitch=up, +yaw=subject-right, +roll=subject-right, radians); with correct
  geometry the pitch scale should match img2pose without the current hacks.

## 7. Validation plan

- **Pitch parity**: v2 pitch vs img2pose/pose_mlp on the same on-camera nod
  sequence — target ≈1:1 slope (vs today's 0.4–0.6×), across the full nod range.
- **Mesh fidelity**: Procrustes residual + scale of the v2 478-mesh vs
  MPDetector's MediaPipe mesh (MPDetector already validated <1px residual,
  Procrustes scale 1.000) — should hold and improve on angled faces.
- **No-regression on the other heads**: AU/emotion/V·A/gaze metrics on the held-
  out eval set must be ≥ current; expect gains on non-frontal / edge faces.
- **Off-frame**: partial-off-frame faces keep correct mesh scale (the clamp-bug
  fixture should no longer compress).

## 8. Open decisions (need the training pipeline to answer)

1. **Canonical chip size.** Keep 224² (÷32 ⇒ 7×7), or a different ÷32 size?
   Keep the two-stage 256→224 (extract 256, random/center-crop 224 as training
   augmentation) or extract directly at 224 with the margin in `expand`?
2. **Expand factor.** Keep 1.2, retune? Apply expand before or after squaring?
3. **Pad mode.** Zeros vs reflect vs replicate — and **masked GAP** yes/no.
4. **Training-data inventory.** Enumerate every label source (AU: DISFA/BP4D/…;
   emotion/V·A: AffectNet/…; gaze; identity) and which are re-croppable from
   originals vs need regeneration.
5. **Precomputed targets.** Which targets live in chip-relative space (mesh,
   img2pose pose distillation) and must be regenerated under the new chip vs
   which are image-relative and only need re-cropping.
6. **Blendshape head.** Add the ported MediaPipe blendshape output as a head in
   this retrain (it's nearly free given the mesh path) — see pyfeat-live task
   "Add blendshape output to Detectorv2" — or defer.
7. **img2pose distillation crop.** img2pose runs on the full aspect-correct
   frame; confirm the pose-target generation is unaffected by the chip change
   (only the *student* input changes).

## 9. Interim state (already committed, pre-redesign)

Until the retrain lands, v2 ships a best-effort pose label/sign fix at the Fex
assembly (`detector_v2.py`): `Pitch=-head[0], Roll=-head[2], Yaw=+head[1]`
(canonical axes/signs; pitch remains under-scaled — documented in the code).
The classic Detector normalizes img2pose/pose_mlp identically. PnP-DLT was
removed. These are stopgaps that the chip redesign makes obsolete.
