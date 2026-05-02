# Detection speedup design

**Status:** Draft
**Author:** Luke Chang (with Claude Code)
**Date:** 2026-05-01
**Related PRs:** #258 (MPS device handoff), #259 (HOGLayer skimage parity)

## Goal

Close the speed gap between py-feat and modern facial-behavior libraries (notably OpenFace 3.0), keeping the existing public API and per-component swappability of detector backends.

The OpenFace 3.0 paper (arXiv:2506.02891) reports ~2.2× over py-feat on CPU (38 ms vs 82 ms per frame on AMD Threadripper 1920X). User-reported gaps in the 30–60× range come from running py-feat with `device='auto'` on Apple Silicon (which silently falls back to CPU because of the MPS guard) while comparing against OpenFace 3.0 on CUDA, plus the fact that py-feat processes frames one at a time despite a `batch_size` argument.

**Concrete success criteria:**

1. On M-series Macs with `device='mps'`, end-to-end `Detector(...).detect(video)` is at least 5× faster than on the same machine's CPU baseline as of `main` after PR #258 lands. (Stretch: 10×.)
2. CUDA users see no regression and a >2× speedup from batched face detection on multi-frame inputs.
3. Existing detector-component swappability (`face_model`, `landmark_model`, `au_model`, `emotion_model`, `identity_model`) is preserved.
4. Public schema of the returned `Fex` DataFrame is unchanged; numerical AU/emotion/identity outputs match pre-change values within a documented tolerance.

**Per-item speedup budget (target = 5–15× combined on MPS):** estimates, not commitments. Each item's contribution should be measured on a fixed video before the next item lands.

| Phase 1 item | Target speedup | Multiplicative? |
|---|---|---|
| 1.1 MPS device handoff (#258) | enables MPS at all (~3-5× over silent-CPU baseline) | base — others stack on this |
| 1.3 Batch img2pose | 3-10× on top of 1.1 | yes |
| 1.4 HOG wire-in | 1.05-1.15× | yes |
| 1.5 Single 224² crop | 1.05-1.15× | yes |
| 1.6 Stream video output | memory-only, no per-frame speedup | n/a |
| 1.7 Kalman tracker (optional) | 1.2-2× on video | yes |
| 1.8 (alt path) YOLO swap | 3-10× over batched-img2pose baseline | replaces 1.3 |

**Non-goals (this spec):**

- Training a single-pass multi-task trunk like OpenFace 3.0. Tracked separately if Phase 1+2 don't close the gap.
- Fine-tuning a vision–language model to emit AUs/emotions in one forward pass.

**Adjacent goal:** Adding YOLO as an alternative `face_model` (Phase 1.8). img2pose stays as the default; YOLO becomes a faster opt-in option. Pose values come from `solvePnP` on the 68 landmarks when YOLO is selected.

## Diagnosis

The current pipeline (`feat/detector.py` `Detector.detect`) runs six stages sequentially, with three independent CNN forward passes plus a CPU/numpy HOG step:

```
img2pose (ResNet18-FPN) ──► bbox + 6DoF
        │
        ├── crop 112² ──► MobileFaceNet  ──► 68 landmarks
        ├── crop 224² ──► ResMaskNet     ──► emotion        ← second crop
        ├── crop 112² ──► FaceNet        ──► 512-d identity
        └── HOG via PIL/numpy on CPU + landmarks ──► XGBoost ──► AUs
```

**Confirmed perf-relevant pain points** (file:line, current `main`):

- `feat/detector.py:320-321` — `for i in range(frames.size(0))` inside `detect_faces` runs img2pose **one frame at a time**. The DataLoader `batch_size` is honored only after this loop, by which point the slowest stage has already serialized.
- `feat/detector.py:317` — `frames.to(self.device)` was a no-op (return discarded). Fixed in #258.
- `feat/utils/__init__.py:353-379` — `set_torch_device('auto')` deliberately avoids MPS on Mac because of mixed-device tensor ops downstream. Fixed in #258 by making the affected ops device-aware.
- `feat/detector.py:357-367` — when `emotion_model='resmasknet'`, faces are cropped a *second* time at 224². Doubles the cropping cost.
- `feat/utils/image_operations.py:1226-1263` — `extract_hog_features` converts each face crop tensor → PIL → numpy → CPU HOG → numpy → tensor. The torch-native `HOGLayer` (line 905) was previously broken; #259 fixes its outputs to match skimage to ~5e-8 absolute tolerance.
- `feat/detector.py:678` — `batch_output.append(batch_results)` accumulates a pandas DataFrame for every batch (each face contributes a 512-d identity vector + AU/emotion/landmark columns). On long videos this balloons RAM (#244 reports 82 GB on 4500 frames). The `save="path.csv"` argument already streams to disk; making it the default for video would resolve this without a new code path.
- `feat/detector.py:591-596` — a `try: next(enumerate(tqdm(data_loader)))` size-sanity check creates a redundant tqdm bar and fetches an extra batch up front. Cosmetic but worth removing.

## Plan

### Phase 0 — housekeeping (independent of speedup)

These are decoupled from the perf work but block users today and should ship as their own PRs.

| # | Item | Closes | Effort |
|---|------|--------|--------|
| 0.1 | Cut a 0.7 PyPI release | #238, #250, #251, #252 | small |
| 0.2 | Modernize install: unpin `numpy`, replace `scipy.integrate.simps` with `simpson`, drop `distutils`; bump CI to test 3.11 / 3.12 / 3.13 | #242, #243, #253 | small |
| 0.3 | Convert `py-feat/mp_facemesh_v2` weights to safetensors and remove `torch.load(weights_only=False)` from `MPDetector.py` | #249 | small |
| 0.4 | Optional: silence the `tqdm` progress bar via a flag | #234 | trivial |

### Phase 1 — speedup quick wins (no retraining)

Order of operations is dictated by which dependency unblocks the next.

**1.1 Land #258 (MPS device handoff).** Status: in review. After merge, `device='mps'` runs end-to-end on Apple Silicon. Without 1.1, every other Phase 1 item is harder to verify on Mac.

**1.2 Land #259 (HOGLayer skimage parity).** Status: in review. After merge, `HOGLayer` produces feature vectors equivalent to `skimage.feature.hog` to float32 epsilon. Doesn't change runtime behavior on its own; unblocks 1.4.

**1.3 Batch img2pose** — *the single largest speedup item*.
- *Where:* `feat/detector.py:302-371` (`detect_faces`)
- *Change:* Replace the `for i in range(frames.size(0))` loop with a single batched forward pass. img2pose's underlying `GeneralizedRCNN` accepts a list of images already; the per-frame postprocess (`postprocess_img2pose`) needs to be vectorized or applied to the list of per-image outputs. Maintain the per-frame `frame_results` dict structure for downstream compatibility.
- *Validation:* New test that calls `detect_faces` on a 4-frame batch and an equivalent 4 single-frame batches; assert outputs are equal element-wise. Existing tests in `feat/tests/test_detector.py` should continue to pass.
- *Expected impact:* On CUDA, ~3-10× depending on batch size and image size. On MPS, similar after 1.1.
- *Risk:* Medium. img2pose's postprocess is non-trivial; the per-image fallback path (NaNs when no face is detected) needs to be preserved per face index, not per batch.

**1.4 Wire `extract_hog_features` to `HOGLayer`.**
- *Where:* `feat/utils/image_operations.py:1226-1263`
- *Change:* Replace the per-face `hog(transforms.ToPILImage()(convex_hull[0]), ...)` call with a single batched `HOGLayer.forward()` call across all faces. Eliminates the tensor → PIL → numpy → CPU HOG → numpy → tensor round-trip per face.
- *Validation:* New end-to-end test that runs the AU detector on a fixed input image with `au_model='xgb'` before vs after the swap; assert AU outputs match within 1e-4 relative tolerance. Because #259 establishes a ~5e-8 numerical match between `HOGLayer` and skimage, the existing trained XGBoost classifier should require no retraining.
- *Expected impact:* On CPU alone, 5-15% per-frame win. With MPS enabled (1.1), bigger because the per-face CPU stall is gone.
- *Risk:* Low–medium. HOG numerical match is verified; failure mode would be a behavioral edge case (e.g., the convex-hull mask zeroing semantics differ from PIL's color handling).

**1.5 Single 224² face crop, downsampled on GPU.**
- *Where:* `feat/detector.py:357-367` (the second `extract_face_from_bbox_torch` call when `emotion_model='resmasknet'`)
- *Change:* Crop each face once at 224² and downsample on-GPU to 112² where needed (landmarks, identity). Re-use the 224² crop directly for ResMaskNet.
- *Validation:* AU/landmark/emotion/identity outputs must match within the existing detector tests' tolerances. Add a regression test asserting only one `extract_face_from_bbox_torch` call happens per face when ResMaskNet is selected (use a counter or mock).
- *Expected impact:* ~5-15%. Larger when face count per frame is high.
- *Risk:* Low. The 112² crops can be a `F.interpolate(crop_224, size=112, mode='bilinear', align_corners=False)` step.

**1.6 Stream video output by default (memory fix).**
- *Where:* `feat/detector.py:586-700` (`detect()`)
- *Change:* When `data_type == 'video'` and `save` is not specified, auto-create a temp CSV path and stream to it. Or, simpler: hold a rolling buffer of N batches' worth of `Fex` rows (configurable) and concatenate at the end if total RAM stays under a threshold; otherwise fall back to disk streaming. Closes #244, #246, #182.
- *Also remove* the redundant `try: next(enumerate(tqdm(data_loader)))` size sanity-check at `:591-596` — replace with a one-shot `try: dataset[0]` if needed.
- *Validation:* New test that runs `detect()` on a long synthetic video and asserts peak RAM stays under a threshold. Existing tests must still pass with `save=None`.
- *Expected impact:* Memory-only fix; not a per-frame speedup but closes the multiple long-video issues.
- *Risk:* Medium. The current code has identity-resolution and `approx_time` post-processing that operate on the full DataFrame; streaming mode needs to handle these via a final pass over the saved CSV (already partially implemented).

**1.7 Optional: Kalman tracker hop for video.**
- *Where:* `feat/detector.py` (`detect()` loop)
- *Change:* Accept the offered PR in #254 (or re-implement) so that face detection runs every Nth frame and a lightweight tracker handles the others. Already partially supported via `skip_frames`.
- *Expected impact:* 1.2-2× on video, varies with how stable the face is.
- *Risk:* Low if behind a flag. Defer if 1.3-1.6 close the gap.

### Phase 1.5 (parallel track) — YOLO alternative

**1.8 is bigger than the rest of Phase 1** and is *not* a "quick win" peer. It introduces a new module, an optional dep, a license audit, a benchmark on WIDER FACE, and a `solvePnP` helper. Treat it as a parallel track — implementable in parallel with 1.3-1.7, with its own success criteria — rather than item 1.8 in the sequential order.

img2pose is a ResNet18-FPN Faster R-CNN variant that runs at ~50-150 ms per frame on CPU [estimate; not verified against the upstream paper]. Modern face-YOLO variants (yolov8-face, yolov11-face from `ultralytics`) reportedly hit ~5-15 ms per frame on CPU, are batched natively, support MPS out of the box, and have well-maintained ONNX/TensorRT export paths [based on community-published benchmarks for yolov8n-face on similar consumer CPUs]. The case for offering YOLO as an alternative `face_model` is strong even if 1.3 (batching img2pose) succeeds.

- *Where:* New `feat/face_detectors/yolo/` module mirroring the structure of `feat/face_detectors/Retinaface/`, registered in `Detector.__init__` as a valid `face_model='yolo'` choice.
- *Change:*
  - Pull in `ultralytics` as an optional dependency (or the smaller `onnxruntime`-based path if we want to avoid the dep). Bundle a yolov8n-face or yolov11n-face checkpoint via the existing HuggingFace integration (PR #228 pattern).
  - Implement a per-batch forward that returns `(bboxes, scores, optional 5-point landmarks)`.
  - Because YOLO does not predict 6DoF head pose, populate the `Pitch/Roll/Yaw/X/Y/Z` columns by running `cv2.solvePnP` on the 68 landmarks emitted by the landmark stage against a fixed 3D face reference model. This is the canonical OpenFace-2.0 approach. Lives in `feat/utils/pose.py` as a small helper.
  - Document the schema: when `face_model='yolo'`, pose values come from solvePnP and will differ slightly from img2pose's regressed values.
- *Validation:*
  - Bbox parity: on a fixed face-detection benchmark (e.g., a subset of WIDER FACE val), median IoU between YOLO and img2pose detections is ≥0.85; precision/recall within 5%.
  - Pose parity: on the same fixed image set, RMSE between solvePnP-pose and img2pose-pose is reported in the docs (won't be zero; documenting is the goal). Decide a deprecation plan only if the gap is small enough that defaulting to YOLO becomes attractive later.
  - Speed: end-to-end `detect()` with `face_model='yolo'` on a 100-frame fixed video is ≥3× faster than `face_model='img2pose'` on the same machine.
- *Expected impact:* On its own, the largest single speedup item in Phase 1 — likely 3-10× over the batched-img2pose baseline depending on hardware. Stacks multiplicatively with 1.4-1.6.
- *Risk:* Medium-high. Tradeoffs:
  - Public-schema implication: users comparing pose values across versions of py-feat will see drift if they switch face_model. Mitigate by keeping `face_model='img2pose'` as the default and documenting the difference.
  - Accuracy on small/profile/occluded faces. Face-YOLO checkpoints vary in quality; some published variants are over-tuned for frontal faces. Need to validate on the harder slices.
  - License audit on chosen YOLO weights (Ultralytics is AGPL; community-released face checkpoints vary).
- *Decision points before starting:*
  - License: AGPL is incompatible with py-feat's MIT. Either (a) use a permissively-licensed face-YOLO checkpoint (deepghs, others), (b) use a non-Ultralytics implementation (e.g., directly from the ONNX export of a permissively-licensed checkpoint), or (c) accept the AGPL constraint and document it.
  - Whether to bundle 5-point landmarks from face-YOLO (some variants emit them) and use those as anchors before the 68-point landmark step, or ignore them.

**Relationship to 1.3:** 1.3 (batch img2pose) and 1.8 (add YOLO) are not mutually exclusive. 1.3 helps users who depend on img2pose's regressed pose; 1.8 gives speed-conscious users a much faster option. If 1.8 ships first and proves out, 1.3 becomes lower-priority.

**Phase 1 stretch goal:** 5–15× over current `main` on M-series with `device='mps'` and `face_model='img2pose'`. With `face_model='yolo'` (1.8), 15-50× becomes plausible.

### Phase 2 — distill heads (selective retraining)

Once Phase 1 is shipped and the pipeline is fully on-device, the remaining hot path is the AU and emotion models. Both are candidates for distillation from the existing detectors' outputs into smaller NN heads.

**2.1 Distill the AU head.**
- *Why:* HOG + XGBoost is the only classical-ML stage left. A small CNN/MLP head trained on the same labels (DISFA, BP4D, plus self-distilled targets from the current XGBoost on a large unlabeled face corpus) can run on the same device as the rest of the pipeline.
- *Inputs:* The 112² aligned face crop and/or landmark coordinates. Possibly a pooled feature from the landmark backbone (would need to expose intermediate features).
- *Architecture:* MobileNetV3-Small or a 2-3M parameter custom CNN. Output: 20 AU intensities (or 17 to match OpenFace's set).
- *Training:* Hybrid loss combining (a) ground-truth labels where available, (b) soft labels from the current XGBoost on unlabeled face crops. Use the existing `feat.train` infrastructure if it exists; otherwise build a standalone training script in `scripts/`.
- *Validation:* AU prediction accuracy on a held-out test split must be within 5% of the current XGBoost. Latency must be ≤2× MobileFaceNet's (which is the existing fast-stage benchmark on the same hardware).
- *Risk:* Medium. Distillation works well when the teacher is consistent; XGBoost on HOG features generally is. Real risk is that the new head needs feature inputs that aren't easily exposed from the existing detector backbones.
- *Expected impact:* Eliminates the per-face HOG step entirely; another 1.5–3× on top of Phase 1 for AU-only workflows.

**2.2 Optional: distill the emotion head.**
- *Why:* ResMaskNet is the heaviest single CNN in the pipeline by parameter count.
- *Approach:* Same as 2.1 — a smaller emotion head trained against ResMaskNet's soft labels on a large face corpus.
- *Validation:* Top-1 emotion accuracy within 3% of ResMaskNet on AffectNet (or your preferred test set).
- *Risk:* Medium. Decide post-2.1 based on whether AU distillation closes the gap enough.

### Out of scope (this spec)

- **Single multi-task trunk (OpenFace-3.0-style rebuild).** Tracked as future work; revisit only if Phase 1 + 2 don't reach the success criteria.
- **VLM-based detection.** A small open-weight VLM (Qwen2-VL, MoonDream, PaliGemma) doing structured-output regression for AUs is *slower* per face than a 30M-param CNN, not faster. Useful as a labeling oracle for Phase 2 distillation if needed; not as the inference engine.
- **Removing img2pose** as the default `face_model`. 1.8 adds YOLO as an alternative; img2pose stays the default until 1.8's accuracy and pose-parity numbers justify a flip (see Open Questions).

## Validation strategy

Add a `feat/tests/perf_smoke.py` (or similar) that:

1. Times `Detector(face_model='img2pose', emotion_model='resmasknet').detect(...)` on a small fixed image and a 100-frame fixed video, on whichever device is available (`cpu`, `mps`, `cuda`).
2. Records peak RAM during video processing.
3. Asserts the timing falls within a documented envelope (loosely, since CI hardware varies).

This isn't a benchmark suite — it's a regression guard. The full benchmarking infrastructure tracked in #67/#184/#226 is the right home for proper measurements; the smoke test just makes sure no PR silently slows the pipeline by 2×.

For numerical correctness:

- Each Phase 1 item ships with a parity test (output before vs after, on a fixed input).
- Phase 2 items ship with held-out-set accuracy comparisons against the model they replace.
- Use of fixed seeds and `torch.use_deterministic_algorithms(True)` where feasible.

## Open questions

1. **CI matrix bump.** Phase 0.2 wants Python 3.12/3.13 in CI. Are GitHub Actions runners stable enough on those versions for our deps? (PyTorch 3.13 wheels exist as of 2.5.0.)
2. **Default value for `save=` on video.** Streaming-by-default is the conservative choice for memory but changes user expectations. Open to (a) auto-detect long video and stream, (b) require opt-in via flag, (c) flip the default.
3. **AU set for the distilled head.** Match the current py-feat 20-AU schema, or adopt OpenFace's 17-AU set for cross-tool comparability?
4. **Where does the spec doc live long-term?** Currently `docs/superpowers/specs/`. Could move to `docs/design/` or similar if a more conventional path is preferred.
5. **YOLO face checkpoint and license (Phase 1.8).** Which checkpoint to bundle and under what license? Ultralytics' YOLOv8/v11 are AGPL. Community face-tuned checkpoints (deepghs, etc.) are usually MIT/Apache but quality varies. Need to pick one and validate accuracy on small/profile/occluded faces.
6. **Default `face_model` after 1.8 ships.** Keep img2pose as default for backwards compatibility, or flip to YOLO and document the schema-drift in pose values? Probably (a) initially, revisit after a release.

## References

- [OpenFace 3.0 paper](https://arxiv.org/abs/2506.02891)
- [OpenFace 3.0 repo](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0)
- PR #258 (MPS device handoff) — landing prerequisite for Phase 1
- PR #259 (HOGLayer skimage parity) — landing prerequisite for 1.4
- Issues #244, #246, #182, #168, #211 — video memory and per-frame growth
- Issues #196, #187, #159, #167 — MPS bug cluster (largely resolved by #258)
- Issue #254 — Kalman tracker PR offer (Phase 1.7 candidate)
- Issue #249 — security flag (Phase 0.3)
