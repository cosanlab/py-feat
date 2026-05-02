# Change Log

# 0.7.0 (in development on `v0.7-dev`)

## Notes

The first release since 0.6.2 (March 2024). This is both a major structural overhaul of py-feat's internal `Detector` class and a coordinated batch of breaking changes that bring py-feat onto modern Python and PyTorch. Users on Python 3.10 or earlier, or who use `nltools.data.Adjacency` from `Fex.distance()`, will need to migrate (see the migration guide below).

Highlights:
- HuggingFace model hub integration: all pre-trained models are now versioned and downloaded on first use.
- End-to-end MPS support on Apple Silicon (was silently disabled before).
- Pure-PyTorch head-pose and gaze for the MediaPipe Face Mesh detector. No OpenCV dependency.
- Modern video decoding via torchcodec (CPU + CUDA; CPU on Apple Silicon, transfer to MPS for inference).
- Fixed `HOGLayer` (previously broken; now matches `skimage.feature.hog` to ~5e-8 absolute tolerance).
- Removed `nltools` as a runtime dependency.

## Breaking API changes

- `Detector.detect_image()` and `Detector.detect_video()` are removed; both flows go through a single `Detector.detect()` with `data_type` in `{'image', 'tensor', 'video'}`.
- The default face detector is `img2pose`, which also provides 6-DoF head-pose estimation.
- **Python 3.11+ required.** Drops support for 3.8 / 3.9 / 3.10. CI tests 3.11 / 3.12 / 3.13. *(PR #262)*
- **`nltools` removed as a dependency.** Functions previously sourced from nltools are now in `feat.utils.stats`: `regress`, `downsample`, `upsample`, `set_decomposition_algorithm`. `Fex.distance()` now returns a `pandas.DataFrame` instead of an `nltools.data.Adjacency`. *(PR #262)*
- **`nilearn` removed as a dependency.** `Fex.clean()` now uses an in-house `feat.utils.stats.clean_signal` (scipy/numpy-based) for detrending, confound regression, Butterworth filtering, and standardization. Same arguments as before; `*args, **kwargs` no longer forwarded.
- **`av` (PyAV) removed as a dependency.** `VideoDataset` now reads frames and metadata via torchcodec only. The `metadata['fps_frac']` field (previously a `fractions.Fraction` from `av.stream.average_rate`) is gone; use `metadata['fps']` (float). *(PR #271)*
- **AU classifier model files migrated to modern format (`*_v2.skops`).** No user-visible behavior change — predictions are bit-identical to the originals across all 20 sub-classifiers (verified at multiple feature scales and seeds via `scripts/migrate_au_skops.py`). Internally, the new files (a) reference their wrapper class via the real module path instead of `__main__`, and (b) for xgb, embed Booster buffers in xgboost's modern UBJ format. This eliminates a deprecated-API warning and an intermittent SIGSEGV in `Booster.__setstate__` on Python 3.13. The original `*.skops` files remain on HuggingFace alongside the v2 files for compatibility with py-feat ≤ 0.6.x installs; py-feat 0.7+ tries v2 first and falls back to v1 if v2 isn't yet on the hub.
- **`face_model` parameter re-introduced on `Detector`.** The v0.7 changelog originally said face detection was hardcoded to img2pose with no swap; that's no longer accurate. `Detector(face_model='img2pose')` (default) preserves prior behavior with 6DoF head pose. `Detector(face_model='retinaface_r34')` activates a new ResNet34-backbone RetinaFace face detector ported from yakhyo's MIT implementation: 88.9% WIDERFACE Hard AP (vs img2pose's 55.5% per Cheong et al, Affective Science 2023), batched-by-default with `torchvision.ops.batched_nms` on-device, ~210 detections/sec on Apple Silicon at batch 32 versus img2pose's per-image cost. With `face_model='retinaface_r34'`, the 6DoF pose columns are populated via DLT-PnP from the 68 landmarks (`feat.utils.face_pose_pnp`); pose values are approximate and can differ from img2pose's regressed values by up to ~30° on Pitch. Stay on `face_model='img2pose'` for pose accuracy that matches the published Cheong et al numbers. Trained weights distributed at `py-feat/retinaface_r34` on HuggingFace.
- **`mp_facemesh_v2` migrated from pickled FX GraphModule to TorchScript** (closes #249). The previous landmark file (`face_landmarks_detector_Nx3x256x256_onnx.pth`) was an `onnx2torch`-converted FX graph, which can only be deserialized via `torch.load(weights_only=False)` — a documented arbitrary-code-execution path — and required `onnx2torch` to be importable at load time. The new file (`face_landmarks_detector.pt`, distributed alongside the legacy file at `py-feat/mp_facemesh_v2`) is a TorchScript module loaded via `torch.jit.load`: no pickle execution, no `onnx2torch` import. Bit-identical outputs across batch sizes 1, 2, 4, 8 verified at conversion time. py-feat 0.7+ tries the new file first and falls back to the legacy file (with a warning) for older HF revisions.
- **`onnx2torch` removed as a runtime dependency.** Was only needed to deserialize the legacy `mp_facemesh_v2` FX GraphModule; with TorchScript loading the package is no longer touched at runtime. Drops one transitive dep + one piece of the security surface.

## Bug fixes

- **`MPDetector(face_model='retinaface')` works again.** The v0.7 RetinaFace rebuild deleted the MobileNet0.25 path that MPDetector's face detector relied on; the prior workaround was a `NotImplementedError` pointing users to `Detector(face_model='retinaface_r34')`. MPDetector now uses the same ResNet34 wrapper as `Detector`, so `MPDetector(face_model='retinaface')` and `MPDetector(face_model='retinaface_r34')` both build and detect end-to-end. Also fixes a pre-existing dtype mismatch in `convert_landmarks_3d` (Python `float` -> float64 vs canonical face model's float32) that would have crashed `estimate_face_pose_from_mesh` once the dataframe reached pose estimation.
- *(More breaking changes to be added as PRs land on `v0.7-dev`.)*

## New features

- **Identity detector** via [facenet](https://github.com/timesler/facenet-pytorch). Each detected face is projected into a 512-d embedding space and clustered by cosine similarity; the resulting label is stored in `Fex.identities` and the embeddings in `Fex.identity_embeddings`. The clustering threshold defaults to 0.8 and can be tuned at detection time via `face_identity_threshold` or recomputed on an existing `Fex` via `.compute_identities(threshold=new_threshold)`.
- **HuggingFace model integration.** Pre-trained weights live at the [py-feat HuggingFace org](https://huggingface.co/py-feat) and are downloaded on first use.
- **Pure-PyTorch head-pose and gaze for MediaPipe Face Mesh.** Closed-form Umeyama similarity alignment replaces the iterative Adam-loop pose estimator; head-pose-compensated gaze in head-centric frame. No OpenCV dependency. *(PR #261)*
- **`feat.utils.io.decode_video`** for sliced or streamed video decoding via torchcodec; replaces the prior PyAV path and fixes the regression where loading a single frame for plotting decoded the entire video into memory. *(PR #263)*
- **`HOGLayer`** now produces feature vectors that match `skimage.feature.hog` to ~5e-8 absolute tolerance for L1, L1-sqrt, L2, and L2-Hys block normalizations. Wiring it into `extract_hog_features` is a follow-up. *(PR #259)*
- *(More features to be added as PRs land on `v0.7-dev`.)*

## Bug fixes

- **MPS device handoff.** Three places in the inference path created tensors on CPU while model weights were on MPS or CUDA, causing `Detector(device='mps')` to crash partway through. End-to-end MPS detection now runs cleanly on Apple Silicon. *(PR #258)*
- *(More bug fixes to be added as PRs land on `v0.7-dev`.)*

## Documentation updates

- Tutorials updated for the new `Detector.detect()` API.
- New [FAQ](https://py-feat.org/pages/faq.html).

## Migration guide (in progress)

If you used:
- `Detector.detect_image()` or `Detector.detect_video()` -> change to `Detector.detect(..., data_type='image')` or `data_type='video'`.
- `from nltools.stats import regress, downsample, upsample` -> change to `from feat.utils.stats import regress, downsample, upsample`.
- `from nltools.data import Adjacency` for the result of `Fex.distance()` -> the result is now a plain `pandas.DataFrame`. Drop the import; use `.shape` instead of `.square_shape()`.
- Python 3.10 or earlier -> upgrade to 3.11+. macOS users may also need `brew install libomp` for xgboost on Python 3.13.

### MPDetector pose / gaze numerics changed

If you compared `Pitch / Roll / Yaw / X / Y / Z` outputs from `MPDetector` between 0.6.x and 0.7.0, the values will differ. The prior estimator minimized the wrong objective (`mean(z_proj²)` instead of a true reprojection error) and produced effectively meaningless head-pose values; the new closed-form Umeyama alignment produces the actual head pose. Treat the prior values as noise.

`MPDetector` also now emits `gaze_pitch` and `gaze_yaw` columns (radians, head-centric frame) in addition to the existing `gaze_angle`. The `gaze_angle` column is preserved for backward compatibility but its semantic shifted from "angle from camera-forward" to "angle from head-forward in head frame" - so a turned head no longer registers as averted gaze even when the eyes are looking along the head's forward axis.

# 0.6.1

## Notes

This version **drops support for Python 3.7** and fixes several dependency related issues:

- [#162](https://github.com/cosanlab/py-feat/issues/162)
- [#176](https://github.com/cosanlab/py-feat/issues/176)
- We can now handle images with an alpha-channel by just grabbing the RGB channels (typically in png files)
- Update minimum `scikit-learn` version requirement to ensure our viz models are loaded correctly
- Soft-pin `numexpr` version until this upstream pandas issue is [fixed](https://github.com/pandas-dev/pandas/issues/54449)

# 0.6.0

## Notes

This is a large model-update release. Several users noted issues with our AU models due to problematic HOG feature extraction. We have now retrained all of our models that were affected by this issue. This version will automatically download the new model weights and use them without any additional user input.

## `Detector` Changes

We have made the decision to make video processing much more memory efficient at the trade-off of increased processing time. Previously `py-feat` would load all frames into RAM and then process them. This was problematic for large videos and would cause kernel panics or system freezes. Now, `py-feat` will lazy-load video-frames one at a time, which scales to videos of any length or size assuming that your system has enough RAM to hold a few frames in memory (determined by `batch_size`). However, this also makes processing videos a bit slower and GPU benefits less dramatic. We have made this trade-off in favor of an easier end-user experience, but will be watching torch's [VideoReader](https://pytorch.org/vision/stable/generated/torchvision.io.VideoReader.html#torchvision.io.VideoReader) implementation closely and likely use that in future versions.

# 0.5.1

## Notes

This is a maintenance release that addresses multiple under-the-hood issues with `py-feat` failing when images or videos contain 0 faces. It addresses the following specific issues amongst others and is recommended for all users:

- [#153](https://github.com/cosanlab/py-feat/issues/153)
- [#155](https://github.com/cosanlab/py-feat/issues/155)
- [#158](https://github.com/cosanlab/py-feat/issues/158)
- [#160](https://github.com/cosanlab/py-feat/issues/160)

# 0.5.0

## Notes

This is a large overhaul and refactor of some of the core testing and API functionality to make future development, maintenance, and testing easier. Notable highlights include:

- tighter integration with `torch` data loaders
- dropping `opencv` as a dependency
- experimental support for macOS m1 GPUs
- passing keyword arguments to underlying `torch` models for more control

## `Detector` Changes

### New

- you can now pass keyword arguments directly to the underlying pytorch/sklearn models on `Detector` initialization using dictionaries. For example you can do: `detector = Detector(facepose_model_kwargs={'keep_top_k': 500})` to initialize `img2pose` to only use 500 instead of 750 features
- all `.detect_*` methods can also pass keyword arguments to the underlying pytorch/sklearn models, albeit these will be passed to their underlying `__call__` methods
- SVM AU model has been retrained with new HOG feature PCA pipeline
- new XGBoost AU model with new HOG feature PCA pipeline
- `.detect_image` and `.detect_video` now display a `tqdm` progressbar
- new `skip_failed_detections` keyword argument to still generate a `Fex` object when processing multiple images and one or more detections fail

### Breaking

- the new default model for landmark detection was changed from `mobilenet` to `mobilefacenet`.
- the new default model for AU detection was changed to our new `xgb` model which gives continuous valued predictions between 0-1
- remove support for `fer` emotion model
- remove support for `jaanet` AU model
- remove support for `logistic` AU model
- remove support for `pnp` facepose detector
- drop support for reading and manipulating Affectiva and FACET data
- `.detect_image` will no longer resize images on load as the new default for `output_size=None`. If you want to process images with `batch_size > 1` and images differ in size, then you will be **required** to manually set `output_size` otherwise py-feat will raise a helpful error message

## `Fex` Changes

### New

- new `.update_sessions()` method that returns a **copy** of a `Fex` frame with the `.sessions` attribute updated, making it easy to chain operations
- `.predict()` and `.regress()` now support passing attributes to `X` and or `Y` using string names that match the attribute names:
  - `'emotions'` use all emotion columns (i.e. `fex.emotions`)
  - `'aus'` use all AU columns (i.e. `fex.aus`)
  - `'poses'` use all pose columns (i.e. `fex.poses`)
  - `'landmarks'` use all landmark columns (i.e. `fex.landmarks`)
  - `'faceboxes'` use all facebox columns (i.e. `fex.faceboxes`)
  - You can also combine feature groups using a **comma-separated string** e.g. `fex.regress(X='emotions,poses', y='landmarks')`
- `.extract_*` methods now include `std` and `sem`. These are also included in `.extract_summary()`

### Breaking

- All `Fex` attributes have been pluralized as indicated below. For the time-being old attribute access will continue to work but will show a warning. We plan to formally drop support in a few versions
  - `.landmark` -> `.landmarks`
  - `.facepose` -> `.poses`
  - `.input` -> `.inputs`
  - `.landmark_x` -> `.landmarks_x`
  - `.landmark_y` -> `.landmarks_y`
  - `.facebox` -> `.faceboxes`

## Development changes

- `test_pretrained_models.py` is now more organized using `pytest` classes
- added tests for `img2pose` models
- added more robust testing for the interaction between `batch_size` and `output_size`

## General Fixes

- data loading with multiple images of potentially different sizes should be faster and more reliable
- fix bug in `resmasknet` that would give poor predictions when multiple faces were present and particularly small
- #150
- #149
- #148
- #147
- #145
- #137
- #134
- #132
- #131
- #130
- #129
- #127
- #121
- #104

# 0.4.0

## Major version breaking release!

- This release includes numerous bug fixes, api updates, and code base changes make it largely incompatible with previous releases
- To fork development from an older version of `py-feat` you can use [this archival repo](https://github.com/cosanlab/py-feat-archive) instead

## New

- Added `animate_face` and `plot_face` functions in `feat.plotting` module
- `Fex` data-classes returned from `Detector.detect_image()` or `Detector.detect_video()` now store the names of the different detectors used as attributes: `.face_model`, `.au_model`, etc
- The AU visualization model used by `plot_face` and `Detector.plot_detections(faces='aus')` has been updated to include AU11 and remove AU18 making it consistent with Py-feat's custom AU detectors (`svm` and `logistic`)
- A new AU visualization model supporting the `jaanet` AU detector, which only has 12 AUs, has now been added and will automatically be used if `Detector(au_model='jaanet')`.
  - This visualization model can also be used by the `plot_face` function by by passing it to the `model` argument: `plot_face(model='jaanet_aus_to_landmarks')`

### Breaking Changes

- `Detector` no longer support unintialized models, e.g. `any_model = None`
  - This is is also true for `Detector.change_model`
- Columns of interest on `Fex` data classes were previously accessed like class _methods_, i.e. `fex.aus()`. These have now been changed to class _attributes_, i.e. `fex.aus`
- Remove support for `DRML` AU detector
- Remove support for `RF` AU and emotion detectors
- New default detectors:
  - `svm` for AUs
  - `resmasknet` for emotions
  - `img2pose` for head-pose

## Development changes

- Revamped pre-trained detector handling in new `feat.pretrained` module
- More tests including testing all detector combinations

## Fixes

- [#80](https://github.com/cosanlab/py-feat/issues/80)
- [#81](https://github.com/cosanlab/py-feat/issues/81)
- [#94](https://github.com/cosanlab/py-feat/issues/94)
- [#98](https://github.com/cosanlab/py-feat/issues/98)
- [#101](https://github.com/cosanlab/py-feat/issues/101)
- [#106](https://github.com/cosanlab/py-feat/issues/106)
- [#110](https://github.com/cosanlab/py-feat/issues/110)
- [#113](https://github.com/cosanlab/py-feat/issues/113)
- [#114](https://github.com/cosanlab/py-feat/issues/114)
- [#118](https://github.com/cosanlab/py-feat/issues/118)
- [#119](https://github.com/cosanlab/py-feat/issues/119)
- [#125](https://github.com/cosanlab/py-feat/issues/125)

# 0.3.7

- Fix import error due to missing init

# 0.3.6

- Trigger Zenodo release

# 0.2.0

- Testing pypi upload
