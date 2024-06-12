# Change Log

# 0.7.0

## New Identity Detector

- Py-feat now includes an **identity detector** via [facenet](https://github.com/timesler/facenet-pytorch). This works by projecting each detected face in an image or video frame into a 512d embedding space and clustering these embeddings using their cosine similarity (default threshold = 0.8).
- `Fex` objects include an extra column containing the identity label for each face (accessible via `Fex.identities`) as well as additional columns for each of the 512 embeddings dimensions (accessible via `Fex.identity_embeddings`). Embeddings can be useful for downstream model-training tasks.
- **Note:** identity embeddings are affected by facial expressions to some degree, and while our default threshold of 0.8 works well for many cases, you should adjust this to tailor it to your particular data.
- To save computation time, we make it possible to recompute identity labels **after** detection has been performed using the `.compute_identities(threshold=new_threshold)` method on `Fex` data objects. By default this returns a new `Fex` object with new labels in the `'Identity'` column, but can also overwrite itself in-place.
- You can also adjust the threshold at detection time using the `face_identity_threshold` keyword argument to `Detector.detect_image()` or `Detector.detect_video()`.
- Recomputing identity labels by changing the threshold **does not** change the 512d embeddings, it just adjusts how clustering is performed to get the identity labels.

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

- `Detector` no longer support uninitialized models, e.g. `any_model = None`
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
