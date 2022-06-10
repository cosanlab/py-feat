# Change Log

## 0.4.0

### Major version breaking release!
- This release includes numerous bug fixes, api updates, and code base changes make it largely incompatible with previous releases
- To fork development from an older version of `py-feat` you can use [this archival repo](https://github.com/cosanlab/py-feat-archive) instead

### New
- Added `animate_face` and `plot_face` functions in `feat.plotting` module
- `Fex` data-classes returned from `Detector.detect_image()` or `Detector.detect_video()` now store the names of the different detectors used as attributes: `.face_model`, `.au_model`, etc
- The AU visualization model used by `plot_face` and `Detector.plot_detections(faces='aus')` has been updated to include AU11 and remove AU18 making it consistent with Py-feat's custom AU detectors (`svm` and `logistic`)
- A new AU visualization model supporting the `jaanet` AU detector, which only has 12 AUs, has now been added and will automatically be used if `Detector(au_model='jaanet')`. 
    - This visualization model can also be used by the `plot_face` function by by passing it to the `model` argument: `plot_face(model='jaanet_aus_to_landmarks')`

#### Breaking Changes
- `Detector` no longer support unintialized models, e.g. `any_model = None`
    - This is is also true for `Detector.change_model`
- Columns of interest on `Fex` data classes were previously accessed like class *methods*, i.e. `fex.aus()`. These have now been changed to class *attributes*, i.e. `fex.aus`
- Remove support for `DRML` AU detector
- Remove support for `RF` AU and emotion detectors
- New default detectors:
    - `svm` for AUs
    - `resmasknet` for emotions
    - `img2pose` for head-pose

### Development changes
- Revamped pre-trained detector handling in new `feat.pretrained` module
- More tests including testing all detector combinations

### Fixes
- [#80](https://github.com/cosanlab/py-feat/issues/80)
- [#94](https://github.com/cosanlab/py-feat/issues/94)
- [#98](https://github.com/cosanlab/py-feat/issues/98)
- [#101](https://github.com/cosanlab/py-feat/issues/101)
- [#110](https://github.com/cosanlab/py-feat/issues/110)
- [#113](https://github.com/cosanlab/py-feat/issues/113)
- [#114](https://github.com/cosanlab/py-feat/issues/114)
- [#118](https://github.com/cosanlab/py-feat/issues/118)
- [#119](https://github.com/cosanlab/py-feat/issues/119)
- [#125](https://github.com/cosanlab/py-feat/issues/125)


## 0.3.7
- Fix import error due to missing init

## 0.3.6
- Trigger Zenodo release

## 0.2.0
- Testing pypi upload