# Usage Tips and Known Issues

Here are few general guidelines and known issues when using `py-feat`. We are always actively trying to improve these issues and make the toolbox easier to use. Please help us out by contributing on github!

Always spot-check your detections! While we've done our best to thoroughly test, benchmark, and document all the [pre-trained models](models.md) included in Py-Feat it's always possible that real-world images and videos reveal a quirk or limitation of an otherwise high-performing detector.

## Known issues
- Currently when performing detections and using `batch_size > 1`, AU models output slightly different values. This is in part due to how Py-Feat integrates the underlying detectors and we are actively working to fix this. You can follow [this issue](https://github.com/cosanlab/py-feat/issues/128) for more details.
- Detectors can be sensitive to difference in images sizes such that very large or very small images result in very different predictions. This is largely due differences in the hyper-parameters and images sizes used to train the underlying pre-trained models. To partially help with this issue, since `0.5.0` Py-Feat supports passing keyword arguments to underlying models during initializing or detection, e.g. `detector = Detector(facepose_model_kwargs={'keep_top_k': 500})`. You can follow [this issue](https://github.com/cosanlab/py-feat/issues/135) for more details. 