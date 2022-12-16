# Tips, Community, and Known Issues

Here are few general guidelines and known issues when using `py-feat`. We are always actively trying to improve these issues and make the toolbox easier to use. Please help us out by contributing on github!

**Always spot-check your detections! While we've done our best to thoroughly test, benchmark, and document all the [pre-trained models](models.md) included in Py-Feat it's always possible that real-world images and videos reveal a quirk or limitation of an otherwise high-performing detector.**

## Community
- [Hypothesis:](https://web.hypothes.is/) a social annotation tool that allows you post and read posts from other users who have visited this site. Click on the `<` on the top right of any page to get started.
- [Discourse Community:](https://www.askpbs.org/c/py-feat/26>) a Stack Overflow like forum where you can view, contribute, and vote on FAQs regarding ``py-feat`` usage. Please ask questions here first so other users can benefit from the answers!
- [Open a Github issue](https://github.com/cosanlab/py-feat/issues>) for all code related problems. You can do so by click the github icon on the top of any page.

## Known issues
- Currently when performing detections and using `batch_size > 1`, AU models output slightly different values. This is in part due to how Py-Feat integrates the underlying detectors and we are actively working to fix this. You can follow [this issue](https://github.com/cosanlab/py-feat/issues/128) for more details.
- Detectors can be sensitive to difference in images sizes such that very large or very small images result in very different predictions. This is largely due differences in the hyper-parameters and images sizes used to train the underlying pre-trained models. To partially help with this issue, since `0.5.0` Py-Feat supports passing keyword arguments to underlying models during initializing or detection, e.g. `detector = Detector(facepose_model_kwargs={'keep_top_k': 500})`. You can follow [this issue](https://github.com/cosanlab/py-feat/issues/135) for more details. 