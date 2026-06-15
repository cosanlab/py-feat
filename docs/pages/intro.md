Py-Feat: Python Facial Expression Analysis Toolbox
============================
[![arXiv-badge](https://img.shields.io/badge/arXiv-2104.03509-red.svg)](https://arxiv.org/abs/2104.03509) 
[![Package versioning](https://img.shields.io/pypi/v/py-feat.svg)](https://pypi.org/project/py-feat/)
[![Tests](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml/badge.svg)](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/py-feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/py-feat?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
[![GitHub forks](https://img.shields.io/github/forks/cosanlab/py-feat)](https://github.com/cosanlab/py-feat/network)
[![GitHub stars](https://img.shields.io/github/stars/cosanlab/py-feat)](https://github.com/cosanlab/py-feat/stargazers)
[![DOI](https://zenodo.org/badge/118517740.svg)](https://zenodo.org/badge/latestdoi/118517740)


Py-Feat provides a comprehensive set of tools and models to detect, preprocess, analyze, and visualize facial expressions (Action Units, emotions, facial landmarks) from images and videos. 

## Why you should use Py-Feat
Facial expressions convey rich information about how a person is thinking, feeling, and what they are planning to do. Recent innovations in computer vision algorithms and deep learning algorithms have led to a flurry of models that can be used to extract facial landmarks, Action Units, and emotional facial expressions with great speed and accuracy. However, researchers seeking to use these algorithms or tools such as [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), [LibreFace](https://boese0601.github.io/libreface/), [PyAFAR](https://pyafar.org/), [iMotions](https://imotions.com/)-[Affectiva](https://www.affectiva.com/science-resource/affdex-sdk-a-cross-platform-realtime-multi-face-expression-recognition-toolkit/), or [Noldus FaceReacer](https://www.noldus.com/facereader/) may find them difficult to install, use, or too expensive to purchase. It's also difficult to use the latest model or know exactly how good the models are for proprietary tools. **We developed Py-Feat to create a free, open-source, and easy to use tool for working with facial expressions data.**

## Who is it for? 
Py-Feat was created for two primary audiences in mind: 
- **Human behavior researchers**: Extract facial expressions from face images or videos with a simple line of code and analyze your data with Feat. 
- **Computer vision researchers**: Develop & share your latest model to a wide audience of users. 

Check out a recent presentation by one of the project leads [Eshin Jolly, PhD](https://eshinjolly.com/) for a broad-overview and introduction:

<iframe src="https://ejolly-py-feat.surge.sh/" style="width: 100%; height: 500px"></iframe>


## Introducing Py-FEAT v2
We have released version 2 of Py-Feat that includes a new multi-task face expression model, many bug fixes, performance optimizations, and a standalone app called [pyfeat-live](https://github.com/cosanlab/pyfeat-live/).
We still support our original version 1 of Py-FEAT that is described in [Cheong et al., 2023](https://link.springer.com/article/10.1007/s42761-023-00191-4) and there are advantages to both versions. You can access them as `Detectorv1` and `Detectorv2`. They return the same `Fex` data structure, so downstream analysis and plotting code is largely shared, but they take different approaches. Both can run on CUDA and MPS gpus and support batching, which greatly accelerates detection.

### Two detectors: `Detectorv1` and `Detectorv2`

**`Detectorv1`** is a *modular pipeline*. You choose the model for each stage — face detection, facial landmarks, Action Units, emotions, gaze, and identity — and you can swap any of them or turn one off (pass `None`). It produces the classic **68-point** facial landmarks and runs separate models (e.g. an XGBoost AU classifier, ResMaskNet emotions) one after another. That flexibility comes at a cost: running several models in sequence makes it **slower on a single frame**.

**`Detectorv2`** runs a single **multi-task neural network** that predicts Action Units, emotions, valence/arousal, gaze, head pose, a **478-point 3D MediaPipe FaceMesh**, and **52 MediaPipe/ARKit blendshapes** in *one* forward pass (plus optional identity embeddings). Because one network replaces the per-task model chain, it is **much faster — especially on single frames**, has improved performance on action unit and gaze prediction — and adds continuous valence/arousal and gaze that v1 does not produce. The trade-off is that the model set is fixed: you don't pick or disable individual components.

| | `Detectorv1` (v1) | `Detectorv2` (v2) |
|---|---|---|
| Architecture | modular, one model per task | single multi-task network |
| Swap / disable models | yes | fixed set |
| Landmarks | 68-point (dlib-style) | 478-point 3D MediaPipe FaceMesh |
| Valence/arousal, gaze | — | built-in |
| Blendshapes | — | 52 MediaPipe/ARKit |
| Single-frame speed | slower | **fast** |
| Best for | specific models, 68-pt conventions, published Cheong et al. benchmarks | speed, video, 3D mesh, valence/arousal + gaze |

```python
from feat import Detectorv1, Detectorv2

# v1 — modular: pick or disable models, 68-point landmarks
detector_v1 = Detectorv1(au_model="xgb", emotion_model="resmasknet", identity_model=None)
fex = detector_v1.detect("face.jpg", data_type="image")

# v2 — one fast multi-task model, 478-point 3D mesh + valence/arousal + gaze
detector_v2 = Detectorv2(device="cuda")        # or device="cpu" / "mps"
fex = detector_v2.detect("face.jpg", data_type="image")
```

## Contributions 
We are excited for people to add new models and features to Py-Feat. Please see the [contribution guides](https://cosanlab.github.io/feat/content/contribute.html). 

## License 
Py-FEAT is provided under the  [MIT license](https://github.com/cosanlab/py-feat/blob/master/LICENSE). You also need to cite and respect the licenses of each model you are using. Note that several models have a non-commercial stipulation. Please see the LICENSE file for links to each model's license information. 