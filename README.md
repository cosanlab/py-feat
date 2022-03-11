# Py-FEAT: Python Facial Expression Analysis Toolbox
[![Package versioning](https://img.shields.io/pypi/v/py-feat.svg)](https://pypi.org/project/py-feat/)
[![Tests](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml/badge.svg)](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/py-feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/py-feat?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
[![DOI](https://zenodo.org/badge/118517740.svg)](https://zenodo.org/badge/latestdoi/118517740)

Py-FEAT is a suite for facial expressions (FEX) research written in Python. This package includes tools to detect faces, extract emotional facial expressions (e.g., happiness, sadness, anger), facial muscle movements (e.g., action units), and facial landmarks, from videos and images of faces, as well as methods to preprocess, analyze, and visualize FEX data. 

For detailed examples, tutorials, and API please refer to the [Py-FEAT website](https://cosanlab.github.io/py-feat/). 

## Installation
Option 1: Easy installation for quick use
Clone the repository    
`pip install py-feat`  

Option 2: Installation in development mode
```
git clone https://github.com/cosanlab/feat.git
cd feat && python setup.py install -e . 
```

## Usage examples
### 1. Detect FEX data from images or videos
FEAT is intended for use in Jupyter Notebook or Jupyter Lab environment. In a notebook cell, you can run the following to detect faces, facial landmarks, action units, and emotional expressions from images or videos. On the first execution, it will automatically download the default model files. You can also change the detection models from the [list of supported models](https://cosanlab.github.io/feat/content/intro.html#available-models).

```python
from feat.detector import Detector
detector = Detector() 
# Detect FEX from video
out = detector.detect_video("input.mp4")
# Detect FEX from image
out = detector.detect_image("input.png")
```

### 2. Visualize FEX data
Visualize FEX detection results.
```python
from feat.detector import Detector
detector = Detector() 
out = detector.detect_image("input.png")
out.plot_detections()
```
### 3. Preprocessing & analyzing FEX data
We provide a number of preprocessing and analysis functionalities including baselining, feature extraction such as timeseries descriptors and wavelet decompositions, predictions, regressions, and intersubject correlations. See examples in our [tutorial](https://cosanlab.github.io/py-feat/content/analysis.html#). 

## Supported Models 
Please respect the usage licenses for each model.

Face detection models
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [RetinaFace](https://github.com/deepinsight/insightface/)
- [img2pose](https://github.com/vitoralbiero/img2pose)

Facial landmark detection models
- [MobileNet](https://github.com/cunjian/pytorch_face_landmark)
- [MobileFaceNet](https://github.com/foamliu/MobileFaceNet)
- [PFLD: Practical Facial Landmark Detector](https://github.com/polarisZhao/PFLD-pytorch)

Action Unit detection models
- FEAT-Random Forest
- FEAT-SVM
- FEAT-Logistic
- [DRML: Deep Region and Multi-Label Learning](https://github.com/AlexHex7/DRML_pytorch)
- [JAANet: Joint AU Detection and Face Alignment via Adaptive Attention](https://github.com/ZhiwenShao/PyTorch-JAANet)

Emotion detection models 
- FEAT-Random Forest
- FEAT-Logistic
- [FerNet](https://www.kaggle.com/gauravsharma99/facial-emotion-recognition?select=fer2013)
- [ResMaskNet: Residual Masking Network](https://github.com/phamquiluan/ResidualMaskingNetwork)

Head pose estimation models
- [img2pose](https://github.com/vitoralbiero/img2pose)
- Perspective-n-Point algorithm to solve 3D head pose from 2D facial landmarks (via `cv2`)

## Contributing
1. Fork the repository on GitHub. 
2. Run the tests with `pytest tests/` to make confirm that all tests pass on your system. If some tests fail, try to find out why they are failing. Common issues may be not having downloaded model files or missing dependencies.
3. Create your feature AND add tests to make sure they are working. 
4. Run the tests again with `pytest tests/` to make sure everything still passes, including your new feature. If you broke something, edit your feature so that it doesn't break existing code. 
5. Create a pull request to the main repository's `master` branch.

### Adding new notebook examples
*Note*: You should execute all notebook example cells *locally* before committing changes to github as our CI workflow **does not** execute notebooks; it just renders them into pages. That's why there's also no need to commit `notebooks/_build` to git as github actions will auto-generate that folder and deploy it to the `gh-pages` branch of the repo.

1. Make sure to install the development requirements: `pip install -r requirements-dev.txt`
2. Add notebooks or markdown to the `notebooks/content` directory
3. Add images to the `notebooks/content/images` directory
4. Update the TOC as needed: `notebooks/_toc.yml` file
5. Build the HTML: `jupyter-book build notebooks`
6. View the rendered HTML by open the following in your browser: `notebooks/_build/html/index.html`  


## Continuous Integration

Automated testing is handled by Github Actions according to the following rules:
1. On pushes to the main branch and every week on Sundays, a full test-suite will be run and docs will be built and deployed
2. On PRs against the main branch, a full test-suite will be run and docs will be built but *not* deployed
3. On publishing a release via github, the package will be uploaded to PyPI and docs will be built and deployed

*Note*: Each of these workflows can also be run manually

## Licenses
Py-FEAT is provided under the MIT license. You also need to respect the licenses of each model you are using. Please see the LICENSE file for links to each model's license information. 
