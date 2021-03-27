# Py-FEAT: Python Facial Expression Analysis Toolbox (FEAT)
[![Package versioning](https://img.shields.io/pypi/v/py-feat.svg)](https://pypi.org/project/py-feat/)
[![Build Status](https://api.travis-ci.org/cosanlab/py-feat.svg?branch=master)](https://travis-ci.org/cosanlab/py-feat/)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/py-feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/py-feat?branch=master)


Py-FEAT is a suite for facial expressions (FEX) research written in Python. This package includes tools to detect faces, extract emotional facial expressions (e.g., happiness, sadness, anger), facial muscle movements (e.g., action units), and facial landmarks, from videos and images of faces, as well as methods to preprocess, analyze, and visualize FEX data. 

For detailed examples, tutorials, and API please refer to the [Py-FEAT website](https://cosanlab.github.io/feat/). 

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
See examples in our [tutorial](https://cosanlab.github.io/py-feat/content/analysis.html#). 

## Supported Models 
Please respect the usage licenses for each model.

Face detection models
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [RetinaFace](https://github.com/deepinsight/insightface/)

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

## Contributing
1. Fork the repository on GitHub. 
2. Run the tests with `pytest tests/` to make confirm that all tests pass on your system. If some tests fail, try to find out why they are failing. Common issues may be not having downloaded model files or missing dependencies.
3. Create your feature AND add tests to make sure they are working. 
4. Run the tests again with `pytest tests/` to make sure everything still passes, including your new feature. If you broke something, edit your feature so that it doesn't break existing code. 
5. Create a pull request to the main repository's `master` branch.

