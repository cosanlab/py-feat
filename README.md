# Py-FEAT  
[![Build Status](https://api.travis-ci.org/cosanlab/feat.svg?branch=master)](https://travis-ci.org/cosanlab/feat/)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/feat?branch=master)

Python Facial Expression Analysis Toolbox (FEAT)

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
### 3. Preprocessing FEX data
Loading a facial expression file and preprocess. 
```python
# Code goes here
```

You can also preprocess facial expression data extracted using other software (e.g., iMotions, FACET, Affectiva, OpenFace)
```python
# Code goes here
```

### 4. Analyze FEX data
Analyze FEX.
```python
# Code goes here
```

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
- [DRML: Deep Region and Multi-Label Learning](https://github.com/AlexHex7/DRML_pytorch)
- [JAANet: Joint AU Detection and Face Alignment via Adaptive Attention](https://github.com/ZhiwenShao/PyTorch-JAANet)

Emotion detection models 
- [FerNet](https://www.kaggle.com/gauravsharma99/facial-emotion-recognition?select=fer2013)
- [ResMaskNet: Residual Masking Network](https://github.com/phamquiluan/ResidualMaskingNetwork)

## Contributing
1. Fork the repository on GitHub. 
2. Run the tests with `pytest tests/` to make confirm that all tests pass on your system. If some tests fail, try to find out why they are failing. Common issues may be not having downloaded model files or missing dependencies.
3. Create your feature AND add tests to make sure they are working. 
4. Run the tests again with `pytest tests/` to make sure everything still passes, including your new feature. If you broke something, edit your feature so that it doesn't break existing code. 
5. Create a pull request to the main repository's `master` branch.

