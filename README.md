# FEAT  
[![Build Status](https://api.travis-ci.org/cosanlab/feat.svg?branch=master)](https://travis-ci.org/cosanlab/feat/)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/feat?branch=master)

Facial Expression Analysis Toolbox (FEAT)

FEAT is a suite for facial expressions (FEX) research written in Python. This package includes tools to extract emotional facial expressions (e.g., happiness, sadness, anger), facial muscle movements (e.g., action units), and facial landmarks, from videos and images of faces, as well as methods to preprocess, analyze, and visualize FEX data. 

## Installation
Option 1 
Clone the repository    
`git clone https://github.com/cosanlab/feat.git`  
Run setup  
`python setup.py install`

Option 2  
`pip install git+https://github.com/cosanlab/feat`

## Usage examples
### 1. Feature extraction
Extract emotion predictions from a face video.
```python
python detect_fex.py -i input.mp4 -o output.csv
```

or detect in notebok.
```python
from feat.detector import Detector
detector = Detector() 
# Detect FEX from video
out = detector.detect_video("input.mp4")
# Detect FEX from image
out = detector.detect_image("input.png")
```

### 2. Visualize FEX data
Visualize results of detections.
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

## Methods (to use when writing up paper)

Emotion prediction model
A deep convolutional neural network was trained on the Facial Expression Recognition database by Carrier et al 2013. 
- [kaggle notebook](https://www.kaggle.com/jcheong0428/facial-emotion-recognition) adapated from [Gaurav Sharma](https://medium.com/analytics-vidhya/facial-emotion-recognition-fer-using-keras-763df7946a64)

AU prediction model 
Joint Facial Action Unit Detection and Face Alignment via Adaptive Attention[Shao, Liu, Cai, and Ma, 2020](https://arxiv.org/pdf/2003.08834.pdf)

## Contributing
1. Fork the repository on GitHub. 
2. Run the tests with `pytest tests/` to make confirm that all tests pass on your system. If some tests fail, try to find out why they are failing. Common issues may be not having downloaded model files or missing dependencies.
3. Create your feature AND add tests to make sure they are working. 
4. Run the tests again with `pytest tests/` to make sure everything still passes, including your new feature. If you broke something, edit your feature so that it doesn't break existing code. 
5. Create a pull request to the main repository's `master` branch.

## Google DOCS for development
https://docs.google.com/document/d/1cqbDp5dkMtnWWdFtAowLGf_l1zhnGmvb8JcOxNsn8dc/edit?usp=sharing


## [Documentation](https://feat.readthedocs.io/en/latest/index.html)
[Short examples](https://paper.dropbox.com/doc/feat_tutorial-JT4sSvNEFA77Hgeo5kVg2) for how to use the toolbox are currently on dropbox papers.  This will eventually be moved to readthedocs.

---------
#### Credit

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) with the following [template](https://github.com/ejolly/cookiecutter-pypackage).
http://eshinjolly.com/pybest/
