Py-Feat: Python Facial Expression Analysis Toolbox
============================
[![Package versioning](https://img.shields.io/pypi/v/py-feat.svg)](https://pypi.org/project/py-feat/)
[![Build Status](https://api.travis-ci.org/cosanlab/py-feat.svg?branch=master)](https://travis-ci.org/cosanlab/py-feat/)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/py-feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/py-feat?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
[![GitHub forks](https://img.shields.io/github/forks/cosanlab/py-feat)](https://github.com/cosanlab/py-feat/network)
[![GitHub stars](https://img.shields.io/github/stars/cosanlab/py-feat)](https://github.com/cosanlab/py-feat/stargazers)
[![DOI](https://zenodo.org/badge/118517740.svg)](https://zenodo.org/badge/latestdoi/118517740)


Py-Feat provides a comprehensive set of tools and models to easily detect facial expressions (Action Units, emotions, facial landmarks) from images and videos, preprocess & analyze facial expression data, and visualize facial expression data. 

## Why you should use Py-Feat
Facial expressions convey rich information about how a person is thinking, feeling, and what they are planning to do. Recent innovations in computer vision algorithms and deep learning algorithms have led to a flurry of models that can be used to extract facial landmarks, Action Units, and emotional facial expressions with great speed and accuracy. However, researchers seeking to use these algorithms or tools such as [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), [iMotions](https://imotions.com/)-[Affectiva](https://www.affectiva.com/science-resource/affdex-sdk-a-cross-platform-realtime-multi-face-expression-recognition-toolkit/), or [Noldus FaceReacer](https://www.noldus.com/facereader/) may find them difficult to install, use, or too expensive to purchase. It's also difficult to use the latest model or know exactly how good the models are for proprietary tools. **We developed Py-Feat to create a free, open-source, and easy to use tool for working with facial expressions data.**

## Who is it for? 
Py-Feat was created for two primary audiences in mind: 
- **Human behavior researchers**: Extract facial expressions from face images or videos with a simple line of code and analyze your data with Feat. 
- **Computer vision researchers**: Develop & share your latest model to a wide audience of users. 

and anyone else interested in analyzing facial expressions!

## Installation
Install from [pip](https://pypi.org/project/py-feat/)
```
pip install py-feat
```

Install from [source](https://github.com/cosanlab/feat)
```
git clone https://github.com/cosanlab/feat.git
cd feat && python setup.py install
```

You can install it in [Google Colab](http://colab.research.google.com/) or [Kaggle](http://kaggle.com/) using the code above. You can also install it in Development Mode:  
```
!git clone https://github.com/cosanlab/feat.git  
!cd feat && pip install -q -r requirements.txt
!cd feat && pip install -q -e . 
!cd feat && python bin/download_models.py
# Click Runtime from top menu and Restart Runtime! 
```

The last development mode installation using the `pip install -e .` can also be useful when contributing to Py-Feat.

## Check installation

Import the Fex class
```python
from feat import Fex
```

Import the Detector class
```python
from feat import Detector
```

## Available models
Below is a list of models implemented in Py-Feat and ready to use. The model names are in the titles followed by the reference publications.
### Action Unit detection
- `rf`: Random Forest model trained on Histogram of Oriented Gradients extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `logistic`: Logistic Classifier model trained on Histogram of Oriented Gradients extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `JAANET`: Joint facial action unit detection and face alignment via adaptive attention trained with BP4D and BP4D+ ([Shao et al., 2020](https://arxiv.org/pdf/2003.08834v1.pdf))
- `DRML`: Deep region and multi-label learning for facial action unit detection by ([Zhao et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf))
- `img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation ([Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2.pdf)). Performs simultaneous (one-shot) face detection and head pose estimation.
- `img2pose-c`: A 'constrained' version of the above model, fine-tuned on images of frontal faces with pitch, roll, yaw measures in the range of (-90, 90) degrees. Shows lesser performance on difficult face detection tasks, but state-of-the-art performance on face pose estimation for frontal faces.
###  Emotion detection
- `rf`: Random Forest model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets
- `fernet`: Deep convolutional network
- `ResMaskNet`: Facial expression recognition using residual masking network by ([Pham et al., 2020](https://ailb-web.ing.unimore.it/icpr/author/3818))
###  Face detection
- `MTCNN`: Multi-task cascaded convolutional networks by ([Zhang et al., 2016](https://arxiv.org/pdf/1604.02878.pdf); [Zhang et al., 2020](https://ieeexplore.ieee.org/document/9239720))
- `FaceBoxes`: A CPU real-time fae detector with high accuracy by ([Zhang et al., 2018](https://arxiv.org/pdf/1708.05234v4.pdf))
- `RetinaFace`: Single-stage dense face localisation in the wild by ([Deng et al., 2019](https://arxiv.org/pdf/1905.00641v2.pdf))
###  Facial landmark detection
- `PFLD`: Practical Facial Landmark Detector by ([Guo et al, 2019](https://arxiv.org/pdf/1902.10859.pdf))
- `MobileFaceNet`: Efficient CNNs for accurate real time face verification on mobile devices ([Chen et al, 2018](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf))
- `MobileNet`: Efficient convolutional neural networks for mobile vision applications ([Howard et al, 2017](https://arxiv.org/pdf/1704.04861v1.pdf))
### Face/Head pose estimation
- `img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation ([Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2.pdf)). Performs simultaneous (one-shot) face detection and head pose estimation.
- `img2pose-c`: A 'constrained' version of the above model, fine-tuned on images of frontal faces with pitch, roll, yaw measures in the range of (-90, 90) degrees. Shows lesser performance on hard face detection tasks, but state-of-the-art performance on head pose estimation for frontal faces.
- `Perspective-n-Point`: A levenberg-marquardt-optimization-based method implemented via `cv2` to solve the [Perspective n Point](https://en.wikipedia.org/wiki/Perspective-n-Point) (PnP) problem to obtain 3D head pose from 2D facial landmarks.

## Contributions 
We are excited for people to add new models and features to Py-Feat. Please see the [contribution guides](https://cosanlab.github.io/feat/content/contribute.html). 

## License 
Py-FEAT is provided under the  [MIT license](https://github.com/cosanlab/py-feat/blob/master/LICENSE). You also need to cite and respect the licenses of each model you are using. Please see the LICENSE file for links to each model's license information. 