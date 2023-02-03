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


Py-Feat provides a comprehensive set of tools and models to easily detect facial expressions (Action Units, emotions, facial landmarks) from images and videos, preprocess & analyze facial expression data, and visualize facial expression data. 

## Why you should use Py-Feat
Facial expressions convey rich information about how a person is thinking, feeling, and what they are planning to do. Recent innovations in computer vision algorithms and deep learning algorithms have led to a flurry of models that can be used to extract facial landmarks, Action Units, and emotional facial expressions with great speed and accuracy. However, researchers seeking to use these algorithms or tools such as [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), [iMotions](https://imotions.com/)-[Affectiva](https://www.affectiva.com/science-resource/affdex-sdk-a-cross-platform-realtime-multi-face-expression-recognition-toolkit/), or [Noldus FaceReacer](https://www.noldus.com/facereader/) may find them difficult to install, use, or too expensive to purchase. It's also difficult to use the latest model or know exactly how good the models are for proprietary tools. **We developed Py-Feat to create a free, open-source, and easy to use tool for working with facial expressions data.**

## Who is it for? 
Py-Feat was created for two primary audiences in mind: 
- **Human behavior researchers**: Extract facial expressions from face images or videos with a simple line of code and analyze your data with Feat. 
- **Computer vision researchers**: Develop & share your latest model to a wide audience of users. 

and anyone else interested in analyzing facial expressions! 

Check out a recent presentation by one of the project leads [Eshin Jolly, PhD](https://eshinjolly.com/) for a broad-overview and introduction:

<iframe src="https://ejolly-py-feat.surge.sh/" style="width: 100%; height: 500px"></iframe>


## Installation
You can easily install the latest stable version from PyPi:

```
pip install py-feat
```

For other installation methods (e.g. Google Collab, development) see the [how to install page](./installation.md)

## Available models
Py-feat includes several **pre-trained** models for Action Unit detection, Emotion detection, Face detection, Facial Landmark detection, and Face/Head post estimation. 

You can check out the full list on the [pre-trained models page](./models.md).

## Contributions 
We are excited for people to add new models and features to Py-Feat. Please see the [contribution guides](https://cosanlab.github.io/feat/content/contribute.html). 

## License 
Py-FEAT is provided under the  [MIT license](https://github.com/cosanlab/py-feat/blob/master/LICENSE). You also need to cite and respect the licenses of each model you are using. Please see the LICENSE file for links to each model's license information. 