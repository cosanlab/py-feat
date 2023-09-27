# Py-FEAT: Python Facial Expression Analysis Toolbox
[![arXiv-badge](https://img.shields.io/badge/arXiv-2104.03509-red.svg)](https://arxiv.org/abs/2104.03509) 
[![Package versioning](https://img.shields.io/pypi/v/py-feat.svg)](https://pypi.org/project/py-feat/)
[![Tests](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml/badge.svg)](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/py-feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/py-feat?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
[![DOI](https://zenodo.org/badge/118517740.svg)](https://zenodo.org/badge/latestdoi/118517740)

Py-FEAT is a suite for facial expressions (FEX) research written in Python. This package includes tools to detect faces, extract emotional facial expressions (e.g., happiness, sadness, anger), facial muscle movements (e.g., action units), and facial landmarks, from videos and images of faces, as well as methods to preprocess, analyze, and visualize FEX data. 

For detailed examples, tutorials, contribution guidelines, and API please refer to the [Py-FEAT website](https://cosanlab.github.io/py-feat/). 

## Installation
Option 1: Easy installation for quick use
Clone the repository    
`pip install py-feat`  

Option 2: Installation in development mode
```
git clone https://github.com/cosanlab/feat.git
cd feat && python setup.py install -e . 
```

If you're running into issues on arm-based macOS (e.g. m1, m2) you should install pytables using one of the methods below *before* installing py-feat:

`pip install git+https://github.com/PyTables/PyTables.git`  
OR  
`conda install pytables`

Py-Feat currently supports both CPU and GPU processing on NVIDIA cards. We have **experimental** support for GPUs on macOS which you can try with `device='auto'`. However, we currently advise using the default (`cpu`) on macOS until PyTorch support stabilizes.

## Contributing

**Note:** If you forked or cloned this repo prior to 04/26/2022, you'll want to create a new fork or clone as we've used `git-filter-repo` to clean up large files in the history. If you prefer to keep working on that old version, you can find an [archival repo here](https://github.com/cosanlab/py-feat-archive)

## Testing

The test have been relocated to `feat/tests/`.
Please ensure all tests pass before creating any pull request or larger change to the code base.

## Continuous Integration

Automated testing is handled by Github Actions according to the following rules:
1. On pushes to the main branch and every week on Sundays, a full test-suite will be run and docs will be built and deployed
2. On PRs against the main branch, a full test-suite will be run and docs will be built but *not* deployed
3. On publishing a release via github, the package will be uploaded to PyPI and docs will be built and deployed

*Note*: Each of these workflows can also be run manually. They can also be skipped by adding 'skip ci' anywhere inside your commit message.

## Model Weights
Model weights are stored as assets in release tag v0.1. They will automatically download as needed.

## Licenses
Py-FEAT is provided under the MIT license. You also need to respect the licenses of each model you are using. Please see the LICENSE file for links to each model's license information. 
