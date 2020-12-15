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

Verify models have been downloaded
`python download_models.py` 

## Usage examples
### 1. Feature extraction
Extract emotion predictions from a face video.
```python
python extract_fex.py -i input.mp4 -o output.csv
```

### 2. Preprocessing FEX data
Loading a facial expression file and preprocess. 
```python
# Code goes here
```

You can also preprocess facial expression data extracted using other software (e.g., iMotions, FACET, Affectiva, OpenFace)
```python
# Code goes here
```

### 3. Analyze FEX data
Analyze FEX.
```python
# Code goes here
```

### 4. Visualize FEX data
Visualize FEX.
```python
# Code goes here
```

## Google DOCS for development
https://docs.google.com/document/d/1cqbDp5dkMtnWWdFtAowLGf_l1zhnGmvb8JcOxNsn8dc/edit?usp=sharing


## [Documentation](https://feat.readthedocs.io/en/latest/index.html)
[Short examples](https://paper.dropbox.com/doc/feat_tutorial-JT4sSvNEFA77Hgeo5kVg2) for how to use the toolbox are currently on dropbox papers.  This will eventually be moved to readthedocs.

---------
#### Credit

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) with the following [template](https://github.com/ejolly/cookiecutter-pypackage).
http://eshinjolly.com/pybest/
