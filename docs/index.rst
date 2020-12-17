FEAT
===============================

Facial Expression Analysis Toolbox

FEAT is a suite for facial expressions (FEX) research written in Python. This package includes tools to extract emotional facial expressions (e.g., happiness, sadness, anger), facial muscle movements (e.g., action units), and facial landmarks, from videos and images of faces, as well as methods to preprocess, analyze, and visualize FEX data. 

Installation
------------

Option 1 
Clone the repository  

.. code-block:: python
    git clone https://github.com/cosanlab/feat.git  
    cd feat
    python setup.py install

Option 2  

.. code-block:: python

    pip install git+https://github.com/cosanlab/feat

Verify models have been downloaded.

.. code-block:: python
    python download_models.py


.. toctree::
   :maxdepth: 2
   :hidden:

   feat
   usage
   api

:doc:`Usage <usage>`
-------------------

1. Detecting FEX features

.. code-block:: python
    from feat.detector import Detector
    detector = Detector() 
    out = detector.detect_image("input.png")
    out = detector.detect_video("input.mp4")

2. Extracting features from FEX 

.. code-block:: python
    # Code goes here

3. Analyzing FEX data 

.. code-block:: python
    # Code goes here

4. Visualizing FEX data

.. code-block:: python
    # Code goes here


:doc:`API <api>`
----------------

    .. automodule:: feat.data
    :members: