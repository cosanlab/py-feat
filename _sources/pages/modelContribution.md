# Contributing new detectors

We would love to developers and researchers to contribute new models to Py-Feat so they can gain a wider audience for their work, while allowing end-users to make more informed choices when selecting models for use. 

## 1. Develop a detector
Currently we have organized our detectors into the following categories:
- Action Unit Detectors
- Emotion Detectors
- Face Detectors
- Landmark Detectors
- Facepose Detectors

Therefore any model you develop should fall into one those categories to be added. If there is a new model  category you would like us to implement (e.g. gaze) please let us know on github!

## 2. Benchmark your detector

We benchmark each model on a standard datasets used in the field. We employ an honor system that your model should not be using these datasets for training. We can also add new benchmark dataset and results as needed.

- Action Unit Detectors: **DIFSA Plus**
- Emotion Detectors: **AffNet**
- Face Detectors: **WIDER**
- Landmark Detectors: **300W**
- Facepose Detectors: **BIWI**

## 3. Add your code to Py-Feat

Adding a new model to the Py-FEAT toolbox is easy if you are familiar with Python, Github, package development, and follow the steps below.

```{note}
It can be helpful to install Py-Feat in development mode so that changes to source files are immediately reflected in any scripts or notebooks that import Py-Feat. To do so, after cloning the code base, install Py-Feat using: `pip install -e .` For more details see the [general contribution guidelines](./contribute.md)
```

Pre-trained models in Py-Feat are organized into sub-folders in the source code based on the detector type:

```
feat/
  au_detectors/
  emo_detectors/
  face_detectors/
  facepose_detectors/
  landmark_detectors/
```

1. Create a folder for the model you are adding in the appropriate model sub-directory 
2. Add your model code which can be a single `.py` file ending in `_test` (e.g. `feat/landmark_detectors/mobilefacenet_test.py`) or a separate sub-directory containing at least 3 files one of which ends in `_model` and the other that ends in `__test` (see `feat/au_detectors/JAANET` for an example): 
    - `__init__.py` (this can be empty)
    - `mynewmodel_model.py` (this should end in `_model`)
    - `mynewmodel_test.py` (this should end in `_test`)
4. Your model should be a class that has the appropriate method that a `Detector` can call. For example, Emotion detectors should have the method `mynewmodel.detect_emotions()` that can be called: 
    ```
    class myNewModel(): 
        ## code to init and load model
        
        detect_emotions(self, imgs, *args **kwargs):
            ## code to detect emotions
        
            return [array with probabilities for 7 emotions]
    ```
5. Add your model to list of models in `feat/pretrained.py` 
6. Upload your trained model weights to an accessible locations (e.g. Google Drive) and add it to `feat/resources/model_list.json`. 
7. Follow the [general contribution guidelines](./contribute.md) to add tests and format your code
8. When your tests pass create a pull-request against the `master`/`main` branch on github!

```{note}
If you enjoy developing/testing in jupyter notebooks, it can be helpful to add the following lines of code into a cell at the top of yoru notebook so that source code changes don't require you to restart the kernel:   
`%load_ext autoreload`  
`%autoreload 2`
```