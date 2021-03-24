# Contributing new detectors

*written by Jin Hyun Cheong*  

In this tutorial we outline the necessary steps for users to contribute new models to the Py-FEAT toolbox. This will allow developers to gain a wide audience for their work and researchers can also benefit from making informed choices for selecting models. 

## 1. Develop a model
Currently we have organized our models into the following categories:
- au_detectors
- emotion_detectors
- face_detectors
- landmark_detectors

Therefore any model you develop that fall into those categories can be added. If there is a new category you would like us to implement (e.g. gaze, head pose) please submit a pull request.

## 2. Benchmark your model. 
We benchmark each model on a specified dataset. These are quite standard datasets. We employ an honor system that your model should not be using these datasets for training. We will also add new benchmark dataset and results as needed. 

- au_detectors: DISFA Plus
- emotion_detectors: AffNet
- face_detectors: WIDER
- landmark_detectors: 300W

## 3. Add models to the toolbox. 
Adding a new model to the Py-FEAT toolbox is easy if you are familiar with Python, Github, and package development. Here is 10 simple steps to adding a new model. 
1. For the Py-FEAT repository from https://github.com/cosanlab/feat
2. Install the package to your computer in development mode. This will allow you to quickly test your code as you develop. 
`python setup.py install -e .`
3. Make sure everything is working using `pytest`
4. If you are testing in a Jupyter Notebook enviornment, enable autoreload so that every change you make on the toolbox is immediately loaded. 
```
%load_ext autoreload
%autoreload 2
```
5. Create a directory for your model.
If you are adding a new emotion detector, and your model name is MYNEWMODEL, create folder `feat/emo_detectors/MYNEWMODEL`
6. Add an empty `__init__.py` file so that it is recognized as a submodule by Python. 
`touch __init__.py`
7. Create the model file, class, and methods. 
For example, this model file could be named `mynewmodel.py` which defines a class called `myNewModel()` which has a method `detect_emo()`
```
class myNewModel(): 
    ## code to init and load model
    
    detect_emo(self, imgs, *args **kwargs):
        ## code to detect emotions
    
        return [array with probabilities for 7 emotions]
```

8. Add your model info and download url link to `model_list.json`
This will allow us to download the model files when necessary. 

9. Lastly, add your method to `detector.py` so that it is accessible. 

10. Before submitting a pull request please run `pytest` to make sure other parts of the package is not broken. 



```python

```
