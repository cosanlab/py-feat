Contribute
=======================
## General contribution guidelines
1. Fork the repository on [GitHub](https://github.com/cosanlab/feat). 
2. Install Feat on your machine. 
3. Install `pytest` with `pip install pytest` then run the tests in the repository with `pytest` or `pytest/feat/tests/`. Confirm that all tests pass on your system. If some tests fail, try to find out why they are failing. Common issues may be not having downloaded model files or missing dependencies.
4. Create your feature.
5. Add tests to `feat/tests/`.
6. Run the tests again with `pytest tests/` to make sure everything still passes, including your new feature. If you broke something, edit your feature so that it doesn't break existing code. 
7. Create a pull request to the main repository's `master` branch!

## Model contribution guidelines
There are three key steps to adding a model. 
1. Create a folder for the model you are building in the appropriate model type directory. 
If you are contributing an Action Unit detection model and calling it `mynewmodel`, create a folder in `feat/feat/au_detectors/`. If it's an Emotion detection model, add a folder `feat/feat/emo_detectors/`. 
2. Add your model architecture. 
Your model files could contained in a single (e.g. `feat/emo_detectors/ResMaskNet/resmasknet_test.py`) or they could be separated in to a model file (e.g. `/au_detectors/JAANet/JAA_model`) and a test file (e.g. `/au_detectors/JAANet/JAA_test`). 
3. Your model should be a class that has the appropriate method that the `Detector` can call it. 
For example, AU detectors should have the function `mynewmodel.detect_au()` that can be called. 
4. Add model to list of models in the `Detector` class in `feat/detector.py`. 
5. Upload your trained model weights to an accessible locations (e.g. Google Drive). 
6. Add tests to make sure your model is working. 
7. Create a pull request to the main repository's `master` branch!



```python

```
