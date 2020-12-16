#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""
from feat.detector import Detector
from feat.utils import get_resource_path
from .utils import get_test_data_path
import pandas as pd
import feat
import os
import wget

def test_models():
    print("Downloading FEX emotion model.")
    fex_emotion_model = "https://github.com/cosanlab/feat/releases/download/v0.1/fer_aug_model.h5"
    wget.download(fex_emotion_model, get_resource_path())

    if os.path.exists(os.path.join(get_resource_path(), "fer_aug_model.h5")):
        print("\nFEX emotion model downloaded successfully.\n")
    else:
        print("Something went wrong. Model not found in directory.")

    print("Downloading landmark detection model.")
    lbfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/lbfmodel.yaml"
    wget.download(lbfmodel, get_resource_path())

    if os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml")):
        print("\nLandmark detection model downloaded successfully.\n")
    else:
        print("Something went wrong. Model not found in directory.")

    emotion_model = "fer_aug_model.h5"
    emotion_model_path = os.path.join(get_resource_path(), emotion_model)
    print("PATH TO EMOTION MODEL",emotion_model_path)
    assert os.path.exists(emotion_model_path)==True

    landmark_model = "lbfmodel.yaml"
    landmark_model_path = os.path.join(get_resource_path(), landmark_model)
    assert os.path.exists(landmark_model_path)==True

def test_detector():
    detector = Detector(n_jobs=1)
    assert detector['n_jobs']==1
    assert type(detector)==Detector

    inputFname = os.path.join(get_test_data_path(), "input.jpg")
    out = detector.detect_image(inputFname = inputFname)
    assert type(out) == pd.DataFrame
    assert len(out) == 1
    assert out.happiness.values[0] > 0 

    outputFname = os.path.join(get_test_data_path(), "output.csv")
    out = detector.detect_image(inputFname=inputFname, outputFname=outputFname)
    assert out
    assert os.path.exists(outputFname)
    out = pd.read_csv(outputFname)
    assert out.happiness.values[0] > 0 