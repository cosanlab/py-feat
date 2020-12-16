#!/usr/bin/env python3
"""
This script will download the necessary models to the feat path. 

Usage
    python3 download_models.py
"""
from feat.utils import get_resource_path
import wget
import os

print("Downloading FEX emotion model.")
fex_emotion_model = "https://github.com/cosanlab/feat/releases/download/v0.1/fer_aug_model.h5"
wget.download(fex_emotion_model, get_resource_path())

if os.path.exists(os.path.join(get_resource_path(), "fer_aug_model.h5")):
    print("\nFEX emotion model downloaded successfully.\n")
else:
    print("Something went wrong. Model not found in directory.")


print("Downloading landmark detection model.")
lbfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/lbfmodel.yaml.txt"
wget.download(lbfmodel, get_resource_path())

if os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml.txt")):
    print("\nLandmark detection model downloaded successfully.\n")
else:
    print("Something went wrong. Model not found in directory.")