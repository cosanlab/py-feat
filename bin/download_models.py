#!/usr/bin/env python3
"""
This script will download the necessary models to the feat path. 

Usage
    python3 download_models.py
"""
from feat.utils import get_resource_path
import wget
import os
import zipfile

print("Downloading FEX emotion model.")
fex_emotion_model = "https://github.com/cosanlab/feat/releases/download/v0.1/fer_aug_model.h5"
if not os.path.exists(os.path.join(get_resource_path(), "fer_aug_model.h5")):
    wget.download(fex_emotion_model, get_resource_path())

if os.path.exists(os.path.join(get_resource_path(), "fer_aug_model.h5")):
    print("\nFEX emotion model downloaded successfully.\n")
else:
    print("Something went wrong. Model not found in directory.")


print("Downloading landmark detection model.")
lbfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/lbfmodel.yaml"

if not os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml")):
    wget.download(lbfmodel, get_resource_path())

if os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml")):
    print("\nLandmark detection model downloaded successfully.\n")
else:
    print("Something went wrong. Model not found in directory.")


print("Downloading AU Detection Model Parameters.")
jaanet_params = "https://github.com/cosanlab/feat/releases/download/v0.1/JAANetparams.zip"

if not (os.path.exists(os.path.join(get_resource_path(), "align_net.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "au_net.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "global_au_feat.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "local_attention_refine.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "local_au_net.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "region_learning.pth"))):
    wget.download(jaanet_params, get_resource_path())

with zipfile.ZipFile(os.path.join(get_resource_path(), "JAANetparams.zip"), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(get_resource_path()))

if os.path.exists(os.path.join(get_resource_path(), "align_net.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "au_net.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "global_au_feat.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "local_attention_refine.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "local_au_net.pth")) and \
    os.path.exists(os.path.join(get_resource_path(), "region_learning.pth")):

    print("\nJAANet Params weight downloaded successfully.\n")
else:
    print("Something went wrong. JAA Net Model not found in directory.")

print("Downloading DRML model.")
drml_model = "https://github.com/cosanlab/feat/releases/download/v0.1/DRMLNetParams.pth"

if not os.path.exists(os.path.join(get_resource_path(), "DRMLNetParams.pth")):
    wget.download(drml_model, get_resource_path())

if os.path.exists(os.path.join(get_resource_path(), "DRMLNetParams.pth")):
    print("\nLandmark detection model downloaded successfully.\n")
else:
    print("Something went wrong. Model not found in directory.")