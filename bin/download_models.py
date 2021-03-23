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

if os.path.exists(os.path.join(get_resource_path(), "best_ferModel.pth")):
    print("Fex model already exists; skipping download.")
else:
    print("Downloading FEX emotion model.")
    try:
        fex_emotion_model = "https://github.com/cosanlab/feat/releases/download/v0.1/best_ferModel.pth"
        wget.download(fex_emotion_model, get_resource_path(), bar=None)
    # except:
    #    try:
    #        fex_emotion_model = "https://www.dropbox.com/s/d3yhtsqggqcrjl2/fer_emotion_model.h5?dl=1"
    #        wget.download(fex_emotion_model, get_resource_path(), bar=None)
    except:
        print("FeX emotion model failed to download")

    if os.path.exists(os.path.join(get_resource_path(), "best_ferModel.pth")):
        print("\nFEX emotion model downloaded successfully.\n")
    else:
        print("Something went wrong. Model not found in directory.")

if os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml")):
    print("Landmark already exists; skipping download.")
else:
    print("Downloading landmark detection model.")
    try:
        lbfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/lbfmodel.yaml"
        wget.download(lbfmodel, get_resource_path(), bar=None)
    except:
        try:
            lbfmodel = "https://www.dropbox.com/s/cqune0z1bwf79zy/lbfmodel.yaml?dl=1"
            wget.download(lbfmodel, get_resource_path(), bar=None)
        except:
            print("Landmark model failed to download")

if os.path.exists(os.path.join(get_resource_path(), "align_net.pth")):
    print("\nAU Detection Model Parameters downloaded successfully.\n")
else:
    print("Downloading JAA Net AU Occurence model.")
    try:
        jaanet_params = "https://github.com/cosanlab/feat/releases/download/v0.1/JAANetparams.zip"
        wget.download(jaanet_params, get_resource_path(), bar=None)
        with zipfile.ZipFile(os.path.join(get_resource_path(), "JAANetparams.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(get_resource_path()))
    except:
        print("JAA parameters failed to download")

    if os.path.exists(os.path.join(get_resource_path(), "align_net.pth")):
        print("\nAU Detection Model Parameters downloaded successfully.\n")
    else:
        print("Something went wrong. Model not found in directory.")

if os.path.exists(os.path.join(get_resource_path(), "DRMLNetParams.pth")):
    print("\nDRML NET model downloaded successfully.\n")
else:
    try:
        # print("Downloading DRML model.")
        drml_model = "https://github.com/cosanlab/feat/releases/download/v0.1/DRMLNetParams.pth"
        wget.download(drml_model, get_resource_path(), bar=None)
        if os.path.exists(os.path.join(get_resource_path(), "DRMLNetParams.pth")):
            print("\nLandmark detection model downloaded successfully.\n")
        else:
            print("Something went wrong. Model not found in directory.")
    except:
        print("DRML model failed to download.")

if os.path.exists(os.path.join(get_resource_path(), "FaceBoxesProd.pth")):
    print("\nFaceBox model downloaded successfully.\n")
else:
    try:
        # print("Downloading nFaceBox model.")
        facebox_model = "https://github.com/cosanlab/feat/releases/download/v0.1/FaceBoxesProd.pth"
        wget.download(facebox_model, get_resource_path(), bar=None)
        if os.path.exists(os.path.join(get_resource_path(), "FaceBoxesProd.pth")):
            print("\nFaceBox model downloaded successfully.\n")
        else:
            print("Something went wrong. Model not found in directory.")
    except:
        print("FaceBox model failed to download.")

if os.path.exists(os.path.join(get_resource_path(), "onet.npy")):
    print("\nMTCNN O Net model downloaded successfully.\n")
else:
    try:
        # print("Downloading MTCNN ONet model.")
        onet_model = "https://github.com/cosanlab/feat/releases/download/v0.1/onet.npy"
        wget.download(onet_model, get_resource_path(), bar=None)
        # if os.path.exists(os.path.join(get_resource_path(), "onet.npy")):
        #     print("\nMTCNN Onet model downloaded successfully.\n")
        # else:
        #     print("Something went wrong. Model not found in directory.")
    except:
        print("MTCNN Onet model failed to download.")

if os.path.exists(os.path.join(get_resource_path(), "pnet.npy")):
    print("\nMTCNN PNet model downloaded successfully.\n")
else:
    try:
        print("Downloading MTCNN pnet model.")
        pnet_model = "https://github.com/cosanlab/feat/releases/download/v0.1/pnet.npy"
        wget.download(pnet_model, get_resource_path(), bar=None)
        if os.path.exists(os.path.join(get_resource_path(), "pnet.npy")):
            print("\nMTCNN pnet model downloaded successfully.\n")
        else:
            print("Something went wrong. Model not found in directory.")
    except:
        print("MTCNN pnet model failed to download.")     

if os.path.exists(os.path.join(get_resource_path(), "rnet.npy")):
    print("\nMTCNN rnet model downloaded successfully.\n")
else:
    try:
        print("Downloading MTCNN rnet model.")
        rnet_model = "https://github.com/cosanlab/feat/releases/download/v0.1/rnet.npy"
        wget.download(rnet_model, get_resource_path(), bar=None)
        if os.path.exists(os.path.join(get_resource_path(), "rnet.npy")):
            print("\nMTCNN rnet model downloaded successfully.\n")
        else:
            print("Something went wrong. Model not found in directory.")
    except:
        print("MTCNN rnet model failed to download.")     

if os.path.exists(os.path.join(get_resource_path(), "mobilenet0.25_Final.pth")):
    print("\nRetinaFace model downloaded successfully.\n")
else:
    try:
        #print("Downloading RetinaFace model.")
        retin_model = "https://github.com/cosanlab/feat/releases/download/v0.1/mobilenet0.25_Final.pth"
        wget.download(retin_model, get_resource_path(), bar=None)
        # if os.path.exists(os.path.join(get_resource_path(), "mobilenet0.25_Final.pth")):
        #     print("\nRetinaFace model downloaded successfully.\n")
        # else:
        #     print("Something went wrong. Model not found in directory.")
    except:
        print("RetinaFace model failed to download.")    

if os.path.exists(os.path.join(get_resource_path(), "mobilenet_224_model_best_gdconv_external.pth.tar")):
    print("\nmobilenet model downloaded successfully.\n")
else:
    try:
        #print("Downloading mobilenet model.")
        mble_model = "https://github.com/cosanlab/feat/releases/download/v0.1/mobilenet_224_model_best_gdconv_external.pth.tar"
        wget.download(mble_model, get_resource_path(), bar=None)
        # if os.path.exists(os.path.join(get_resource_path(), "mobilenet_224_model_best_gdconv_external.pth.tar")):
        #     print("\nmobilenet model downloaded successfully.\n")
        # else:
        #     print("Something went wrong. Model not found in directory.")
    except:
        print("mobilenet model failed to download.")    

if os.path.exists(os.path.join(get_resource_path(), "pfld_model_best.pth.tar")):
    print("\nPFLD model downloaded successfully.\n")
else:
    try:
        #print("Downloading PFLD model.")
        pfld_model = "https://github.com/cosanlab/feat/releases/download/v0.1/pfld_model_best.pth.tar"
        wget.download(pfld_model, get_resource_path(), bar=None)
        # if os.path.exists(os.path.join(get_resource_path(), "pfld_model_best.pth.tar")):
        #     print("\nPFLD model downloaded successfully.\n")
        # else:
        #     print("Something went wrong. Model not found in directory.")
    except:
        print("PFLD model failed to download.")    

if os.path.exists(os.path.join(get_resource_path(), "mobilefacenet_model_best.pth.tar")):
    print("\nMobileFaceNet model downloaded successfully.\n")
else:
    try:
        #print("Downloading MobileFaceNet model.")
        mbfa_model = "https://github.com/cosanlab/feat/releases/download/v0.1/mobilefacenet_model_best.pth.tar"
        wget.download(mbfa_model, get_resource_path(), bar=None)
        # if os.path.exists(os.path.join(get_resource_path(), "mobilefacenet_model_best.pth.tar")):
        #     print("\nMobileFaceNet model downloaded successfully.\n")
        # else:
        #     print("Something went wrong. Model not found in directory.")
    except:
        print("MobileFaceNet model failed to download.")   

if os.path.exists(os.path.join(get_resource_path(), "ResMaskNet_Z_resmasking_dropout1_rot30.pth")):
    print("\nResidualMaskingNetwork model downloaded successfully.\n")
else:
    try:
        resmasknetmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/ResMaskNet_Z_resmasking_dropout1_rot30.pth"
        wget.download(resmasknetmodel, get_resource_path(), bar=None)
    except:
        print("ResidualMaskingNetwork model failed to download.")   

if os.path.exists(os.path.join(get_resource_path(), "svm_568.joblib")):
    print("\nSVM Action Unit model downloaded successfully.\n")
else:
    try:
        svmmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/svm_568.joblib"
        wget.download(svmmodel, get_resource_path(), bar=None)
    except:
        print("SVM Action Unit model failed to download.")  

if os.path.exists(os.path.join(get_resource_path(), "RF_568.joblib")):
    print("\nRandom Forest Action Unit model downloaded successfully.\n")
else:
    try:
        rfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/RF_568.joblib"
        wget.download(rfmodel, get_resource_path(), bar=None)
    except:
        print("RF Action Unit model failed to download.")  

if os.path.exists(os.path.join(get_resource_path(), "Logistic_520.joblib")):
    print("\nLogistic Regression Action Unit model downloaded successfully.\n")
else:
    try:
        logmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/Logistic_520.joblib"
        wget.download(logmodel, get_resource_path(), bar=None)
    except:
        print("Logistic Action Unit model failed to download.")  

if os.path.exists(os.path.join(get_resource_path(), "hog_pca_all_emotio.joblib")):
    print("\nPCA model downloaded successfully.\n")
else:
    try:
        pcamodel = "https://github.com/cosanlab/feat/releases/download/v0.1/hog_pca_all_emotio.joblib"
        wget.download(pcamodel, get_resource_path(), bar=None)
    except:
        print("PCA model failed to download.")  

