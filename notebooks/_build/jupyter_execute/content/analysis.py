# Analysis

## How to analyze facial expression data with Feat.

*Written by Jin Hyun Cheong*

Here we will be using a sample dataset by David Watson on ["A Data-Driven Characterisation Of Natural Facial Expressions When Giving Good And Bad News"](https://journals.plos.org/ploscompbiol/article/peerReview?id=10.1371/journal.pcbi.1008335) by Watson & Johnston 2020. The full dataset is available on [OSF](https://osf.io/6tbwj/).

First, we download the necessary files & videos. 

import subprocess
files_to_download = ["4c5mb", "n6rt3", "3gh8v", "twqxs", "nc7d9", "nrwcm", "2rk9c", "mxkzq", "c2na7", "wj7zy", "mxywn", 
                     "6bn3g", "jkwsp", "54gtv", "c3hpm", "utdqj", "hpw4a", "94swe", "qte5y", "aykvu", "3d5ry"]

for fid in files_to_download:
    subprocess.run(f"wget --content-disposition https://osf.io/{fid}/download".split())

Check that videos have been downloaded and the attributes file, `clip_attrs.csv) explaining 

import os, glob
import numpy as np
import pandas as pd

clip_attrs = pd.read_csv("clip_attrs.csv")
videos = np.sort(glob.glob("*.mp4"))
print(videos)

Process each video using our detector. 

from feat import Detector
detector = Detector(au_model = "jaanet", emotion_model = "resmasknet")
for video in videos: 
    detector.detect_video(video, outputFname = video.replace(".mp4", ".csv"))

from feat.utils import read_feat
import pandas as pd

for ix ,video in enumerate(videos):
    outputF = video.replace(".mp4", ".csv")
    if ix == 0: 
        fex = read_feat(outputF)
    else:
        fex = pd.concat([out, read_feat(outputF)])
fex = fex.dropna()

# Load in conditions
clip_attrs = pd.read_csv("clip_attrs.csv")
clip_attrs["input"] = clip_attrs.clipN.apply(lambda x: str(x).zfill(3)+".mp4")
input_class_map = dict(zip(clip_attrs.input, clip_attrs['class']))


## Extract features

fex.sessions = fex.input()
average_au_intensity_per_video = fex.extract_mean()
display(average_au_intensity_per_video.head())

## Simple t-test

average_au_intensity_per_video.sessions = average_au_intensity_per_video.index.map(input_class_map)
t, p = average_au_intensity_per_video.iloc[:10].aus().ttest()
pd.DataFrame({"t": t, "p": p}, index= average_au_intensity_per_video.au_columns)

## Prediction

fex.sessions = fex.input().map(input_class_map)

from sklearn.linear_model import LogisticRegression
clf = fex.predict(X=fex.emotions(), y = fex.sessions, model = LogisticRegression, solver="liblinear")
print("score: ", clf.score(fex.emotions(), fex.sessions))
print("coefficients: ")
display(pd.DataFrame(clf.coef_, columns = fex.emotions().columns))

from sklearn.linear_model import LogisticRegression
clf = fex.predict(X=fex.aus(), y = fex.sessions, model = LogisticRegression, solver="liblinear")
print("score: ", clf.score(fex.aus(), fex.sessions))
print("coefficients: ")
display(pd.DataFrame(clf.coef_, columns = fex.aus().columns))

# Regression

fex.sessions = fex.input().map(input_class_map).replace({"gn":.5, "ists":-.5})
X = pd.DataFrame(fex.sessions)
X['intercept'] = 1
b, t, p, df, residuals = fex.regress(X = X, y = fex.emotions())
print("Betas predicting good news estimated for each emotion.")
display(b.round(3).loc[[0]])

fex.sessions = fex.input().map(input_class_map).replace({"gn":.5, "ists":-.5})
X = pd.DataFrame(fex.sessions)
X['intercept'] = 1
b, t, p, df, residuals = fex.regress(X = X, y = fex.aus())
print("Betas predicting good news estimated for each AU.")
display(b.round(3).loc[[0]])