# Preprocessing FEX data

## How to preprocess and analyze facial expression data with Feat.

*Written by Jin Hyun Cheong*

Here we will be using a sample dataset by David Watson on ["A Data-Driven Characterisation Of Natural Facial Expressions When Giving Good And Bad News"](https://journals.plos.org/ploscompbiol/article/peerReview?id=10.1371/journal.pcbi.1008335) by Watson & Johnston 2020. The full dataset is available on [OSF](https://osf.io/6tbwj/).

Let's start by installing Py-FEAT if you have not already done so or using this on Google Colab

!pip install -q py-feat

First, we download the necessary files & videos. 

import subprocess
# files_to_download = ["4c5mb", "n6rt3", "3gh8v", "twqxs", "nc7d9", "nrwcm", "2rk9c", "mxkzq", "c2na7", "wj7zy", "mxywn", 
#                      "6bn3g", "jkwsp", "54gtv", "c3hpm", "utdqj", "hpw4a", "94swe", "qte5y", "aykvu", "3d5ry"]

for fid in files_to_download:
    subprocess.run(f"wget --content-disposition https://osf.io/{fid}/download".split())

Check that videos have been downloaded and the attributes file, `clip_attrs.csv) explaining 

%load_ext autoreload
%autoreload 2

import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context("talk")

clip_attrs = pd.read_csv("clip_attrs.csv")
# videos = np.sort(glob.glob("*.mp4"))
videos = np.sort(glob.glob("../../../../watsondata/*.mp4"))[134:]

print(videos)

Process each video using our detector. 

from feat import Detector
detector = Detector(au_model = "rf", emotion_model = "resmasknet")
for video in videos: 
    detector.detect_video(video, outputFname = video.replace(".mp4", ".csv"))

Load the saved output data using the `read_feat` function

from feat.utils import read_feat
import pandas as pd

for ix ,video in enumerate(videos):
    outputF = video.replace(".mp4", ".csv")
    if ix == 0: 
        fex = read_feat(outputF)
    else:
        fex = pd.concat([fex, read_feat(outputF)])
fex = fex.dropna()

# Load in conditions
clip_attrs = pd.read_csv("clip_attrs.csv")
clip_attrs = clip_attrs.assign(input = clip_attrs.clipN.apply(lambda x: str(x).zfill(3)+".mp4"),
                               condition = clip_attrs['class'].replace({"gn":"goodNews", "ists":"badNews"}))
input_class_map = dict(zip(clip_attrs.input, clip_attrs['condition']))
clip_attrs.head()

## Extract features
You can set the `sessions` attribute to provide a grouping of your experimental setup. This could be the name of each video if you want to extract features per video or it could be conditions to extract features per condition. 

# Extract conditions between the two condtiiosn (gn: good news, ists: bad news)
conditions = dict(zip(clip_attrs.input, clip_attrs['condition']))
fex.sessions = fex.input().map(conditions)
average_au_intensity_per_video = fex.extract_mean()
display(average_au_intensity_per_video.head())

Or simply extract features per video

# Extract features per video
fex.sessions = fex.input()
average_au_intensity_per_video = fex.extract_mean()
display(average_au_intensity_per_video.head())

# Analyzing FEX data
## Simple t-test
You can use a simple t-test to test if the average activation of a certain AU is significantly higher than .5 (chance). The results suggests that AU10 (upper lip raiser), 12 (lip corner puller), and 14 (dimpler) is significantly activitated when providing good news. 

average_au_intensity_per_video.sessions = average_au_intensity_per_video.index.map(input_class_map)
t, p = average_au_intensity_per_video[average_au_intensity_per_video.sessions=="goodNews"].aus().ttest_1samp(.5)
pd.DataFrame({"t": t, "p": p}, index= average_au_intensity_per_video.au_columns)

## Two sample independent t-test
You can also perform an independent two sample ttest between two sessions which in this case is goodNews vs badNews.

import matplotlib.pyplot as plt

columns2compare = "mean_AU12"
sessions = ("goodNews", "badNews")
t, p = average_au_intensity_per_video.ttest_ind(col = columns2compare, sessions=sessions)
print(f"T-test between {sessions[0]} vs {sessions[1]}: t={t:.2g}, p={p:.3g}")
with sns.plotting_context("poster", font_scale=1):
    f,ax = plt.subplots(figsize=(5,7))
    sns.barplot(x = average_au_intensity_per_video.sessions, 
                y = columns2compare, 
                data = average_au_intensity_per_video);
    ax.set(ylim=[0,1.3], yticks=[0, .4, .8, 1.2], ylabel="mean AU12\nintensity")
    sns.despine()

fig_df = average_au_intensity_per_video.reset_index()[["index", "mean_AU01", "mean_AU12","mean_AU17"]]
fig_df.columns = ['index', '1', '12', '17']
fig_df = fig_df.melt(id_vars="index", 
                       value_vars= ['1', '12', '17'],
                       var_name = "Action Units",
                       value_name = "Average intensity")
fig_df = fig_df.assign(condition=fig_df['index'].map(input_class_map))

with sns.plotting_context("poster", font_scale=1.5):
    f,ax = plt.subplots(figsize=(10,5))
    sns.barplot(x="Action Units", y="Average intensity", data=fig_df, hue='condition', ax=ax)
    ax.set(ylim=[0,1.2], yticks=[0, .4, .8, 1.2], ylabel="Average intensity")
    sns.despine()
plt.legend(bbox_to_anchor=(0.5, 1.2), loc=9, borderaxespad=0., ncol=2, fontsize="xx-large", frameon=False)

plt.legend?

## Prediction
If you want to know what combination of features predic the good news or bad news conditions. To investigate this problem, we can train a Logistc Regression model using emotion labels to predict the conditions. Results suggest that detections of happy expressions predict the delivery of good news. 

fex.sessions = fex.input().map(input_class_map)

from sklearn.linear_model import LogisticRegression
clf = fex.predict(X=fex.emotions(), y = fex.sessions, model = LogisticRegression, solver="liblinear")
print(f"score: {clf.score(fex.emotions(), fex.sessions):.3g}")
print(f"coefficients for predicting class: {clf.classes_[1]}")
display(pd.DataFrame(clf.coef_, columns = fex.emotions().columns))

Also run the same analysis with Action Units to predict goodNews

fex.sessions = fex.input().map(input_class_map)

from sklearn.linear_model import LogisticRegression
clf = fex.predict(X=fex.aus(), y = fex.sessions, model = LogisticRegression, solver="liblinear")
print(f"score: {clf.score(fex.aus(), fex.sessions):.3g}")
print(f"coefficients for predicting class: {clf.classes_[1]}")
display(pd.DataFrame(clf.coef_, columns = fex.aus().columns))

from feat import Fex
aus = Fex(pd.DataFrame(clf.coef_, columns = fex.aus().columns), 
          au_columns = fex.aus().columns,
         detector="feat")
aus.plot_aus(0, feature_range=(0,2), muscles = {'all': 'heatmap'})

aus = Fex(pd.DataFrame(-clf.coef_, columns = fex.aus().columns), 
          au_columns = fex.aus().columns,
         detector="feat")
aus.plot_aus(0, feature_range=(0,2), muscles = {'all': 'heatmap'})

from sklearn.linear_model import LogisticRegression

from numpy.random import default_rng
average_au_intensity_per_video
ys = np.array(["goodNews"]*10 + ["badNews"]*10)
ys = pd.DataFrame({"y": ys}, index = average_au_intensity_per_video.index)

rng = default_rng(1)
goodNewsIdx = list(average_au_intensity_per_video.index[:10])
rng.shuffle(goodNewsIdx)
badNewsIdx = list(average_au_intensity_per_video.index[10:])
rng.shuffle(badNewsIdx)

scores = []
for gix, bix in zip(goodNewsIdx, badNewsIdx):
    trainX = average_au_intensity_per_video.query("index!=@gix and index!=@bix")
    trainY = ys.query("index!=@gix and index!=@bix").values.ravel()

    testX = average_au_intensity_per_video.query("index==@gix or index==@bix")
    testY = ys.query("index==@gix or index==@bix").values.ravel()

    clf = trainX.predict(X = trainX.au_columns, y = trainY, model = LogisticRegression)
    print(clf.predict(testX.aus()), testY)
    scores.append(clf.score(testX.aus(), testY))

clf = average_au_intensity_per_video.predict(average_au_intensity_per_video.au_columns, 
                                             ys.values.ravel(), 
                                             model = LogisticRegression)
print(scores)
print(f"coefficients for predicting class: {clf.classes_[1]}")
display(pd.DataFrame(clf.coef_, columns = fex.aus().columns))



from feat import Fex
aus = Fex(pd.DataFrame(clf.coef_, columns = fex.aus().columns), 
          au_columns = fex.aus().columns,
         detector="feat")
aus.plot_aus(0, feature_range=(0,1), muscles = {'all': 'heatmap'})

aus = Fex(pd.DataFrame(-clf.coef_, columns = fex.aus().columns), 
          au_columns = fex.aus().columns,
         detector="feat")
aus.plot_aus(0, feature_range=(0,1), muscles = {'all': 'heatmap'})

## Regression
We can also run an fMRI style regression to predict the Action Unit activities from a contrast of conditions. This analysis can be conducted through the `regress` method. In this example, we identify the action units that are significantly more active in the good news versus the bad news conditions.

fex.sessions = fex.input().map(input_class_map).replace({"goodNews":.5, "badNews":-.5})
X = pd.DataFrame(fex.sessions)
X['intercept'] = 1
b, t, p, df, residuals = fex.regress(X = X, y = fex.aus())
print("Betas predicting good news estimated for each emotion.")
results = pd.concat([b.round(3).loc[[0]].rename(index={0:"betas"}),
                    t.round(3).loc[[0]].rename(index={0:"t-stats"}),
                    p.round(3).loc[[0]].rename(index={0:"p-values"})])
display(results)

## Intersubject (or intervideo) correlations
To compare the similarity of signals over time between subjects or videos, you can use the `isc` method. You can get a sense of how much two signals, such as a certain action unit activity, correlates over time. 

In this example, we are calculating the ISC over videos. We want to check how similar AU01 activations are across videos so our session is set to the `input` which is the video name. Executing the `isc` method shows that the temporal profile of AU01 activations form two clusters between the goodNews and the badNews conditions. 

fex.sessions = fex.input()
isc = fex.isc(col = "AU01")
sns.heatmap(isc.corr(), center=0, vmin=-1, vmax=1, cmap="RdBu_r");

