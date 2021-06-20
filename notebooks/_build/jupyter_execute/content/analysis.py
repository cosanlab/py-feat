#!/usr/bin/env python
# coding: utf-8

# # Preprocessing FEX data
# 
# ## How to preprocess and analyze facial expression data with Feat.
# 
# *Written by Jin Hyun Cheong*
# 
# Here we will be using a sample dataset by David Watson on ["A Data-Driven Characterisation Of Natural Facial Expressions When Giving Good And Bad News"](https://journals.plos.org/ploscompbiol/article/peerReview?id=10.1371/journal.pcbi.1008335) by Watson & Johnston 2020. The full dataset is available on [OSF](https://osf.io/6tbwj/).
# 
# Let's start by installing Py-FEAT if you have not already done so or using this on Google Colab

# In[ ]:


get_ipython().system('pip install -q py-feat')


# First, we download the necessary files & videos. 

# In[1]:


import subprocess
files_to_download = ["4c5mb", "n6rt3", "3gh8v", "twqxs", "nc7d9", "nrwcm", "2rk9c", "mxkzq", "c2na7", "wj7zy", "mxywn", 
                     "6bn3g", "jkwsp", "54gtv", "c3hpm", "utdqj", "hpw4a", "94swe", "qte5y", "aykvu", "3d5ry"]

for fid in files_to_download:
    subprocess.run(f"wget --content-disposition https://osf.io/{fid}/download".split())


# Check that videos have been downloaded and the attributes file, `clip_attrs.csv) explaining 

# In[4]:


import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context("talk")

clip_attrs = pd.read_csv("clip_attrs.csv")
videos = np.sort(glob.glob("*.mp4"))
print(videos)


# Process each video using our detector. 

# In[ ]:


from feat import Detector
detector = Detector(au_model = "rf", emotion_model = "resmasknet")
for video in videos: 
    detector.detect_video(video, outputFname = video.replace(".mp4", ".csv"))


# In[9]:


from feat.utils import read_feat
import pandas as pd

for ix ,video in enumerate(videos):
    outputF = video.replace(".mp4", ".csv")
    if ix == 0: 
        fex = read_feat(outputF)
    else:
        fex = pd.concat([fex, read_feat(outputF)])
fex = fex.dropna()


# In[10]:


# Load in conditions
clip_attrs = pd.read_csv("clip_attrs.csv")
clip_attrs = clip_attrs.assign(input = clip_attrs.clipN.apply(lambda x: str(x).zfill(3)+".mp4"),
                               condition = clip_attrs['class'].replace({"gn":"goodNews", "ists":"badNews"}))
input_class_map = dict(zip(clip_attrs.input, clip_attrs['condition']))
clip_attrs.head()


# ## Extract features
# You can set the `sessions` attribute to provide a grouping of your experimental setup. This could be the name of each video if you want to extract features per video or it could be conditions to extract features per condition. 

# In[11]:


# Extract conditions between the two condtiiosn (gn: good news, ists: bad news)
conditions = dict(zip(clip_attrs.input, clip_attrs['condition']))
fex.sessions = fex.input().map(conditions)
average_au_intensity_per_video = fex.extract_mean()
display(average_au_intensity_per_video.head())


# Or simply extract features per video

# In[12]:


# Extract features per video
fex.sessions = fex.input()
average_au_intensity_per_video = fex.extract_mean()
display(average_au_intensity_per_video.head())


# # Analyzing FEX data
# ## Simple t-test
# You can use a simple t-test to test if the average activation of a certain AU is significantly higher than .5 (chance). The results suggests that AU10 (upper lip raiser), 12 (lip corner puller), and 14 (dimpler) is significantly activitated when providing good news. 

# In[13]:


average_au_intensity_per_video.sessions = average_au_intensity_per_video.index.map(input_class_map)
t, p = average_au_intensity_per_video[average_au_intensity_per_video.sessions=="goodNews"].aus().ttest_1samp(.5)
pd.DataFrame({"t": t, "p": p}, index= average_au_intensity_per_video.au_columns)


# ## Two sample independent t-test
# You can also perform an independent two sample ttest between two sessions which in this case is goodNews vs badNews.

# In[14]:


columns2compare = "mean_AU12"
sessions = ("goodNews", "badNews")
t, p = average_au_intensity_per_video.ttest_ind(col = columns2compare, sessions=sessions)
print(f"T-test between {sessions[0]} vs {sessions[1]}: t={t:.2g}, p={p:.3g}")
sns.barplot(x = average_au_intensity_per_video.sessions, 
            y = columns2compare, 
            data = average_au_intensity_per_video);


# ## Prediction
# If you want to know what combination of features predic the good news or bad news conditions. To investigate this problem, we can train a Logistc Regression model using emotion labels to predict the conditions. Results suggest that detections of happy expressions predict the delivery of good news. 

# In[15]:


fex.sessions = fex.input().map(input_class_map)

from sklearn.linear_model import LogisticRegression
clf = fex.predict(X=fex.emotions(), y = fex.sessions, model = LogisticRegression, solver="liblinear")
print(f"score: {clf.score(fex.emotions(), fex.sessions):.3g}")
print(f"coefficients for predicting class: {clf.classes_[1]}")
display(pd.DataFrame(clf.coef_, columns = fex.emotions().columns))


# ## Regression
# We can also run an fMRI style regression to predict the Action Unit activities from a contrast of conditions. This analysis can be conducted through the `regress` method. In this example, we identify the action units that are significantly more active in the good news versus the bad news conditions.

# In[16]:


fex.sessions = fex.input().map(input_class_map).replace({"goodNews":.5, "badNews":-.5})
X = pd.DataFrame(fex.sessions)
X['intercept'] = 1
b, t, p, df, residuals = fex.regress(X = X, y = fex.aus())
print("Betas predicting good news estimated for each emotion.")
results = pd.concat([b.round(3).loc[[0]].rename(index={0:"betas"}),
                    t.round(3).loc[[0]].rename(index={0:"t-stats"}),
                    p.round(3).loc[[0]].rename(index={0:"p-values"})])
display(results)


# ## Intersubject (or intervideo) correlations
# To compare the similarity of signals over time between subjects or videos, you can use the `isc` method. You can get a sense of how much two signals, such as a certain action unit activity, correlates over time. 
# 
# In this example, we are calculating the ISC over videos. We want to check how similar AU01 activations are across videos so our session is set to the `input` which is the video name. Executing the `isc` method shows that the temporal profile of AU01 activations form two clusters between the goodNews and the badNews conditions. 

# In[18]:


fex.sessions = fex.input()
isc = fex.isc(col = "AU01")
sns.heatmap(isc.corr(), center=0, vmin=-1, vmax=1, cmap="RdBu_r");


# In[ ]:




