# Training AU visualization model
You will first need to gather the datasets for training. In this tutorial we use the datasets EmotioNet, DISFA Plus, and BP4d. After you download each model you should extract the labels and landmarks from each dataset. Detailed code on how to do that is described at the bottom of this tutorial. Once you have the labels and landmark files for each dataset you can train the AU visualization model with the following. 

%matplotlib inline
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from feat.plotting import predict, plot_face
from feat.utils import registration, neutral
from natsort import natsorted
import os, glob 
import pandas as pd, numpy as np
import seaborn as sns
sns.set_style("white")

base_dir = "/Storage/Projects/feat_benchmark/scripts/jcheong/openface_train"

labels_emotionet = pd.read_csv(os.path.join(base_dir, "emotionet_labels.csv"))
landmarks_emotionet = pd.read_csv(os.path.join(base_dir, "emotionet_landmarks.csv"))
print("EmotioNet: ", len(labels_emotionet))
labels_disfaplus = pd.read_csv(os.path.join(base_dir, "disfaplus_labels.csv"))
landmarks_disfaplus = pd.read_csv(os.path.join(base_dir, "disfaplus_landmarks.csv"))
# Disfa is rescaled to 0 - 1
disfaplus_aus = [col for col in labels_disfaplus.columns if "AU" in col]
labels_disfaplus[disfaplus_aus] = labels_disfaplus[disfaplus_aus].astype('float')/5
print("DISFA Plus: ", len(labels_disfaplus))
labels_bp4d = pd.read_csv(os.path.join(base_dir, "bp4d_labels.csv"))
landmarks_bp4d = pd.read_csv(os.path.join(base_dir, "bp4d_landmarks.csv"))
bp4d_pruned_idx = labels_bp4d.replace({9: np.nan})[au_cols].dropna(axis=1).index
print("BP4D: ", len(labels_bp4d))

We aggregate the datasets and specify the AUs we want to train. 

labels = pd.concat([
                    labels_emotionet.replace({999: np.nan}), 
                    labels_disfaplus,
                    labels_bp4d.replace({9: np.nan}).iloc[bp4d_pruned_idx,:]
                   ]).reset_index(drop=True)
landmarks = pd.concat([
                       landmarks_emotionet, 
                       landmarks_disfaplus,
                       landmarks_bp4d.iloc[bp4d_pruned_idx,:]
                      ]).reset_index(drop=True)

landmarks = landmarks.iloc[labels.index]

au_cols = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 28, 43]
au_cols = [f"AU{au}" for au in au_cols]
labels = labels[au_cols].fillna(0)

We train our model using PLSRegression with a minimum of 500 samples for each AU activation. We evaluate the model in a 3-fold split and retrain the model with all the data which is distributed with the package.

min_pos_sample = 500

print('Pseudo balancing samples')
balY = pd.DataFrame()
balX = pd.DataFrame()
for AU in labels[au_cols].columns:
    if np.sum(labels[AU]==1) > min_pos_sample:
        replace = False
    else:
        replace = True
    newSample = labels[labels[AU]>.5].sample(min_pos_sample, replace=replace, random_state=0)
    balX = pd.concat([balX, newSample])
    balY = pd.concat([balY, landmarks.loc[newSample.index]])
X = balX[au_cols].values
y = registration(balY.values, neutral)

# Model Accuracy in KFold CV
print("Evaluating model with KFold CV")
n_components=len(au_cols)
kf = KFold(n_splits=3)
scores = []
for train_index, test_index in kf.split(X):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    clf = PLSRegression(n_components=n_components, max_iter=2000)
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_test,y_test))
print('3-fold accuracy mean', np.round(np.mean(scores),2))

# Train real model
clf = PLSRegression(n_components=n_components, max_iter=2000)
clf.fit(X,y)
print('N_comp:',n_components,'Rsquare', np.round(clf.score(X,y),2))

We visualize the results of our model. The regression was trained on labels 0-1 so we do not recommend exceeding 1 for the intensity. Setting the intensity to 2 will exaggerate the face and anything beyond that might give you strange faces. 

# Plot results for each action unit
f,axes = plt.subplots(5,4,figsize=(12,18))
axes = axes.flatten()
# Exaggerate the intensity of the expression for clearer visualization. 
# We do not recommend exceeding 2. 
intensity = 2
for aui, auname in enumerate(axes):
    try:
        auname=au_cols[aui]
        au = np.zeros(clf.n_components)
        au[aui] = intensity
        predicted = clf.predict([au]).reshape(2,68)
        plot_face(au=au, model=clf,
                  vectorfield={"reference": neutral.T, 'target': predicted,
                               'color':'r','alpha':.6},
                 ax = axes[aui])
        axes[aui].set(title=auname)
    except:
        pass
    finally:
        ax = axes[aui]
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

## Interactive visualization

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual


def f(x):
    return x

interact(f, x=10);


Here is how we would export our model into an h5 format which can be loaded using our load_h5 function. 

# save out trained model
# import h5py
# hf = h5py.File('../feat/resources/pyfeat_aus_to_landmarks.h5', 'w')
# hf.create_dataset('coef', data=clf.coef_)
# hf.create_dataset('x_mean', data=clf._x_mean)
# hf.create_dataset('x_std', data=clf._x_std)
# hf.create_dataset('y_mean', data=clf._y_mean)
# hf.close()

