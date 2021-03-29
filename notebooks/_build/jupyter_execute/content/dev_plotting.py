# Plotting examples
*written by Jin Hyun Cheong*

Included in the toolbox are two models for Action Units to landmark visualization. The 'pyfeat_aus_to_landmarks.h5' model was created by using landmarks extracted using Py-FEAT to align each face in the dataset to a neutral face with numpy's least squares function. Then the PLS model was trained using Action Unit labels to predict transformed landmark data. 

Draw a standard neutral face. Figsize can be altered but a ratio of 4:5 is recommended. 

%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = 'retina'
intensity = 2

# Load modules
%matplotlib inline
from feat.plotting import plot_face, predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_face(au=np.zeros(20))

## Draw lineface using input vector

Affectiva vectors should be divided by twenty for use with our 'blue' model. 

from feat.plotting import plot_face, predict
import numpy as np
import matplotlib.pyplot as plt



# Add data, AU is ordered as such: 
# AU1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 28, 43

# Activate AU1: Inner brow raiser 
au = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

neutral = predict(np.zeros(len(au)))

vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}

# Plot face
fig, axes = plt.subplots(1,2)
plot_face(model=None, vectorfield = vectors,
          ax = axes[0], au = np.zeros(20), color='k', linewidth=1, linestyle='-')
plot_face(model=None, vectorfield = vectors,
          ax = axes[1], au = np.array(au), color='k', linewidth=1, linestyle='-')

intensity=3
feature_range = (0, 2) 
au = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

vectors = {'target': predict(au, feature_range=feature_range),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, 
          muscles = {'all': 'heatmap'}, 
          feature_range=feature_range, 
          au = np.array(au), color='k', linewidth=1, linestyle='-')

feature_range = (0,2)

f,axes = plt.subplots(1, 4, figsize=(15,4))
ax = axes[0]
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
# Add data, AU is ordered as such: 
# AU 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 28, 43

# Activate AU1: Inner brow raiser 
au = np.array([intensity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vectors = {'target': predict(au, feature_range=feature_range),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'}, feature_range=feature_range,
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("AU1: Inner brow raiser")

ax = axes[1]
au = np.array([0, 0, 0, 0, 0, 0, 0, 0, intensity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vectors = {'target': predict(au, feature_range=feature_range),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},feature_range=feature_range,
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("AU12: Lip corner puller")

ax = axes[2]
au = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, intensity, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vectors = {'target': predict(au, feature_range=feature_range),
           'reference':  neutral, 'color': 'blue'}
plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},feature_range=feature_range,
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("AU15: Lip corner depressor")

ax = axes[3]
au = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, intensity])
vectors = {'target': predict(au, feature_range=feature_range),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},feature_range=feature_range,
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("AU43: Eye closer")

f,axes = plt.subplots(1, 4, figsize=(15,4))
ax = axes[0]
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
# Add data, AU is ordered as such: 
# AU 1, 2, 4, 5, 6, 
# 7, 9, 10, 12, 14, 
# 15, 17, 18, 20, 23, 
# 24, 25, 26, 28, 43
intensity = 2
# Activate AU1: Inner brow raiser 
au = np.array([0, 
               0, 
               0, 
               0, 
               intensity/2, 
               0, 
               0, 
               0, 
               intensity, 
               0, 
               0, 
               0, 
               0, 
               0, 
               0, 
               0, 
               0, 
               0, 
               0, 
               0])
vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("Happiness")

ax = axes[1]
au = np.array([intensity/2, 
               0, 
               intensity/2, 
               0, 0, 0, 0, 0, 0, 0, intensity, 0, 0, 0, 0, 0, 0, 0, 0, 0])
vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("Sadness")

ax = axes[2]
au = np.array([intensity, intensity, 0, intensity, 0, 
               0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 
               0, 0, intensity/2, 0, 0])
vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}
plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("Surprise")

ax = axes[3]

# FEAR
au = np.array([intensity, intensity, intensity, intensity, 0, 
               intensity, 0, 0, 0, 0, 
               0, 0, 0, intensity/2, 0, 
               0, 0, intensity/2, 0, 0])
vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}

plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},
          ax = ax, au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("Fear")

au = np.array([intensity, intensity, 0, intensity, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, intensity, 0, 0])
vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}
plot_face(model=None, vectorfield = vectors, muscles = {'all': 'heatmap'},
           au = np.array(au), color='k', linewidth=1, linestyle='-')
ax.set_title("Surprise")


aus, xs, ys = [], [], []
AUname = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 28, 43]
AUdesc = ["inner brow raiser", "outer brow raiser", "brow lowerer", "upper lid raiser", "cheek raiser",
         "lid tightener", "nose wrinkler", "upper lip raiser", "lip corner puller", "dimpler",
         "lip corner depressor", "chin raiser", "lip puckerer", "lip stretcher", "lip tightener",
         "lip pressor", "lips part", "jaw drop", "lip suck", "eyes closed"]
df = pd.DataFrame()
for intensity in np.arange(0, 3.1 ,.5):
    for au in range(20):
        aus = np.zeros(20)
        aus[au] = intensity
        xs, ys = predict(aus)     
        AUtitle = f"{AUname[au]}\n"+AUdesc[au]
        _df = pd.DataFrame({"xs": xs, 
                          "ys": ys, 
                          "coord_id": range(68),
                          "intensity": intensity, 
                          "AU": AUtitle,
                          "AUidx": au,
                          "color": "k"})
        
        idxs = [17, 23, 29, 39, 46, 53]
        for idx in idxs:
            df1 = _df.iloc[:idx].copy()
            df1.loc[-1] = [np.nan, np.nan, np.nan, intensity, AUtitle, au, "k"]
            df2 = _df.iloc[idx:]
            df2.index = df2.index+1
            _df = pd.concat([df1.reset_index(drop=True), df2])
            
        df = pd.concat([df, _df])

def visualize_autolandmark(df):
    import plotly.express as px
    import plotly.graph_objects as go
    fig = px.line(df, x="xs", y="ys", animation_frame="intensity", 
                  color="color", color_discrete_map={"k":"black"},
                     hover_name="AU", facet_col="AU", facet_col_wrap=4,
                     range_x=[30,170], range_y=[250, 80],
                     title= "Action Unit activation to landmarks",
                     width=800, height=600
                    )

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        visible=False
      )
    fig.update_xaxes(
        visible=False
    )
    fig.update_layout({
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
    })
    fig.show()




from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
config={'showLink': False, 'displayModeBar': False}
plot(fig, filename = 'au_to_landmark1.html', config = config)

from IPython.core.display import display, HTML
display(HTML('au_to_landmark1.html'))

import plotly.express as px
import plotly.graph_objects as go

_df = df.query("AUidx>=16")
fig = px.line(_df, x="xs", y="ys", animation_frame="intensity", 
              color="color", color_discrete_map={"k":"black"},
                 hover_name="AU", facet_col="AU", facet_col_wrap=4,
                 range_x=[30,170], range_y=[250, 80],
                 title= "Action Unit activation to landmarks",
                 width=800, height=600
                )

fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    visible=False
  )
fig.update_xaxes(
    visible=False
)
fig.update_layout({
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
})
fig.show()

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
config={'showLink': False, 'displayModeBar': False}
plot(fig, filename = 'au_to_landmark3.html', config = config)

visualize_autolandmark(df.query("AUidx<8"))

visualize_autolandmark(df.query("AUidx>=8 and AUidx<16"))

visualize_autolandmark(df.query("AUidx>=16"))





# draw lineface

aus = []
xs = []
ys = []
AUname = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 28, 43]
df = pd.DataFrame()
for intensity in np.arange(0, 5.1 ,.5):
    for au in range(20):
        aus = np.zeros(20)
        aus[au] = intensity
        xs, ys = predict(aus)   
        _df = pd.DataFrame({"xs": xs, 
                          "ys": ys, 
                          "coord_id": range(68),
                          "intensity": intensity, 
                          "AU": AUname[au],
                          "color": "k"})
        
        idxs = [17, 23, 29, 39, 46, 53]
        for idx in idxs:
            df1 = _df.iloc[:idx]
            df1.loc[-1] = [np.nan, np.nan, np.nan, intensity, 1.0, "k"]
            df2 = _df.iloc[idx:]
            df2.index = df2.index+1
            _df = pd.concat([df1.reset_index(drop=True), df2])
            
        df = pd.concat([df, _df])

import plotly.express as px
import plotly.graph_objects as go

_df = df.query("AU == '1'")

fig = px.line(_df, x="xs", y="ys", animation_frame="intensity", color="color", color_discrete_map={"k":"black"},
                 hover_name="coord_id", facet_col="AU", facet_col_wrap=4,
                 range_x=[0,200], range_y=[250, 75],
                 width=400, height=600
                )

fig.update_traces(connectgaps=False)

fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.show()





"""
x: xcoords
y: ycoords
animation_frame: intensity
animation_group: AU
facet_col="AU"
"""

import plotly.express as px
import plotly.graph_objects as go

_df = df.query("AU == '1' and intensity==5")

idxs = [17, 24, 29, 39, 46, 53]
for idx in idxs:
    df1 = _df.iloc[:idx]
    df1.loc[-1] = [np.nan, np.nan, np.nan, 5.0, 1.0, "k"]
    df2 = _df.iloc[idx:]
    df2.index = df2.index+1
    _df = pd.concat([df1.reset_index(drop=True), df2])

fig = px.line(_df, x="xs", y="ys", animation_frame="intensity", color="color", color_discrete_map={"k":"black"},
                 hover_name="coord_id", facet_col="AU", facet_col_wrap=4,
                 range_x=[0,200], range_y=[250, 75],
                 width=400, height=600
                )

fig.update_traces(connectgaps=False)

fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.show()

df1 = _df.iloc[:17]
df1.loc[-1] = [np.nan, np.nan, np.nan, 5.0, 1.0, "k"]

df2 = _df.iloc[17:]
df2.index = df2.index+1

_df = pd.concat([df1.reset_index(drop=True), df2])













## Add a vectorfield with arrows from the changed face back to neutral and vice versa 

from feat.plotting import plot_face, predict
from feat.utils import load_h5
import numpy as np
import matplotlib.pyplot as plt

model = load_h5('pyfeat_aus_to_landmarks.h5')
# Add data activate AU1, and AU12
au = np.array([2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])

# Get neutral landmarks
neutral = predict(np.zeros(len(au)))

# Provide target landmarks and other vector specifications
vectors = {'target': predict(au),
           'reference':  neutral, 'color': 'blue'}

fig, axes = plt.subplots(1,2)
# Plot face where vectorfield goes from neutral to target, with target as final face
plot_face(model = model, ax = axes[0], au = np.array(au), 
            vectorfield = vectors, color='k', linewidth=1, linestyle='-')

# Plot face where vectorfield goes from neutral to target, with neutral as base face
plot_face(model = model, ax = axes[1], au = np.zeros(len(au)), 
            vectorfield = vectors, color='k', linewidth=1, linestyle='-')

## Add muscle heatmaps to the plot

from feat.plotting import plot_face
from feat.utils import load_h5
import numpy as np
import matplotlib.pyplot as plt

# Add data
model = load_h5()

au = np.array([2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Add some muscles
muscles = {'orb_oris_l': 'yellow', 'orb_oris_u': "blue"}
muscles = {'all': 'heatmap'}

plot_face(model=model, au = np.array(au), 
          muscles = muscles, color='k', linewidth=1, linestyle='-')

from feat.plotting import plot_face
from feat.utils import load_h5
import numpy as np
import matplotlib.pyplot as plt

# Add data
au = [0.127416, 0.809139, 0, 0.343189, 0.689964, 1.23862, 1.28464, 0.79003, 0.842145, 0.111669, 
      0.450328, 1.02961, 0.871225, 0, 1.1977,  0.457218, 0, 0, 0, 0]
au = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Add some muscles
muscles = {'all': 'heatmap'}

# Plot face
plot_face(model=None, au = np.array(au), muscles = muscles, color='k', linewidth=1, linestyle='-')

## Make sure muscle array contains 'facet' for a facet heatmap

from feat.plotting import plot_face
from feat.utils import load_h5
import numpy as np
import matplotlib.pyplot as plt

# Add data
au = np.array([0.127416, 0.809139, 0, 0.343189, 0.689964, 1.23862, 1.28464, 0.79003, 0.842145, 0.111669, 
      0.450328, 1.02961, 0.871225, 0, 1.1977,  0.457218, 0, 0, 0, 0])

# Load a model 
model = load_h5()

# Add muscles
muscles = {'all': 'heatmap', 'facet': 1}

# Plot face
plot_face(model=model, au = au, muscles = muscles, color='k', linewidth=1, linestyle='-')

## Add gaze vectors
Add gaze vectors to indicate where the eyes are looking.   
Gaze vectors are length 4 (lefteye_x, lefteye_y, righteye_x, righteye_y) where the y orientation is positive for looking upwards.

from feat.plotting import plot_face
from feat.utils import load_h5
import numpy as np
import matplotlib.pyplot as plt

# Add data
au = np.zeros(20)

# Add some gaze vectors: (lefteye_x, lefteye_y, righteye_x, righteye_y)
gaze = [-1, 5, 1, 5]

# Plot face
plot_face(model=None, au = au, gaze = gaze, color='k', linewidth=1, linestyle='-')

## Call plot method on Fex instances
It is possible to call the `plot_aus` method within openface, facet, affdex fex instances

OpenFace

from feat.plotting import plot_face
import numpy as np
import matplotlib.pyplot as plt
from feat.utils import  load_h5, get_resource_path, read_openface
from feat.tests.utils import get_test_data_path
from os.path import join

test_file = join(get_test_data_path(),'OpenFace_Test.csv')
openface = read_openface(test_file)
openface.plot_aus(12, muscles={'all': "heatmap"}, gaze = None)