#!/usr/bin/env python
# coding: utf-8

# # Loading data from other detectors
# *written by Jin Hyun Cheong*
# 
# While Py-FEAT provides it's own set of detectors, you can still use Py-FEAT if you extracted features from other models. Currently we support data files extracted from OpenFace, FACET iMotions, and Affectiva JavaScript SDK. Please open an Issue if you would like to see support for other model outputs. 
# 
# ## Loading OpenFace data

# In[1]:


import glob, os
from feat.tests.utils import get_test_data_path
from feat.utils import read_openface

openface_file = os.path.join(get_test_data_path(), "OpenFace_Test.csv")
detections = read_openface(openface_file)
print(type(detections))
display(detections.head())


# All functionalities of the `Fex` class will be available when you load an OpenFace file using `read_openface`. For example, you can quickly grab the facial landmark columns using `landmarks()` or the aus using `aus()`

# In[2]:


detections.landmark().head()


# In[3]:


detections.aus().head()


# ## Loading FACET iMotions data
# Loading a FACET file as a Fex class is also simple using `read_facet`. 

# In[4]:


from feat.utils import read_facet

facet = os.path.join(get_test_data_path(), "iMotions_Test_v6.txt")
detections = read_facet(facet)
print(type(detections))
display(detections.head())


# You can take advantage of the Fex functionalties such as grabbing the emotions

# In[5]:


detections.emotions().head()


# You can also extract features from the data. For example, to extract bags of temporal features from this video, you would simply set the sampling frequency and run `extract_boft`. 

# In[6]:


detections.sampling_freq = 30
detections.emotions().dropna().extract_boft()


# ## Loading Affectiva API file
# You can also load an affectiva file processed through the [Affectiva Javascript SDK](https://blog.affectiva.com/javascript-emotion-sdk-for-youtube-facebook-and-twitch-api) available [here](http://jinhyuncheong.com/affectiva-app/affectiva_emotion_detector_photo.html)

# In[7]:


from feat.utils import read_affectiva

facet = os.path.join(get_test_data_path(), "sample_affectiva-api-app_output.json")
detections = read_affectiva(facet)
print(type(detections))
display(detections.head())


# ## Loading a completely new file as a Fex class
# It's easy to cast a dataframe which might be neither an OpenFace or FACET outputs into a Fex class. Simply cast your dataframe as a Fex class.

# In[24]:


from feat import Fex
import pandas as pd, numpy as np
au_columns = [f"AU{i}" for i in range(20)]
fex = Fex(pd.DataFrame(np.random.rand(20,20)))
fex.columns = au_columns
print(type(fex))


# To take full advantage of Py-Feat's features, make sure you set the attributes. 

# In[26]:


fex.au_columns = au_columns
display(fex.aus().head())


# In[ ]:




