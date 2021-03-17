# Installation example

*Written by Jin Hyun Cheong*

Open the current notebook in [Google Colab](http://colab.research.google.com/) and run the cell below to install Py-Feat. Make sure to `Restart Runtime` so that the installation is recognized. 

# Install Py-Feat from source.
!git clone https://github.com/cosanlab/feat.git  
!cd feat && pip install -q -r requirements.txt
!cd feat && pip install -q -e . 
!cd feat && python bin/download_models.py
# Click Runtime from top menu and Restart Runtime! 

```{warning}
Make sure you `Restart Runtime` from the `Runtime` top menu bar to refresh your session. 
In Kaggle, click `Restart & clear outputs` from the `Run` top menu bar. NOT `Restart session`
```

# Check Fex class installation
from feat import Fex
fex = Fex()

# Check Detector class installation
from feat import Detector
detector = Detector()