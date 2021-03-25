# Loading data from other detectors
*written by Jin Hyun Cheong*

While Py-FEAT provides it's own set of detectors, you can still use Py-FEAT if you extracted features from other models. Currently we support data files extracted from OpenFace, FACET iMotions, and Affectiva JavaScript SDK. Please open an Issue if you would like to see support for other model outputs. 

## Loading OpenFace data

import glob, os
from feat.tests.utils import get_test_data_path
from feat.utils import read_openface

openface_file = os.path.join(get_test_data_path(), "OpenFace_Test.csv")
detections = read_openface(openface_file)
print(type(detections))
display(detections.head())

All functionalities of the `Fex` class will be available when you load an OpenFace file using `read_openface`. For example, you can quickly grab the facial landmark columns using `landmarks()` or the aus using `aus()`

detections.landmark().head()

detections.aus().head()

## Loading FACET iMotions data
Loading a FACET file as a Fex class is also simple using `read_facet`. 

from feat.utils import read_facet

facet = os.path.join(get_test_data_path(), "iMotions_Test_v6.txt")
detections = read_facet(facet)
print(type(detections))
display(detections.head())

You can take advantage of the Fex functionalties such as grabbing the emotions

detections.emotions().head()

You can also extract features from the data. For example, to extract bags of temporal features from this video, you would simply set the sampling frequency and run `extract_boft`. 

detections.sampling_freq = 30
detections.emotions().dropna().extract_boft()

## Loading Affectiva API file
You can also load an affectiva file processed through the [Affectiva Javascript SDK](https://blog.affectiva.com/javascript-emotion-sdk-for-youtube-facebook-and-twitch-api) available [here](http://jinhyuncheong.com/affectiva-app/affectiva_emotion_detector_photo.html)

from feat.utils import read_affectiva

facet = os.path.join(get_test_data_path(), "sample_affectiva-api-app_output.json")
detections = read_affectiva(facet)
print(type(detections))
display(detections.head())