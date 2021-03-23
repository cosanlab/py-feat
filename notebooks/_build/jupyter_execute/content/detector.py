# Detector

## How to use the Feat Detector class.

*Written by Jin Hyun Cheong*

Here is an example of how to use the `Detector` class to detect faces, facial landmarks, Action Units, and emotions, from face images or videos. 

## Detecting facial expressions from images. 

First, load the detector class. You can specify which models you want to use.

from feat import Detector
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "jaanet"
emotion_model = "fer"
detector = Detector()

Find the file you want to process. In our case, we'll use our test image `input.jpg`. 

# Find the file you want to process.
from feat.tests.utils import get_test_data_path
import os
test_data_dir = get_test_data_path()
test_image = os.path.join(test_data_dir, "input.jpg")

Here is what our test image looks like.

from PIL import Image
import matplotlib.pyplot as plt
f, ax = plt.subplots()
im = Image.open(test_image)
ax.imshow(im);

Now we use our initialized `detector` instance to make predictions with the `detect_image()` method.

image_prediction = detector.detect_image(test_image)
# Show results
image_prediction

The output is a `Fex` class instance which allows you to run the built-in methods for `Fex`. 

## Visualizing detection results.

For example, you can easily plot the detection results. 

image_prediction.plot_detections();

## Accessing face expression columns of interest.  

You can also access the columns of interests (AUs, emotion) quickly. 

image_prediction.facebox()

image_prediction.aus()

image_prediction.emotions()

## Detecting facial expressions and saving to a file. 

You can also output the results into file by specifying the `outputFname`. The detector will return `True` when it's finished. 

detector.detect_image(test_image, outputFname = "output.csv")

## Loading detection results from saved file. 

The outputs can be loaded using our `read_feat()` function or a simple Pandas `read_csv()`. We recommend using `read_feat()` because that will allow you to use the full suite of Feat functionalities more easily.

from feat.utils import read_feat
image_prediction = read_feat("output.csv")
# Show results
image_prediction

import pandas as pd
image_prediction = pd.read_csv("output.csv")
# Show results
image_prediction

## Detecting facial expressions images with many faces. 
Feat's Detector can find multiple faces in a single image. 

test_image = glob.glob(os.path.join(test_data_dir, "tim-mossholder-hOF1bWoet_Q-unsplash.jpg"))
image_prediction = detector.detect_image(test_image)
# Show results
image_prediction

## Visualize multiple faces in a single image.

image_prediction.plot_detections();

## Detecting facial expressions from multiple images
You can also detect facial expressions from a list of images. Just place the path to images in a list and pass it to `detect_images`. 

# Find the file you want to process.
from feat.tests.utils import get_test_data_path
import os, glob
test_data_dir = get_test_data_path()
test_images = glob.glob(os.path.join(test_data_dir, "*.jpg"))
print(test_images)

image_prediction = detector.detect_image(test_images)
image_prediction

When you have multiple images, you can still call the plot_detection which will plot results for all input images. If you have a lot of images, we recommend checking one by one using slicing. 

image_prediction.plot_detections();

You can use the slicing function to plot specific rows in the detection results or for a particular input file.

image_prediction.iloc[[1]].plot_detections();

image_to_plot = image_prediction.input().unique()[1]
image_prediction.query("input == @image_to_plot").plot_detections();

## Detecting facial expressions from videos. 
Detecting facial expressions in videos is also easy by using the `detect_video()` method. This sample video is by [Wolfgang Langer](https://www.pexels.com/@wolfgang-langer-1415383?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/video/a-woman-exhibits-different-emotions-through-facial-expressions-3063838/).

# Find the file you want to process.
from feat.tests.utils import get_test_data_path
import os, glob
test_data_dir = get_test_data_path()
test_video = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

# Show video
from IPython.display import Video
Video(test_video, embed=True)

Let's predict facial expressions from the video using the `detect_video()` method.

video_prediction = detector.detect_video(test_video)
video_prediction.head()

You can also plot the detection results from a video. The frames are not extracted from the video (that will result in thousands of images) so the visualization only shows the detected face without the underlying image.

The video has 24 fps and the actress show sadness around the 0:02, and happiness at 0:14 seconds.

video_prediction.iloc[[2*24]].plot_detections();

video_prediction.iloc[[13*24]].plot_detections();

We can also leverage existing pandas plotting functions to show how emotions unfold over time. We can clearly see how her emotions change from sadness to happiness.

video_prediction.emotions().plot()

In situations you don't need predictions for EVERY frame of the video, you can specify how many frames to skip with `skip_frames`.

video_prediction = detector.detect_video(test_video, skip_frames=20)
video_prediction
