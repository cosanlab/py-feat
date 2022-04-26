# Included pre-trained detectors

Below is a list of detectors included in Py-Feat and ready to use. The model names are in the titles followed by the reference publications.

You can specify any of these models for use in the `Detector` class by passing in the name as a string, e.g.

```
from feat import Detector

detector = Detector(au_model='rf', emotion_model='svm')
```

```{note}
Models names are case-insensitive: `'resmasknet' == 'ResMaskNet'`
```

## Action Unit detection
- `rf`: Random Forest model trained on Histogram of Oriented Gradients extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `logistic`: Logistic Classifier model trained on Histogram of Oriented Gradients extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `JAANET`: Joint facial action unit detection and face alignment via adaptive attention trained with BP4D and BP4D+ ([Shao et al., 2020](https://arxiv.org/pdf/2003.08834v1.pdf))
- `DRML`: Deep region and multi-label learning for facial action unit detection by ([Zhao et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf))

```{caution}
If you plan to use a Histogram-based Action Unit model (e.g. `rf`, `svm`, `logistic`) you should *also* use a Histogram-based Emotion detection model (e.g. `rf`, `svm`). It's not advised to mix Histogram-based models for Action Unit detection with deep-learning models for emotion detection and visa-versa.
```

##  Emotion detection
- `rf`: Random Forest model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets
- `fer`: Deep convolutional network
- `ResMaskNet`: Facial expression recognition using residual masking network by ([Pham et al., 2020](https://ailb-web.ing.unimore.it/icpr/author/3818))

##  Face detection
- `MTCNN`: Multi-task cascaded convolutional networks by ([Zhang et al., 2016](https://arxiv.org/pdf/1604.02878.pdf); [Zhang et al., 2020](https://ieeexplore.ieee.org/document/9239720))
- `FaceBoxes`: A CPU real-time fae detector with high accuracy by ([Zhang et al., 2018](https://arxiv.org/pdf/1708.05234v4.pdf))
- `RetinaFace`: Single-stage dense face localisation in the wild by ([Deng et al., 2019](https://arxiv.org/pdf/1905.00641v2.pdf))
- `img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation ([Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2.pdf)). Performs simultaneous (one-shot) face detection and head pose estimation
- `img2pose-c`: A 'constrained' version of the above model, fine-tuned on images of frontal faces with pitch, roll, yaw measures in the range of (-90, 90) degrees. Shows lesser performance on difficult face detection tasks, but state-of-the-art performance on face pose estimation for frontal faces

##  Facial landmark detection
- `PFLD`: Practical Facial Landmark Detector by ([Guo et al, 2019](https://arxiv.org/pdf/1902.10859.pdf))
- `MobileFaceNet`: Efficient CNNs for accurate real time face verification on mobile devices ([Chen et al, 2018](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf))
- `MobileNet`: Efficient convolutional neural networks for mobile vision applications ([Howard et al, 2017](https://arxiv.org/pdf/1704.04861v1.pdf))

## Face/Head pose estimation
- `img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation ([Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2.pdf)). Performs simultaneous (one-shot) face detection and head pose estimation
- `img2pose-c`: A 'constrained' version of the above model, fine-tuned on images of frontal faces with pitch, roll, yaw measures in the range of (-90, 90) degrees. Shows lesser performance on hard face detection tasks, but state-of-the-art performance on head pose estimation for frontal faces.
- `Perspective-n-Point`: [Efficient PnP (EPnP)](https://link.springer.com/article/10.1007/s11263-008-0152-6) method implemented via `cv2` to solve the [Perspective n Point](https://en.wikipedia.org/wiki/Perspective-n-Point) (PnP) problem to obtain 3D head pose from 2D facial landmarks
