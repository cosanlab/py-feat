# Included pre-trained detectors

Below is a list of detectors included in Py-Feat and ready to use. The model names are in the titles followed by the reference publications. Bolded models are defaults.

You can specify any of these models for use in the `Detector` class by passing in the name as a string, e.g.

```
from feat import Detector

detector = Detector(emotion_model='svm')
```

```{note}
Models names are case-insensitive: `'resmasknet' == 'ResMaskNet'`
```

## Face and Facial Pose detection

- **`img2pose`: Face Alignment and Detection via 6DoF, Face Pose Estimation** ([Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2.pdf)). Performs simultaneous (one-shot) face detection and head pose estimation

## Facial landmark detection

- **`mobilefacenet`: Efficient CNNs for accurate real time face verification on mobile devices** ([Chen et al, 2018](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf))
- `mobilenet`: Efficient convolutional neural networks for mobile vision applications ([Howard et al, 2017](https://arxiv.org/pdf/1704.04861v1.pdf))
- `pfld`: Practical Facial Landmark Detector by ([Guo et al, 2019](https://arxiv.org/pdf/1902.10859.pdf))

## Action Unit detection

- **`xgb`: XGBoost Classifier model trained on Histogram of Oriented Gradients\*** extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets
- `svm`: SVM model trained on Histogram of Oriented Gradients\*\* extracted from BP4D, DISFA, CK+, UNBC-McMaster shoulder pain, and AFF-Wild2 datasets

```{note}
\*For AU07, our `xbg` detector was trained with hinge-loss instead of cross-entropy loss like other AUs as this yielded substantially better detection peformance given the labeled data available for this AU. This means that while it returns continuous probability predictions,  these are more likely to appear binary in practice (i.e. be 0 or 1) and should be interpreted as *proportion of decision-trees with a detection* rather than *average decision-tree confidence* like other AU values.
```

```{note}
\*\* Our `svm` detector uses the [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) implementation from `sklearn` and thus returns **binary values** for each AU rather than probabilities. If your use-case requires continuous-valued detections, we recommend the `xgb` detector instead.
```

## Emotion detection

- **`resmasknet`: Facial expression recognition using residual masking network** by ([Pham et al., 2020](https://ieeexplore.ieee.org/document/9411919))
- `svm`: SVM model trained on Histogram of Oriented Gradients extracted from ExpW, CK+, and JAFFE datasets

## Identity detection

- **`facenet`: FaceNet: A unified embedding for face recognition and clustering ([Schroff et al, 2015](https://arxiv.org/abs/1503.03832))**. Inception Resnet (V1) pretrained on VGGFace2 and CASIA-Webface.
