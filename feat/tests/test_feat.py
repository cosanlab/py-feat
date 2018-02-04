#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pytest
import pandas as pd
import numpy as np
from os.path import join, exists
from .utils import get_test_data_path
from feat.data import Fex, _check_if_fex
from nltools.data import Adjacency

def test_fex(tmpdir):
    imotions_columns = ['StudyName', 'ExportDate', 'Name', 'Age', 'Gender', 'StimulusName',
       'SlideType', 'EventSource', 'Timestamp', 'MediaTime', 'PostMarker',
       'Annotation', 'FrameNo', 'FrameTime', 'NoOfFaces', 'FaceRect X',
       'FaceRect Y', 'FaceRect Width', 'FaceRect Height', 'Joy Evidence',
       'Joy Intensity', 'Anger Evidence', 'Anger Intensity',
       'Surprise Evidence', 'Surprise Intensity', 'Fear Evidence',
       'Fear Intensity', 'Contempt Evidence', 'Contempt Intensity',
       'Disgust Evidence', 'Disgust Intensity', 'Sadness Evidence',
       'Sadness Intensity', 'Confusion Evidence', 'Confusion Intensity',
       'Frustration Evidence', 'Frustration Intensity', 'Neutral Evidence',
       'Neutral Intensity', 'Positive Evidence', 'Positive Intensity',
       'Negative Evidence', 'Negative Intensity', 'AU1 Evidence',
       'AU2 Evidence', 'AU4 Evidence', 'AU5 Evidence', 'AU6 Evidence',
       'AU7 Evidence', 'AU9 Evidence', 'AU10 Evidence', 'AU12 Evidence',
       'AU14 Evidence', 'AU15 Evidence', 'AU17 Evidence', 'AU18 Evidence',
       'AU20 Evidence', 'AU23 Evidence', 'AU24 Evidence', 'AU25 Evidence',
       'AU26 Evidence', 'AU28 Evidence', 'AU43 Evidence',
       'HasGlasses Probability', 'IsMale Probability', 'Yaw Degrees',
       'Pitch Degrees', 'Roll Degrees', 'LEFT_EYE_LATERAL X',
       'LEFT_EYE_LATERAL Y', 'LEFT_EYE_PUPIL X', 'LEFT_EYE_PUPIL Y',
       'LEFT_EYE_MEDIAL X', 'LEFT_EYE_MEDIAL Y', 'RIGHT_EYE_MEDIAL X',
       'RIGHT_EYE_MEDIAL Y', 'RIGHT_EYE_PUPIL X', 'RIGHT_EYE_PUPIL Y',
       'RIGHT_EYE_LATERAL X', 'RIGHT_EYE_LATERAL Y', 'NOSE_TIP X',
       'NOSE_TIP Y', '7 X', '7 Y', 'LiveMarker', 'KeyStroke', 'MarkerText',
       'SceneType', 'SceneOutput', 'SceneParent']

    filename = join(get_test_data_path(), 'iMotions_Test.csv')
    dat = Fex(pd.read_csv(filename,skiprows=5, sep='\t'), sampling_freq=30)

    # Test length
    assert len(dat)==519

    # Test Downsample
    assert len(dat.downsample(target=10))==52

    # Test upsample
    assert len(dat.upsample(target=60,target_type='hz'))==(len(dat)-1)*2

    # Test distance
    d = dat.distance()
    assert isinstance(d, Adjacency)
    assert d.square_shape()[0]==len(dat)

    # Test Copy
    assert isinstance(dat.copy(), Fex)
    assert dat.copy().sampling_freq==dat.sampling_freq

    # # Check if file is missing columns
    # data_bad = data.iloc[:,0:10]
    # with pytest.raises(Exception):
    #     _check_if_fex(data_bad, imotions_columns)
    #
    # # Check if file has too many columns
    # data_bad = data.copy()
    # data_bad['Test'] = 0
    # with pytest.raises(Exception):
    #     _check_if_fex(data_bad, imotions_columns)
    #
    # # Test loading file
    # fex = Fex(filename)
    # assert isinstance(fex, Fex)
    #
    # # Test initializing with pandas
    # data = pd.read_csv(filename)
    # fex = Fex(data)
    # assert isinstance(fex, Fex)
    #
