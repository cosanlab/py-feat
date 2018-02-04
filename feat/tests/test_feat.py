#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pytest
import pandas as pd
import numpy as np
from os.path import join, exists
from .utils import get_test_data_path
from feat.data import Fex, _check_if_fex, Facet
from feat.utils import read_facet
from nltools.data import Adjacency

def test_fex(tmpdir):
    imotions_columns = ['Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                  'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                  'Neutral Evidence','Positive Evidence','Negative Evidence','AU1 Evidence','AU2 Evidence',
                  'AU4 Evidence','AU5 Evidence','AU6 Evidence','AU7 Evidence','AU9 Evidence','AU10 Evidence',
                  'AU12 Evidence','AU14 Evidence','AU15 Evidence','AU17 Evidence','AU18 Evidence','AU20 Evidence',
                  'AU23 Evidence','AU24 Evidence','AU25 Evidence','AU26 Evidence','AU28 Evidence','AU43 Evidence',
                  'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']

    filename = join(get_test_data_path(), 'iMotions_Test.txt')
    dat = Fex(read_facet(filename), sampling_freq=30)

    # Test length
    assert len(dat)==519

    # Test Downsample
    assert len(dat.downsample(target=10))==52

    # Test upsample
    assert len(dat.upsample(target=60,target_type='hz'))==(len(dat)-1)*2

    # Test interpolation
    assert np.sum(dat.interpolate(method='linear').isnull().sum()==0) == len(dat.columns)

    # Test distance
    d = dat.interpolate(method='linear').distance()
    assert isinstance(d, Adjacency)
    assert d.square_shape()[0]==len(dat)

    # Test Copy
    assert isinstance(dat.copy(), Fex)
    assert dat.copy().sampling_freq==dat.sampling_freq

    # Test baseline
    assert isinstance(dat.baseline(baseline='median'), Fex)
    assert isinstance(dat.baseline(baseline='mean'), Fex)
    assert isinstance(dat.baseline(baseline=dat.mean()), Fex)

    # Test facet subclass
    facet = Facet(filename=filename,sampling_freq=30)
    facet.read_file()
    assert len(facet)==519

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
