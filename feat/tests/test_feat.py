#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pytest
import pandas as pd
import numpy as np
from os.path import join, exists
from .utils import get_test_data_path
from feat.data import Fex, _check_if_fex
from feat.utils import read_facet
from nltools.data import Adjacency

def test_fex(tmpdir):
    # For iMotions-FACET data file
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

    # For iMotions-FACET data file
    filename = join(get_test_data_path(), 'OpenFace_Test.csv')
    dat = Fex(read_openface(filename), sampling_freq=30)

    # Test length
    assert len(dat)==100

    # Test Downsample
    assert len(dat.downsample(target=10))==10

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
