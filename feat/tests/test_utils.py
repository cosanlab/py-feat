#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pytest
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from os.path import join, exists
from .utils import get_test_data_path
from feat.utils import read_facet, read_openface, read_affectiva, registration, neutral, softmax, load_h5, load_pickled_model
from nltools.data import Adjacency
import unittest

def test_utils():
    sample = read_openface(join(get_test_data_path(), 'OpenFace_Test.csv'))
    lm_cols = ['x_'+str(i) for i in range(0,68)]+['y_'+str(i) for i in range(0,68)]
    sample_face = np.array([sample[lm_cols].values[0]])
    registered_lm = registration(sample_face)
    assert(registered_lm.shape==(1,136))

    with pytest.raises(ValueError):
        registration(sample_face, method='badmethod')
    with pytest.raises(TypeError):
        registration(sample_face, method = np.array([1,2,3,4]))
    with pytest.raises(AssertionError):
        registration([sample_face[0]])
    with pytest.raises(AssertionError):
        registration(sample_face[0])
    with pytest.raises(AssertionError):
        registration(sample_face[:,:-1])

    # Test softmax
    assert(softmax(0) == .5)
    # Test badfile. 
    with pytest.raises(Exception):
        load_h5("badfile.h5")

    # Test loading of pickled model 
    out = load_pickled_model()
    with pytest.raises(Exception):
        load_pickled_model("badfile.pkl")