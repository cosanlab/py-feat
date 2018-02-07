#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pytest
import pandas as pd
import numpy as np
from os.path import join, exists
from .utils import get_test_data_path
from feat.data import Fex, Facet, Openface
from feat.utils import read_facet, read_openface
from nltools.data import Adjacency
import unittest

def test_fex(tmpdir):
    # For iMotions-FACET data file
    filename = join(get_test_data_path(), 'iMotions_Test.txt')
    dat = Fex(read_facet(filename), sampling_freq=30)

    # Test KeyError
    class MyTestCase(unittest.TestCase):
        def test1(self):
            with self.assertRaises(KeyError):
                Fex(read_facet(filename, features=['NotHere']), sampling_freq=30)

    # Test length
    assert len(dat)==519

    # Test Downsample
    assert len(dat.downsample(target=10))==52

    # Test upsample
    assert len(dat.upsample(target=60,target_type='hz'))==(len(dat)-1)*2

    # Test interpolation
    assert np.sum(dat.interpolate(method='linear').isnull().sum()==0) == len(dat.columns)
    dat = dat.interpolate(method='linear')

    # Test distance
    d = dat.distance()
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

    # Test Bag Of Temporal Features Extraction
    facet_filled = facet.fillna(0)
    assert isinstance(facet_filled,Facet)
    assert isinstance(facet_filled.extract_boft(), Facet)
    filters, histograms = 8, 12
    assert facet_filled.extract_boft().shape[1]==facet.columns.shape[0] * filters * histograms

    # Test mean, min, and max Features Extraction
    assert isinstance(facet_filled.extract_mean(), Facet)
    assert isinstance(facet_filled.extract_min(), Facet)
    assert isinstance(facet_filled.extract_max(), Facet)

    # Test if a method returns subclass.
    facet = facet.downsample(target=10,target_type='hz')
    assert isinstance(facet,Facet)

    ### Test Openface importer and subclass ###

    # For OpenFace data file
    filename = join(get_test_data_path(), 'OpenFace_Test.csv')
    openface = Fex(read_openface(filename), sampling_freq=30)

    # Test KeyError
    class MyTestCase(unittest.TestCase):
        def test1(self):
            with self.assertRaises(KeyError):
                Fex(read_openface(filename, features=['NotHere']), sampling_freq=30)


    # Test length
    assert len(openface)==100

    # Test loading from filename
    openface = Openface(filename=filename, sampling_freq = 30)
    openface.read_file()

    assert len(openface)==100

    # Test if a method returns subclass.
    openface = openface.downsample(target=10,target_type='hz')
    assert isinstance(openface,Openface)

    # Check if file is missing columns
    data_bad = dat.iloc[:,0:10]
    with pytest.raises(Exception):
        _check_if_fex(data_bad, imotions_columns)

    # Check if file has too many columns
    data_bad = dat.copy()
    data_bad['Test'] = 0
    with pytest.raises(Exception):
        _check_if_fex(data_bad, imotions_columns)

    # Test clean
    assert isinstance(dat.clean(), Fex)
    assert dat.clean().columns is dat.columns
    assert dat.clean().sampling_freq == dat.sampling_freq

    # Test Decompose
    n_components = 3
    stats = dat.decompose(algorithm='pca', axis=1,
                          n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    stats = dat.decompose(algorithm='ica', axis=1,
                          n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    new_dat = dat+100
    stats = new_dat.decompose(algorithm='nnmf', axis=1,
                              n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    stats = dat.decompose(algorithm='fa', axis=1,
                          n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    stats = dat.decompose(algorithm='pca', axis=0,
                          n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    stats = dat.decompose(algorithm='ica', axis=0,
                          n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    new_dat = dat+100
    stats = new_dat.decompose(algorithm='nnmf', axis=0,
                              n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]

    stats = dat.decompose(algorithm='fa', axis=0,
                          n_components=n_components)
    assert n_components == stats['components'].shape[1]
    assert n_components == stats['weights'].shape[1]
