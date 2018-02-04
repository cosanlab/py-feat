#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pytest
import pandas as pd
import numpy as np
from os.path import join, exists
from .utils import get_test_data_path
from feat.data import Fex, _check_if_fex
from feat.utils import read_facet, read_openface
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

    # Test clean
    assert isinstance(dat.clean(), Fex)
    assert dat.clean().columns is dat.columns
    assert dat.clean().sampling_freq == dat.sampling_freq

    # Test Decompose
    n_components = 3
    dat = dat.interpolate(method='linear')
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


    # For OpenFace data file
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

    # Test clean
    assert isinstance(dat.clean(), Fex)
    assert dat.clean().columns is dat.columns
    assert dat.clean().sampling_freq == dat.sampling_freq

    # Test Decompose
    n_components = 3
    dat = dat.interpolate(method='linear')
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
