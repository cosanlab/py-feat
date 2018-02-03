#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `feat` package."""

import pandas as pd
import numpy as np
import os
import pytest

def test_fex(tmpdir):
    imotions_columns = ['Joy', 'Anger', 'Surprise', 'Fear', 'Contempt', 'Disgust', 'Sadness',
                   'Confusion', 'Frustration', 'Neutral', 'Positive', 'Negative', 'AU1',
                   'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14',
                   'AU15', 'AU17', 'AU18', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28',
                   'AU43', 'NoOfFaces', 'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']

    file = os.path.join(str(tmpdir),'Tests','iMotions_Test.csv')
    data = pd.read_csv(file)


    # Check if file is missing columns
    data_bad = data.iloc[:,0:10]
    with pytest.raises(Exception):
        _check_if_fex(data_bad, imotions_columns)

    # Check if file has too many columns
    data_bad = data.copy()
    data_bad['Test'] = 0
    with pytest.raises(Exception):
        _check_if_fex(data_bad, imotions_columns)

    # Test loading file
    fex = Fex(file)
    assert isinstance(fex, Fex)

    # Test initializing with pandas
    data = pd.read_csv(file)
    fex = Fex(data)
    assert isinstance(fex, Fex)

    # Test Mean
    assert isinstance(fex.mean(), Fex)

    # Test Std
    assert isinstance(fex.std(), Fex)

    # Test Sum
    assert isinstance(fex.sum(), Fex)

    # Test Copy
    assert isinstance(fex.copy(), Fex)

    # Test length
    assert len(fex)==51
