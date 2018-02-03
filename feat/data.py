"""Class definitions."""

import os
import numpy as np
import pandas as pd
import six
from copy import deepcopy

class Fex(object):
    def __init__(self, data=None, sampling_freq=None, features=None, *args, **kwargs):
        imotions_columns = ['Joy', 'Anger', 'Surprise', 'Fear', 'Contempt', 'Disgust', 'Sadness',
                           'Confusion', 'Frustration', 'Neutral', 'Positive', 'Negative', 'AU1',
                           'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14',
                           'AU15', 'AU17', 'AU18', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28',
                           'AU43', 'NoOfFaces', 'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']
        if data is None:
            self.data = pd.DataFrame(columns=data_column_names)
        elif isinstance(data, six.string_types):
            data = pd.read_csv(data, index_col=False)
            if _check_if_fex(data, imotions_columns):
                self.data = data
            else:
                raise ValueError('File is not an iMotions csv file.')
        elif _check_if_fex(data, imotions_columns):
            self.data = data

        self.sampling_freq = sampling_freq

        if features is not None:
            if isinstance(features, six.string_types):
                if os.path.isfile(features):
                    features = pd.read_csv(features, header=None, index_col=None)
            if isinstance(features, pd.DataFrame):
                if self.data.shape[0] != len(features):
                    raise ValueError("features does not match the correct size of "
                                     "data")
                self.features = features
            else:
                raise ValueError("Make sure features is a pandas data frame.")
        else:
            self.features = pd.DataFrame()

    def __len__(self):
        return self.data.shape[0]

    def copy(self):
        '''Return a deepcopy of the fex object'''
        return deepcopy(self)

    def mean(self):
        '''Calculate mean across action units and features'''
        out = self.copy()
        out.data = out.data.mean()
        out.features = out.features.mean()
        return out

    def std(self):
        '''Calculate std across action units and features'''
        out = self.copy()
        out.data = out.data.std()
        out.features = out.features.std()
        return out

    def sum(self):
        '''Calculate sum across action units and features'''
        out = self.copy()
        out.data = out.data.sum()
        out.features = out.features.sum()
        return out

def _check_if_fex(data, column_list):
    '''Check if data is a facial expression dataframe from iMotions

    Notes: can eventually make this an importer of different data types

    Args:
        data: (pd.DataFrame) must have columns from iMotions
        column_list: (list) list of column names that file should contain

    Returns:
        boolean

    '''

    if isinstance(data, pd.DataFrame):
        if len(set(data.columns)-set(column_list)) > 0:
            raise ValueError('Data as too many variables (e.g., more than standard iMotions File.')
        if len(set(column_list)-set(data.columns)) > 0:
            raise ValueError('Missing several imotions columns')
        return True
    else:
        return False
