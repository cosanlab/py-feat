"""Class definitions."""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import six
from copy import deepcopy
from nltools.data import Adjacency, design_matrix
from nltools.stats import (downsample,
                           upsample,
                           transform_pairwise)
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.utils import check_random_state

class FexSeries(Series):

    """
    This is a sub-class of pandas series. While not having additional methods
    of it's own required to retain normal slicing functionality for the
    Fex class, i.e. how slicing is typically handled in pandas.
    All methods should be called on Fex below.
    """

    @property
    def _constructor(self):
        return FexSeries

    @property
    def _constructor_expanddim(self):
        return Fex

class Fex(DataFrame):

    """Fex is a class to represent facial expression data. It is essentially
        an enhanced pandas df, with extra attributes and methods. Methods
        always return a new design matrix instance.

    Args:
        sampling_freq (float, optional): sampling rate of each row in Hz;
                                         defaults to None
        features (pd.Dataframe, optional): features that correspond to each
                                          Fex row
    """

    _metadata = ['sampling_freq', 'features']

    def __init__(self, *args, **kwargs):
        self.sampling_freq = kwargs.pop('sampling_freq', None)
        self.features = kwargs.pop('features', False)
        super(Fex, self).__init__(*args, **kwargs)
        imotions_columns = ['Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                  'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                  'Neutral Evidence','Positive Evidence','Negative Evidence','AU1 Evidence','AU2 Evidence',
                  'AU4 Evidence','AU5 Evidence','AU6 Evidence','AU7 Evidence','AU9 Evidence','AU10 Evidence',
                  'AU12 Evidence','AU14 Evidence','AU15 Evidence','AU17 Evidence','AU18 Evidence','AU20 Evidence',
                  'AU23 Evidence','AU24 Evidence','AU25 Evidence','AU26 Evidence','AU28 Evidence','AU43 Evidence',
                  'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']
        # if not set(imotions_columns).issubset(self):
        #     raise ValueError('Missing key facial expression features.')

    @property
    def _constructor(self):
        return Fex

    @property
    def _constructor_sliced(self):
        return FexSeries

    # @classmethod
    # def from_file(cls, filename, **kwargs):
    #     """Alternate constructor to create a ``GeoDataFrame`` from a file.
    #     Can load a ``GeoDataFrame`` from a file in any format recognized by
    #     `fiona`. See http://toblerity.org/fiona/manual.html for details.
    #     Parameters
    #     ----------
    #     filename : str
    #         File path or file handle to read from. Depending on which kwargs
    #         are included, the content of filename may vary. See
    #         http://toblerity.org/fiona/README.html#usage for usage details.
    #     kwargs : key-word arguments
    #         These arguments are passed to fiona.open, and can be used to
    #         access multi-layer data, data stored within archives (zip files),
    #         etc.
    #     Examples
    #     --------
    #     >>> df = geopandas.GeoDataFrame.from_file('nybb.shp')
    #     """
    #     return geopandas.io.file.read_file(filename, **kwargs)

    def info(self):
        """Print class meta data.

        """
        return '%s.%s(sampling_freq=%s, shape=%s, features_shape=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_freq,
            self.shape,
            self.features.shape,
            )

    def append(self, data):
        '''Append a new Fex object to an existing object'''
        if not isinstance(data, Fex):
            raise ValueError('Make sure data is a Fex instance.')

        if self.empty:
            out = data.copy()
        else:
            out = self.copy()
            if out.sampling_freq!=data.sampling_freq:
                raise ValueError('Make sure Fex objects have the same '
                                 'sampling frequency')
            out.data = out.data.append(data.data, ignore_index=True)
            out.features = out.features.append(data.features, ignore_index=True)
            # Need to check if features match
        return out

    def regress(self):
        pass

    def ttest(self, threshold_dict=None):
        pass

    def predict(self, *args, **kwargs):
        pass

    def downsample(self, target, **kwargs):
        """ Downsample Fex columns. Relies on nltools.stats.downsample,
           but ensures that returned object is a Fex object.

        Args:
            target(float): downsampling target, typically in samples not
                            seconds
            kwargs: additional inputs to nltools.stats.downsample

        """
        df_ds = downsample(self, sampling_freq=self.sampling_freq,
                           target=target, **kwargs)
        if self.features:
            ds_features = downsample(self.features,
                                     sampling_freq=self.sampling_freq,
                                     target=target, **kwargs)
        else:
            ds_features = self.features
        return Fex(df_ds, sampling_freq=target, features=ds_features)

    def upsample(self, target, target_type='hz', **kwargs):
        """ Upsample Fex columns. Relies on nltools.stats.upsample,
            but ensures that returned object is a Fex object.

        Args:
            target(float): upsampling target, default 'hz' (also 'samples',
                           'seconds')
            kwargs: additional inputs to nltools.stats.upsample

        """
        print(self.sampling_freq, type(self.sampling_freq))
        print(target, type(target))
        df_us = upsample(self, sampling_freq=self.sampling_freq,
                         target=target, target_type=target_type, **kwargs)
        if self.features:
            us_features = upsample(self.features,
                                   sampling_freq=self.sampling_freq,
                                   target=target, target_type=target_type,
                                   **kwargs)
        else:
            us_features = self.features
        return Fex(df_us, sampling_freq=target, features=us_features)

    def distance(self, method='euclidean', **kwargs):
        """ Calculate distance between rows within a Fex() instance.

            Args:
                method: type of distance metric (can use any scikit learn or
                        sciypy metric)

            Returns:
                dist: Outputs a 2D distance matrix.

        """

        return Adjacency(pairwise_distances(self, metric=method, **kwargs),
                         matrix_type='Distance')

    def baseline(self, baseline='median'):
        ''' Reference a Fex object to a baseline.

            Args:
                method: {median, mean, Fex object}. Will subtract baseline
                        from Fex object (e.g., mean, median).  If passing a Fex
                        object, it will treat that as the baseline.

            Returns:
                Fex object
        '''

        out = self.copy()
        if baseline is 'median':
            return out-out.median()
        elif baseline is 'mean':
            return out-out.mean()
        elif isinstance(baseline, (Series, FexSeries)):
            return out-baseline
        elif isinstance(baseline, (Fex, DataFrame)):
            raise ValueError('Must pass in a FexSeries not a Fex Instance.')
        else:
            raise ValueError('%s is not implemented please use {mean, median, Fex}' % baseline)



def _check_if_fex(data, column_list):
    '''Check if data is a facial expression dataframe from iMotions

    Notes: can eventually make this an importer of different data types

    Args:
        data: (pd.DataFrame) must have columns from iMotions
        column_list: (list) list of column names that file should contain

    Returns:
        boolean

    '''

    if isinstance(data, (pd.DataFrame, Fex)):
        if len(set(list(data.columns))-set(column_list)) > 0:
            raise ValueError('Data as too many variables (e.g., more than standard iMotions File.')
        if len(set(column_list)-set(list(data.columns))) > 0:
            raise ValueError('Missing several imotions columns')
        return True
    else:
        return False
