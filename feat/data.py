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
from nilearn.signal import clean

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
        imotions_columns = ['Joy', 'Anger', 'Surprise', 'Fear', 'Contempt',
                            'Disgust', 'Sadness', 'Confusion', 'Frustration',
                            'Neutral', 'Positive', 'Negative', 'AU1', 'AU2',
                            'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12',
                            'AU14', 'AU15', 'AU17', 'AU18', 'AU20', 'AU23',
                            'AU24', 'AU25', 'AU26', 'AU28', 'AU43',
                            'NoOfFaces', 'Yaw Degrees', 'Pitch Degrees',
                            'Roll Degrees']
        # if not set(imotions_columns).issubset(self):
        #     raise ValueError('Missing key facial expression features.')

    @property
    def _constructor(self):
        return Fex

    @property
    def _constructor_sliced(self):
        return FexSeries

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

    def clean(self, detrend=True, standardize=True, confounds=None,
              low_pass=None, high_pass=None, ensure_finite=False,
              *args, **kwargs):

        """ Clean Time Series signal

            This function wraps nilearn functionality and can filter, denoise,
            detrend, etc.

            See http://nilearn.github.io/modules/generated/nilearn.signal.clean.html

            This function can do several things on the input signals, in
            the following order:
                - detrend
                - standardize
                - remove confounds
                - low- and high-pass filter

            Args:
                confounds: (numpy.ndarray, str or list of Confounds timeseries)
                            Shape must be (instant number, confound number),
                            or just (instant number,). The number of time
                            instants in signals and confounds must be identical
                            (i.e. signals.shape[0] == confounds.shape[0]). If a
                            string is provided, it is assumed to be the name of
                            a csv file containing signals as columns, with an
                            optional one-line header. If a list is provided,
                            all confounds are removed from the input signal,
                            as if all were in the same array.

                low_pass: (float) low pass cutoff frequencies in Hz.
                high_pass: (float) high pass cutoff frequencies in Hz.
                detrend: (bool) If detrending should be applied on timeseries
                         (before confound removal)
                standardize: (bool) If True, returned signals are set to unit
                             variance.
                ensure_finite: (bool) If True, the non-finite values
                               (NANs and infs) found in the data will be
                               replaced by zeros.
            Returns:
                cleaned Fex instance

        """
        return Fex(pd.DataFrame(clean(self.values, detrend=detrend,
                                      standardize=standardize,
                                      confounds=confounds,
                                      low_pass=low_pass,
                                      high_pass=high_pass,
                                      ensure_finite=ensure_finite,
                                      t_r=1/self.sampling_freq,
                                      *args, **kwargs),
                                columns=self.columns),
                    sampling_freq=self.sampling_freq)

    # def decompose(self, algorithm='pca', axis='voxels', n_components=None,
    #               *args, **kwargs):
    #     ''' Decompose Brain_Data object
    #
    #     Args:
    #         algorithm: (str) Algorithm to perform decomposition
    #                     types=['pca','ica','nnmf','fa']
    #         axis: dimension to decompose ['voxels','images']
    #         n_components: (int) number of components. If None then retain
    #                     as many as possible.
    #     Returns:
    #         output: a dictionary of decomposition parameters
    #     '''
    #
    #     out = {}
    #     out['decomposition_object'] = set_decomposition_algorithm(
    #                                                 algorithm=algorithm,
    #                                                 n_components=n_components,
    #                                                 *args, **kwargs)
    #     if axis is 'images':
    #         out['decomposition_object'].fit(self.data.T)
    #         out['components'] = self.empty()
    #         out['components'].data = out['decomposition_object'].transform(
    #                                                             self.data.T).T
    #         out['weights'] = out['decomposition_object'].components_.T
    #     if axis is 'voxels':
    #         out['decomposition_object'].fit(self.data)
    #         out['weights'] = out['decomposition_object'].transform(self.data)
    #         out['components'] = self.empty()
    #         out['components'].data = out['decomposition_object'].components_
    #     return out

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
