"""Class definitions."""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import six
import abc
from copy import deepcopy
from nltools.data import Adjacency, design_matrix
from nltools.stats import (downsample,
                           upsample,
                           transform_pairwise)
from nltools.utils import (set_decomposition_algorithm)
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.utils import check_random_state
from feat.utils import read_facet, read_openface, wavelet, calc_hist_auc
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
        filepath: path to file
        sampling_freq (float, optional): sampling rate of each row in Hz;
                                         defaults to None
        features (pd.Dataframe, optional): features that correspond to each
                                          Fex row
    """
    # __metaclass__  = abc.ABCMeta
    _metadata = ['filename','sampling_freq', 'features']

    def __init__(self, *args, **kwargs):
        self.filename = kwargs.pop('filename', None)
        self.sampling_freq = kwargs.pop('sampling_freq', None)
        self.features = kwargs.pop('features', False)
        super(Fex, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return self.__class__

    @property
    def _constructor_sliced(self):
        return FexSeries

    @abc.abstractmethod
    def read_file(self, *args, **kwargs):
        """ Loads file into FEX class """
        pass

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
            if self.features:
                if out.features.shape[1]==data.features[1]:
                    out.features = out.features.append(data.features, ignore_index=True)
                else:
                    raise ValueError('Different number of features in new dataset.')
        return out

    def regress(self):
        NotImplemented

    def ttest(self, threshold_dict=None):
        NotImplemented

    def predict(self, *args, **kwargs):
        NotImplemented

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
        return self.__class__(df_ds, sampling_freq=target, features=ds_features)

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
        return self.__class__(df_us, sampling_freq=target, features=us_features)

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
            return self.__class__(out-out.median(), sampling_freq=out.sampling_freq)
        elif baseline is 'mean':
            return self.__class__(out-out.mean(), sampling_freq=out.sampling_freq)
        elif isinstance(baseline, (Series, FexSeries)):
            return self.__class__(out-baseline, sampling_freq=out.sampling_freq)
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
        return self.__class__(pd.DataFrame(clean(self.values, detrend=detrend,
                                      standardize=standardize,
                                      confounds=confounds,
                                      low_pass=low_pass,
                                      high_pass=high_pass,
                                      ensure_finite=ensure_finite,
                                      t_r=1/self.sampling_freq,
                                      *args, **kwargs),
                                columns=self.columns),
                    sampling_freq=self.sampling_freq)

    def decompose(self, algorithm='pca', axis=1, n_components=None,
                  *args, **kwargs):
        ''' Decompose Fex instance

        Args:
            algorithm: (str) Algorithm to perform decomposition
                        types=['pca','ica','nnmf','fa']
            axis: dimension to decompose [0,1]
            n_components: (int) number of components. If None then retain
                        as many as possible.

        Returns:
            output: a dictionary of decomposition parameters

        '''

        out = {}
        out['decomposition_object'] = set_decomposition_algorithm(
                                                    algorithm=algorithm,
                                                    n_components=n_components,
                                                    *args, **kwargs)
        com_names = ['c%s' % str(x+1) for x in range(n_components)]
        if axis == 0:
            out['decomposition_object'].fit(self.T)
            out['components'] = self.__class__(pd.DataFrame(out['decomposition_object'].transform(self.T), index=self.columns, columns=com_names), sampling_freq=None)
            out['weights'] = self.__class__(pd.DataFrame(out['decomposition_object'].components_.T,index=self.index,columns=com_names), sampling_freq=self.sampling_freq)
        if axis == 1:
            out['decomposition_object'].fit(self)
            out['components'] = self.__class__(pd.DataFrame(out['decomposition_object'].transform(self), columns=com_names), sampling_freq=self.sampling_freq)
            out['weights'] = self.__class__(pd.DataFrame(out['decomposition_object'].components_, index=com_names, columns=self.columns), sampling_freq=None).T
        return out

    def extract_boft(self, min_freq=.06, max_freq=.66, bank=8, *args, **kwargs):
        """ Extract Bag of Temporal features
        Args: 
            min_freq: maximum frequency of temporal filters
            max_freq: minimum frequency of temporal filters
            bank: number of temporal filter banks, filters are on exponential scale

        Returns:
            wavs: list of Morlet wavelets with corresponding freq
            hzs:  list of hzs for each Morlet wavelet   


        """
        # First generate the wavelets
        target_hz = self.sampling_freq
        freqs = np.geomspace(min_freq,max_freq,bank)
        wavs, hzs = [],[]
        for i, f in enumerate(freqs):
            wav = wavelet(f,sampling_rate=target_hz)
            wavs.append(wav)
            hzs.append(str(np.round(freqs[i],2)))
        wavs = np.array(wavs)[::-1,:]
        hzs = np.array(hzs)[::-1]
        # # check asymptotes at lowest freq
        # asym = wavs[-1,:10].sum()
        # if asym > .001:
        #     print("Lowest frequency asymptotes at %2.8f " %(wavs[-1,:10].sum()))

        # Convolve data with wavelets
        Feats2Use = self.columns
        feats = pd.DataFrame()
        for feat in Feats2Use:
            _d = self[[feat]].T
            assert _d.isnull().sum().any()==0, "Data contains NaNs. Cannot convolve. "
            for iw, cm in enumerate(wavs):
                convolved = np.apply_along_axis(lambda m: np.convolve(m, cm,mode='full'),axis=1,arr=_d.as_matrix())
                # Extract bin features.
                out = pd.DataFrame(convolved.T).apply(calc_hist_auc,args=(None))
                colnames = ['pos'+str(i)+'_hz_'+hzs[iw]+'_'+feat for i in range(6)]
                colnames.extend(['neg'+str(i)+'_hz_'+hzs[iw]+'_'+feat for i in range(6)])
                out = out.T
                out.columns = colnames
                feats = pd.concat([feats, out], axis=1)
        return self.__class__(feats)

class Facet(Fex):
    """
    Facet is a subclass of Fex.
    You can use the Facet subclass to load iMotions-FACET data files.
    It will also have Facet specific methods.
    """
    def read_file(self, *args, **kwargs):
        super(Fex, self).__init__(read_facet(self.filename, *args, **kwargs), *args, **kwargs)

class Affdex(Fex):
    def read_file(self, *args, **kwargs):
        # super(Fex, self).__init__(read_affdex(self.filename, *args, **kwargs), *args, **kwargs)
        pass

class Openface(Fex):
    """
    Openface is a subclass of Fex.
    You can use the Openface subclass to load Openface data files.
    It will also have Openface specific methods.
    """
    def read_file(self, *args, **kwargs):
        super(Fex, self).__init__(read_openface(self.filename, *args, **kwargs), *args, **kwargs)
