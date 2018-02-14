from __future__ import division

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
from pandas.core.index import Index
from scipy.signal import convolve

class FexSeries(Series):

    """
    This is a sub-class of pandas series. While not having additional methods
    of it's own required to retain normal slicing functionality for the
    Fex class, i.e. how slicing is typically handled in pandas.
    All methods should be called on Fex below.
    """
    _metadata = ['name', 'sampling_freq', 'sessions']

    def __init__(self, *args, **kwargs):
        self.sampling_freq = kwargs.pop('sampling_freq', None)
        self.sessions = kwargs.pop('sessions', None)
        super(FexSeries, self).__init__(*args, **kwargs)

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
        filepath: (str) path to file
        sampling_freq (float, optional): sampling rate of each row in Hz;
                                         defaults to None
        features (pd.Dataframe, optional): features that correspond to each
                                          Fex row
        sessions: Unique values indicating rows associated with a specific
                  session (e.g., trial, subject, etc). Must be a 1D array of
                  n_samples elements; defaults to None
    """
    # __metaclass__  = abc.ABCMeta
    _metadata = ['filename', 'sampling_freq', 'features', 'sessions']

    def __init__(self, *args, **kwargs):
        self.filename = kwargs.pop('filename', None)
        self.sampling_freq = kwargs.pop('sampling_freq', None)
        self.features = kwargs.pop('features', None)
        self.sessions = kwargs.pop('sessions', None)
        super(Fex, self).__init__(*args, **kwargs)
        if self.sessions is not None:
            if not len(self.sessions) == len(self):
                raise ValueError('Make sure sessions is same length as data.')
            self.sessions = np.array(self.sessions)

        # Set _metadata attributes on series: Kludgy solution
        for k in self:
            self[k].sampling_freq = self.sampling_freq
            self[k].sessions = self.sessions

    @property
    def _constructor(self):
        return self.__class__

    @property
    def _constructor_sliced(self):
        return FexSeries

    def _ixs(self, i, axis=0):
        """ Override indexing to ensure Fex._metadata is propogated correctly
            when integer indexing

        i : int, slice, or sequence of integers
        axis : int
        """
        result = super(self.__class__, self)._ixs(i, axis=axis)

        # Override columns
        if axis == 1:
            """
            Notes
            -----
            If slice passed, the resulting data will be a view
            """

            label = self.columns[i]
            if isinstance(i, slice):
                # need to return view
                lab_slice = slice(label[0], label[-1])
                return self.loc[:, lab_slice]
            else:
                if isinstance(label, Index):
                    return self._take(i, axis=1, convert=True)

                index_len = len(self.index)

                # if the values returned are not the same length
                # as the index (iow a not found value), iget returns
                # a 0-len ndarray. This is effectively catching
                # a numpy error (as numpy should really raise)
                values = self._data.iget(i)

                if index_len and not len(values):
                    values = np.array([np.nan] * index_len, dtype=object)
                result = self._constructor_sliced(
                    values, index=self.index, name=label, fastpath=True,
                    sampling_freq=self.sampling_freq, sessions=self.sessions)

                # this is a cached value, mark it so
                result._set_as_cached(label, self)
        return result

    @abc.abstractmethod
    def read_file(self, *args, **kwargs):
        """ Loads file into FEX class """
        pass

    @abc.abstractmethod
    def calc_pspi(self, *args, **kwargs):
        """ Calculates PSPI (Prkachin and Solomon Pain Intensity) levels which is metric of pain as a linear combination of facial action units(AU).
        The included AUs are brow lowering (AU4), eye tightening (AU6,7), eye closure(AU43,45), nose wrinkling (AU9) and lip raise (AU10).
        Originally PSPI is calculated based on AU intensity scale of 1-5 but for Facet data it is in Evidence units.

        Citation:
        Prkachin and Solomon, (2008) The structure, reliability and validity of pain expression: Evidence from Patients with shoulder pain, Pain, vol 139, non 2 pp 267-274

        Formula:
        PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43 (or AU45 for Openface)

        Return:
            PSPI calculated at each frame.
        """
        pass

    def info(self):
        """Print class meta data.

        """
        if self.features is not None:
            features = self.features.shape
        else:
            features = self.features

        if self.sessions is not None:
            sessions = len(np.unique(self.sessions))
        else:
            sessions = self.sessions

        return '%s.%s(sampling_freq=%s, shape=%s, n_sessions=%s, features_shape=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_freq,
            self.shape,
            sessions,
            features,
            )

    def itersessions(self):
        ''' Iterate over Fex sessions as (session, series) pairs.

        Returns:
            it: a generator that iterates over the sessions of the fex instance

        '''
        for x in np.unique(self.sessions):
            yield x, self.loc[self.sessions==x, :]

    def append(self, data, session_id=None, axis=0):
        ''' Append a new Fex object to an existing object

        Args:
            data: (Fex) Fex instance to append
            session_id: session label
            axis: ([0,1]) Axis to append. Rows=0, Cols=1
        Returns:
            Fex instance
        '''
        if not isinstance(data, self.__class__):
            raise ValueError('Make sure data is a Fex instance.')

        if self.empty:
            out = data.copy()
            if session_id is not None:
                out.sessions = np.repeat(session_id, len(data))
        else:
            if self.sampling_freq != data.sampling_freq:
                raise ValueError('Make sure Fex objects have the same '
                                 'sampling frequency')
            if axis==0:
                out = self.__class__(pd.concat([self, data],
                                               axis=axis,
                                               ignore_index=True),
                                     sampling_freq=self.sampling_freq)
                if session_id is not None:
                    out.sessions = np.hstack([self.sessions, np.repeat(session_id, len(data))])
                if self.features is not None:
                    if data.features is not None:
                        if self.features.shape[1]==data.features.shape[1]:
                            out.features = self.features.append(data.features, ignore_index=True)
                        else:
                            raise ValueError('Different number of features in new dataset.')
                    else:
                        out.features = self.features
                elif data.features is not None:
                    out = data.features
            elif axis==1:
                out = self.__class__(pd.concat([self, data], axis=axis),
                                     sampling_freq=self.sampling_freq)
                if self.sessions is not None:
                    if data.sessions is not None:
                        if np.array_equal(self.sessions, data.sessions):
                            out.sessions = self.sessions
                        else:
                            raise ValueError('Both sessions must be identical.')
                    else:
                        out.sessions = self.sessions
                elif data.sessions is not None:
                    out.sessions = data.sessions
                if self.features is not None:
                    out.features = self.features
                    if data.features is not None:
                        out.features.append(data.features, axis=1, ignore_index=True)
                elif data.features is not None:
                    out.features = data.features
            else:
                raise ValueError('Axis must be 1 or 0.')
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
        if self.features is not None:
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
        if self.features is not None:
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
              ignore_sessions=False, *args, **kwargs):

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

            If Fex.sessions is not None, sessions will be cleaned separately.

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
                ignore_sessions: (bool) If True, will ignore Fex.sessions
                                 information. Otherwise, method will be applied
                                 separately to each unique session.
            Returns:
                cleaned Fex instance

        """
        if self.sessions is not None:
            if ignore_sessions:
                sessions=None
            else:
                sessions = self.sessions
        return self.__class__(pd.DataFrame(clean(self.values, detrend=detrend,
                                                 standardize=standardize,
                                                 confounds=confounds,
                                                 low_pass=low_pass,
                                                 high_pass=high_pass,
                                                 ensure_finite=ensure_finite,
                                                 t_r=1./np.float(self.sampling_freq),
                                                 sessions=sessions,
                                                 *args, **kwargs),
                                        columns=self.columns),
                                sampling_freq=self.sampling_freq,
                                features=self.features,
                                sessions=self.sessions)

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
            out['weights'] = self.__class__(pd.DataFrame(out['decomposition_object'].components_.T,
                                                        index=self.index,columns=com_names),
                                            sampling_freq=self.sampling_freq,
                                            features=self.features,
                                            sessions=self.sessions)
        if axis == 1:
            out['decomposition_object'].fit(self)
            out['components'] = self.__class__(pd.DataFrame(out['decomposition_object'].transform(self),
                                                            columns=com_names),
                                               sampling_freq=self.sampling_freq,
                                               features=self.features,
                                               sessions=self.sessions)
            out['weights'] = self.__class__(pd.DataFrame(out['decomposition_object'].components_, index=com_names, columns=self.columns), sampling_freq=None).T
        return out

    def extract_mean(self, ignore_sessions=False, *args, **kwargs):
        """ Extract mean of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately
                                    by sessions if available.
        Returns:
            Fex: mean values for each feature

        """
        if ignore_sessions:
            feats = pd.DataFrame(self.mean()).transpose()
            feats.columns = 'mean_' + feats.columns
            return self.__class__(feats, sampling_freq=self.sampling_freq)
        else:
            if self.sessions is None:
                raise ValueError('Fex instance does not have sessions attribute.')
            else:
                feats = pd.DataFrame()
                for k,v in self.itersessions():
                    feats = feats.append(pd.Series(v.mean(), name=k))
                feats.columns = 'mean_' + feats.columns
                return self.__class__(feats, sampling_freq=self.sampling_freq,
                                      sessions=np.unique(self.sessions))

    def extract_min(self, ignore_sessions=False, *args, **kwargs):
        """ Extract minimum of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately
                                    by sessions if available.
        Returns:
            Fex: (Fex) minimum values for each feature

        """
        if ignore_sessions:
            feats = pd.DataFrame(self.min()).transpose()
            feats.columns = 'min_' + feats.columns
            return self.__class__(feats, sampling_freq=self.sampling_freq)
        else:
            if self.sessions is None:
                raise ValueError('Fex instance does not have sessions attribute.')
            else:
                feats = pd.DataFrame()
                for k,v in self.itersessions():
                    feats = feats.append(pd.Series(v.min(), name=k))
                feats.columns = 'min_' + feats.columns
                return self.__class__(feats, sampling_freq=self.sampling_freq,
                                      sessions=np.unique(self.sessions))

    def extract_max(self, ignore_sessions=False, *args, **kwargs):
        """ Extract maximum of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately
                                    by sessions if available.
        Returns:
            fex: (Fex) maximum values for each feature

        """
        if ignore_sessions:
            feats = pd.DataFrame(self.max()).transpose()
            feats.columns = 'max_' + feats.columns
            return self.__class__(feats, sampling_freq=self.sampling_freq)
        else:
            if self.sessions is None:
                raise ValueError('Fex instance does not have sessions attribute.')
            else:
                feats = pd.DataFrame()
                for k,v in self.itersessions():
                    feats = feats.append(pd.Series(v.max(), name=k))
                feats.columns = 'max_' + feats.columns
                return self.__class__(feats, sampling_freq=self.sampling_freq,
                                      sessions=np.unique(self.sessions))

    def extract_summary(self, mean=None, max=None, min=None,
                        ignore_sessions=False, *args, **kwargs):
        """ Extract summary of multiple features

        Args:
            mean: (bool) extract mean of features
            max: (bool) extract max of features
            min: (bool) extract min of features
            ignore_sessions: (bool) ignore sessions or extract separately
                                    by sessions if available.

        Returns:
            fex: (Fex) maximum values for each feature

        """

        out = self.__class__()
        if mean is not None:
            out = out.append(self.extract_mean(*args, **kwargs), axis=1)
        if max is not None:
            out = out.append(self.extract_max(*args, **kwargs), axis=1)
        if min is not None:
            out = out.append(self.extract_min(*args, **kwargs), axis=1)
        return out

    def extract_wavelet(self, freq, num_cyc=3, mode='complex',
                        ignore_sessions=False):
        ''' Perform feature extraction by convolving with a complex morlet
            wavelet

            Args:
                freq: (float) frequency to extract
                num_cyc: (float) number of cycles for wavelet
                mode: (str) feature to extract, e.g.,
                            ['complex','filtered','phase','magnitude','power']
                ignore_sessions: (bool) ignore sessions or extract separately
                                        by sessions if available.
            Returns:
                convolved: (Fex instance)
        '''
        wav = wavelet(freq, sampling_freq=self.sampling_freq, num_cyc=num_cyc)
        if self.sessions is None:
            convolved = self.__class__(pd.DataFrame({x:convolve(y, wav, mode='same') for x,y in self.iteritems()}), sampling_freq=self.sampling_freq)
        else:
            if ignore_sessions:
                convolved = self.__class__(pd.DataFrame({x:convolve(y, wav, mode='same') for x,y in self.iteritems()}), sampling_freq=self.sampling_freq)
            else:
                convolved = self.__class__(sampling_freq=self.sampling_freq)
                for k,v in self.itersessions():
                    session = self.__class__(pd.DataFrame({x:convolve(y, wav, mode='same') for x,y in v.iteritems()}), sampling_freq=self.sampling_freq)
                    convolved = convolved.append(session, session_id=k)
        if mode is 'complex':
            convolved = convolved
        elif mode is 'filtered':
            convolved = np.real(convolved)
        elif mode is 'phase':
            convolved = np.angle(convolved)
        elif mode is 'magnitude':
            convolved = np.abs(convolved)
        elif mode is 'power':
            convolved = np.abs(convolved)**2
        else:
            raise ValueError("Mode must be ['complex','filtered','phase',"
                             "'magnitude','power']")
        convolved = self.__class__(convolved, sampling_freq=self.sampling_freq,
                                   features=self.features,
                                   sessions=self.sessions)
        convolved.columns = 'f' + '%s' % round(freq, 2) + '_' + mode + '_' + self.columns
        return convolved

    def extract_multi_wavelet(self, min_freq=.06, max_freq=.66, bank=8, *args, **kwargs):
        ''' Convolve with a bank of morlet wavelets. Wavelets are equally
            spaced from min to max frequency. See extract_wavelet for more
            information and options.

            Args:
                min_freq: (float) minimum frequency to extract
                max_freq: (float) maximum frequency to extract
                bank: (int) size of wavelet bank
                num_cyc: (float) number of cycles for wavelet
                mode: (str) feature to extract, e.g.,
                            ['complex','filtered','phase','magnitude','power']
                ignore_sessions: (bool) ignore sessions or extract separately
                                        by sessions if available.
            Returns:
                convolved: (Fex instance)
        '''
        out = []
        for f in np.geomspace(min_freq, max_freq, bank):
            out.append(self.extract_wavelet(f, *args, **kwargs))
        return self.__class__(pd.concat(out, axis=1),
                              sampling_freq=self.sampling_freq,
                              features=self.features,
                              sessions=self.sessions)

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
        freqs = np.geomspace(min_freq, max_freq,bank)
        wavs, hzs = [],[]
        for i, f in enumerate(freqs):
            wav = wavelet(f, sampling_freq=target_hz)
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
                convolved = np.apply_along_axis(lambda m: np.convolve(m, cm, mode='full'),axis=1,arr=_d.as_matrix())
                # Extract bin features.
                out = pd.DataFrame(convolved.T).apply(calc_hist_auc,args=(None))
                colnames = ['pos'+str(i)+'_hz_'+hzs[iw]+'_'+feat for i in range(6)]
                colnames.extend(['neg'+str(i)+'_hz_'+hzs[iw]+'_'+feat for i in range(6)])
                out = out.T
                out.columns = colnames
                feats = pd.concat([feats, out], axis=1)
        return self.__class__(feats, sampling_freq=self.sampling_freq,
                              features=self.features,
                              sessions=self.sessions)

class Facet(Fex):
    """
    Facet is a subclass of Fex.
    You can use the Facet subclass to load iMotions-FACET data files.
    It will also have Facet specific methods.
    """
    def read_file(self, *args, **kwargs):
        super(Fex, self).__init__(read_facet(self.filename, *args, **kwargs), *args, **kwargs)

    def calc_pspi(self, *args, **kwargs):
        out = self['AU4'] + self[['AU6','AU7']].max(axis=1) + self[['AU9','AU10']].max(axis=1) + self['AU43']
        return out

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

    def calc_pspi(self, *args, **kwargs):
        out = self['AU04_r'] + self[['AU06_r','AU07_r']].max(axis=1) + self[['AU09_r','AU10_r']].max(axis=1) + self['AU45_r']
        return out
