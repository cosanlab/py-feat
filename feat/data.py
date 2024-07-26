"""
Py-FEAT Data classes.
"""

import warnings

# Suppress nilearn warnings that come from importing nltools
warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")
import os
from typing import Iterable
from copy import deepcopy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from copy import deepcopy
from functools import reduce
from nltools.data import Adjacency
from nltools.stats import downsample, upsample, regress
from nltools.utils import set_decomposition_algorithm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from torchvision.transforms import Compose
from torchvision import transforms
from torchvision.io import read_image, read_video
from torch.utils.data import Dataset
from torch import swapaxes
from feat.transforms import Rescale
from feat.utils import flatten_list
from feat.utils.io import read_feat, read_openface
from feat.utils.stats import wavelet, calc_hist_auc, cluster_identities
from feat.plotting import (
    plot_face,
    draw_lineface,
    draw_facepose,
    load_viz_model,
    face_part_path,
    draw_plotly_landmark,
    face_polygon_svg,
    draw_plotly_au,
    draw_plotly_pose,
    emotion_annotation_position,
)
from feat.pretrained import AU_LANDMARK_MAP
from feat.utils import flatten_list
from feat.utils.io import load_pil_img
from nilearn.signal import clean
from scipy.signal import convolve
from scipy.stats import ttest_1samp, ttest_ind
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from textwrap import wrap
import torch
from PIL import Image
import logging
import av
from itertools import islice
import plotly.graph_objects as go

__all__ = [
    "FexSeries",
    "Fex",
    "ImageDataset",
    "VideoDataset",
    "_inverse_face_transform",
    "_inverse_landmark_transform",
]


class FexSeries(Series):
    """
    This is a sub-class of pandas series. While not having additional methods
    of it's own required to retain normal slicing functionality for the
    Fex class, i.e. how slicing is typically handled in pandas.
    All methods should be called on Fex below.
    """

    def __init__(self, *args, **kwargs):
        ### Columns ###
        self.au_columns = kwargs.pop("au_columns", None)
        self.emotion_columns = kwargs.pop("emotion_columns", None)
        self.facebox_columns = kwargs.pop("facebox_columns", None)
        self.landmark_columns = kwargs.pop("landmark_columns", None)
        self.facepose_columns = kwargs.pop("facepose_columns", None)
        self.identity_columns = kwargs.pop("identity_columns", None)
        self.gaze_columns = kwargs.pop("gaze_columns", None)
        self.time_columns = kwargs.pop(
            "time_columns", ["Timestamp", "MediaTime", "FrameNo", "FrameTime"]
        )
        self.design_columns = kwargs.pop("design_columns", None)

        ### Meta data ###
        self.filename = kwargs.pop("filename", None)
        self.sampling_freq = kwargs.pop("sampling_freq", None)
        self.detector = kwargs.pop("detector", None)
        self.face_model = kwargs.pop("face_model", None)
        self.landmark_model = kwargs.pop("landmark_model", None)
        self.au_model = kwargs.pop("au_model", None)
        self.emotion_model = kwargs.pop("emotion_model", None)
        self.facepose_model = kwargs.pop("facepose_model", None)
        self.identity_model = kwargs.pop("identity_model", None)
        self.features = kwargs.pop("features", None)
        self.sessions = kwargs.pop("sessions", None)
        super().__init__(*args, **kwargs)

    _metadata = [
        "au_columns",
        "emotion_columns",
        "facebox_columns",
        "landmark_columns",
        "facepose_columns",
        "identity_columns",
        "gaze_columns",  # TODO: Not currently supported
        "time_columns",
        "design_columns",
        "fex_columns",
        "filename",
        "sampling_freq",
        "features",
        "sessions",
        "detector",
        "face_model",
        "landmark_model",
        "au_model",
        "emotion_model",
        "facepose_model",
        "identity_model",
        "verbose",
    ]

    @property
    def _constructor(self):
        return FexSeries

    @property
    def _constructor_expanddim(self):
        return Fex

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self"""
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    @property
    def aus(self):
        """Returns the Action Units data

        Returns:
            DataFrame: Action Units data
        """
        return self[self.au_columns]

    @property
    def emotions(self):
        """Returns the emotion data

        Returns:
            DataFrame: emotion data
        """
        return self[self.emotion_columns]

    @property
    def landmarks(self):
        """Returns the landmark data

        Returns:
            DataFrame: landmark data
        """
        return self[self.landmark_columns]

    # DEPRECATE
    @property
    def landmark(self):
        """Returns the landmark data

        Returns:
            DataFrame: landmark data
        """
        warnings.warn(
            "Fex.landmark has now been renamed to Fex.landmarks", DeprecationWarning
        )
        return self[self.landmark_columns]

    @property
    def poses(self):
        """Returns the facepose data

        Returns:
            DataFrame: facepose data
        """

        return self[self.facepose_columns]

    # DEPRECATE
    @property
    def facepose(self):
        """Returns the facepose data

        Returns:
            DataFrame: facepose data
        """

        warnings.warn(
            "Fex.facepose has now been renamed to Fex.poses", DeprecationWarning
        )
        return self[self.facepose_columns]

    @property
    def inputs(self):
        """Returns input column as string

        Returns:
            string: path to input image
        """
        return self["input"]

    # DEPRECATE
    @property
    def input(self):
        """Returns input column as string

        Returns:
            string: path to input image
        """
        warnings.warn(
            "Fex.input has now been renamed to Fex.inputs", DeprecationWarning
        )
        return self["input"]

    @property
    def landmarks_x(self):
        """Returns the x landmarks.

        Returns:
            DataFrame: x landmarks.
        """
        x_cols = [col for col in self.landmark_columns if "x" in col]
        return self[x_cols]

    # DEPRECATE
    @property
    def landmark_x(self):
        """Returns the x landmarks.

        Returns:
            DataFrame: x landmarks.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.landmark_x has been renamed to Fex.landmarks_x", DeprecationWarning
            )

        x_cols = [col for col in self.landmark_columns if "x" in col]
        return self[x_cols]

    @property
    def landmarks_y(self):
        """Returns the y landmarks.

        Returns:
            DataFrame: y landmarks.
        """
        y_cols = [col for col in self.landmark_columns if "y" in col]
        return self[y_cols]

    # DEPRECATE
    @property
    def landmark_y(self):
        """Returns the y landmarks.

        Returns:
            DataFrame: y landmarks.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.landmark_y has been renamed to Fex.landmarks_y", DeprecationWarning
            )

        y_cols = [col for col in self.landmark_columns if "y" in col]
        return self[y_cols]

    @property
    def faceboxes(self):
        """Returns the facebox data

        Returns:
            DataFrame: facebox data
        """
        return self[self.facebox_columns]

    # DEPRECATE
    @property
    def facebox(self):
        """Returns the facebox data

        Returns:
            DataFrame: facebox data
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.facebox has been renamed to Fex.faceboxes", DeprecationWarning
            )

        return self[self.facebox_columns]

    @property
    def identity(self):
        """Returns the identity data

        Returns:
            DataFrame: identity data
        """
        return self[self.identity_columns]

    @property
    def time(self):
        """Returns the time data

        Returns:
            DataFrame: time data
        """
        return self[self.time_columns]

    @property
    def design(self):
        """Returns the design data

        Returns:
            DataFrame: time data
        """
        return self[self.design_columns]

    @property
    def info(self):
        """Print class meta data."""
        attr_list = []
        for name in self._metadata:
            attr_list.append(name + ": " + str(getattr(self, name, None)) + "\n")
        print(f"{self.__class__}\n" + "".join(attr_list))

    def plot_detections(self, *args, **kwargs):
        """Alias for Fex.plot_detections"""
        return Fex(self).T.__finalize__(self).plot_detections(*args, **kwargs)

    def iplot_detections(self, *args, **kwargs):
        """Alias for Fex.iplot_detections"""
        return Fex(self).T.__finalize__(self).iplot_detections(*args, **kwargs)


# TODO: Switch all print statements to respect verbose
class Fex(DataFrame):
    """Fex is a class to represent facial expression (Fex) data

    Fex class is  an enhanced pandas dataframe, with extra attributes and methods to help with facial expression data analysis.

    Args:
        filename: (str, optional) path to file
        detector: (str, optional) name of software used to extract Fex. Currently only
        'Feat' is supported
        sampling_freq (float, optional): sampling rate of each row in Hz; defaults to None
        features (pd.Dataframe, optional): features that correspond to each Fex row
        sessions: Unique values indicating rows associated with a specific session (e.g., trial, subject, etc).Must be a 1D array of n_samples elements; defaults to None
    """

    _metadata = [
        "au_columns",
        "emotion_columns",
        "facebox_columns",
        "landmark_columns",
        "facepose_columns",
        "identity_columns",
        "gaze_columns",  # TODO: Not currently supported
        "time_columns",
        "design_columns",
        "fex_columns",
        "filename",
        "sampling_freq",
        "features",
        "sessions",
        "detector",
        "face_model",
        "landmark_model",
        "au_model",
        "emotion_model",
        "facepose_model",
        "identity_model",
        "verbose",
    ]

    def __finalize__(self, other, method=None, **kwargs):
        """Propagate metadata from other to self"""
        self = super().__finalize__(other, method=method, **kwargs)
        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                print("self", name, self.au_columns, other.left.au_columns)
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        return self

    def __init__(self, *args, **kwargs):
        ### Columns ###
        self.au_columns = kwargs.pop("au_columns", None)
        self.emotion_columns = kwargs.pop("emotion_columns", None)
        self.facebox_columns = kwargs.pop("facebox_columns", None)
        self.landmark_columns = kwargs.pop("landmark_columns", None)
        self.facepose_columns = kwargs.pop("facepose_columns", None)
        self.identity_columns = kwargs.pop("identity_columns", None)
        self.gaze_columns = kwargs.pop("gaze_columns", None)
        self.time_columns = kwargs.pop("time_columns", None)
        self.design_columns = kwargs.pop("design_columns", None)

        ### Meta data ###
        self.filename = kwargs.pop("filename", None)
        self.sampling_freq = kwargs.pop("sampling_freq", None)
        self.detector = kwargs.pop("detector", None)
        self.face_model = kwargs.pop("face_model", None)
        self.landmark_model = kwargs.pop("landmark_model", None)
        self.au_model = kwargs.pop("au_model", None)
        self.emotion_model = kwargs.pop("emotion_model", None)
        self.facepose_model = kwargs.pop("facepose_model", None)
        self.identity_model = kwargs.pop("identity_model", None)
        self.features = kwargs.pop("features", None)
        self.sessions = kwargs.pop("sessions", None)

        self.verbose = kwargs.pop("verbose", False)

        super().__init__(*args, **kwargs)
        if self.sessions is not None:
            if not len(self.sessions) == len(self):
                raise ValueError("Make sure sessions is same length as data.")
            self.sessions = np.array(self.sessions)

        # Set _metadata attributes on series: Kludgy solution
        for k in self:
            self[k].sampling_freq = self.sampling_freq
            self[k].sessions = self.sessions

    @property
    def _constructor(self):
        return Fex

    @property
    def _constructor_sliced(self):
        """
        Propagating custom metadata from sub-classed dfs to sub-classed series is not
        automatically handled. See: https://github.com/pandas-dev/pandas/issues/19850
        _constructor_sliced (which dataframes call when their return type is a series
        can only return a function definition or class definition. So to make sure we
        propagate attributes from Fex -> FexSeries we define another
        function that calls .__finalize__ on the returned FexSeries

        Inspired by how GeoPandas subclasses dataframes. See their _constructor_sliced
        here:
        https://github.com/geopandas/geopandas/blob/2eac5e212a7e2ebbca71f35707a2a196e4b09527/geopandas/geodataframe.py#L1460

        And their constructor function here: https://github.com/geopandas/geopandas/blob/2eac5e212a7e2ebbca71f35707a2a196e4b09527/geopandas/geoseries.py#L31
        """

        def _fexseries_constructor(*args, **kwargs):
            return FexSeries(*args, **kwargs).__finalize__(self)

        return _fexseries_constructor

    @property
    def aus(self):
        """Returns the Action Units data

        Returns Action Unit data using the columns set in fex.au_columns.

        Returns:
            DataFrame: Action Units data
        """
        return self[self.au_columns]

    @property
    def emotions(self):
        """Returns the emotion data

        Returns emotions data using the columns set in fex.emotion_columns.

        Returns:
            DataFrame: emotion data
        """
        return self[self.emotion_columns]

    @property
    def landmarks(self):
        """Returns the landmark data

        Returns landmark data using the columns set in fex.landmark_columns.

        Returns:
            DataFrame: landmark data
        """
        return self[self.landmark_columns]

    @property
    def landmark(self):
        """Returns the landmark data

        Returns landmark data using the columns set in fex.landmark_columns.

        Returns:
            DataFrame: landmark data
        """
        warnings.warn(
            "Fex.landmark has now been renamed to Fex.landmarks", DeprecationWarning
        )
        return self[self.landmark_columns]

    @property
    def poses(self):
        """Returns the facepose data using the columns set in fex.facepose_columns

        Returns:
            DataFrame: facepose data
        """
        return self[self.facepose_columns]

    # DEPRECATE
    @property
    def facepose(self):
        """Returns the facepose data using the columns set in fex.facepose_columns

        Returns:
            DataFrame: facepose data
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.facepose has now been renamed to Fex.poses", DeprecationWarning
            )

        return self[self.facepose_columns]

    @property
    def inputs(self):
        """Returns input column as string

        Returns input data in the "input" column.

        Returns:
            string: path to input image
        """
        return self["input"]

    # DEPRECATE
    @property
    def input(self):
        """Returns input column as string

        Returns input data in the "input" column.

        Returns:
            string: path to input image
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.input has now been renamed to Fex.inputs", DeprecationWarning
            )

        return self["input"]

    @property
    def landmarks_x(self):
        """Returns the x landmarks.

        Returns:
            DataFrame: x landmarks.
        """
        x_cols = [col for col in self.landmark_columns if "x" in col]
        return self[x_cols]

    # DEPRECATE
    @property
    def landmark_x(self):
        """Returns the x landmarks.

        Returns:
            DataFrame: x landmarks.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.landmark_x has been renamed to Fex.landmarks_x", DeprecationWarning
            )
        x_cols = [col for col in self.landmark_columns if "x" in col]
        return self[x_cols]

    @property
    def landmarks_y(self):
        """Returns the y landmarks.

        Returns:
            DataFrame: y landmarks.
        """
        y_cols = [col for col in self.landmark_columns if "y" in col]
        return self[y_cols]

    # DEPRECATE
    @property
    def landmark_y(self):
        """Returns the y landmarks.

        Returns:
            DataFrame: y landmarks.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.landmark_y has been renamed to Fex.landmarks_y", DeprecationWarning
            )

        y_cols = [col for col in self.landmark_columns if "y" in col]
        return self[y_cols]

    @property
    def faceboxes(self):
        """Returns the facebox data

        Returns:
            DataFrame: facebox data
        """
        return self[self.facebox_columns]

    # DEPRECATE
    @property
    def facebox(self):
        """Returns the facebox data

        Returns:
            DataFrame: facebox data
        """

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Fex.facebox has been renamed to Fex.faceboxes", DeprecationWarning
            )

        return self[self.facebox_columns]

    @property
    def identities(self):
        """Returns the identity labels

        Returns:
            DataFrame: identity data
        """
        return self[self.identity_columns[0]]

    @property
    def identity_embeddings(self):
        """Returns the identity embeddings

        Returns:
            DataFrame: identity data
        """
        return self[self.identity_columns[1:]]

    @property
    def time(self):
        """Returns the time data

        Returns the time information using fex.time_columns.

        Returns:
            DataFrame: time data
        """
        return self[self.time_columns]

    @property
    def design(self):
        """Returns the design data

        Returns the study design information using columns in fex.design_columns.

        Returns:
            DataFrame: time data
        """
        return self[self.design_columns]

    def read_file(self):
        """Loads file into FEX class

        Returns:
            DataFrame: Fex class
        """
        if self.detector == "OpenFace":
            return self.read_openface(self.filename)

        return self.read_feat(self.filename)

    @property
    def info(self):
        """Print all meta data of fex

        Loops through metadata set in self._metadata and prints out the information.
        """
        attr_list = []
        for name in self._metadata:
            attr_list.append(name + ": " + str(getattr(self, name, None)) + "\n")
        print(f"{self.__class__}\n" + "".join(attr_list))

    def _update_extracted_colnames(self, prefix=None, mode="replace"):
        cols2update = [
            "au_columns",
            "emotion_columns",
            "facebox_columns",
            "landmark_columns",
            "facepose_columns",
            # "gaze_columns", # doesn't currently exist
            "time_columns",
        ]

        # Existing columns to handle different __init__s
        cols2update = list(
            filter(lambda col: getattr(self, col) is not None, cols2update)
        )
        original_vals = [getattr(self, c) for c in cols2update]

        # Ignore prefix and remove any existing
        if mode == "reset":
            new_vals = [
                list(map(lambda name: "".join(name.split("_")[1:]), names))
                for names in original_vals
            ]
            _ = [setattr(self, col, val) for col, val in zip(cols2update, new_vals)]
            return

        if not isinstance(prefix, list):
            prefix = [prefix]

        for i, p in enumerate(prefix):
            current_vals = [getattr(self, c) for c in cols2update]
            new_vals = [
                list(map(lambda name: f"{p}_{name}", names)) for names in original_vals
            ]
            if i == 0:
                update = new_vals
            else:
                update = [current + new for current, new in zip(current_vals, new_vals)]
            _ = [setattr(self, col, val) for col, val in zip(cols2update, update)]

    def _parse_features_labels(self, X, y):
        feature_groups = [
            "sessions",
            "emotions",
            "aus",
            "poses",
            "landmarks",
            "faceboxes",
        ]

        # String attribute access
        if isinstance(X, str) and any(
            map(lambda feature: feature in X, feature_groups)
        ):
            X = X.split(",") if "," in X else [X]
            mX = []
            for x in X:
                mX.append(getattr(self, x))

            mX = pd.concat(mX, axis=1)
            if X == ["sessions"]:
                mX.columns = X

        elif isinstance(X, list):
            mX = self[X]
        else:
            mX = X

        if isinstance(y, str) and any(
            map(lambda feature: feature in y, feature_groups)
        ):
            y = y.split(",") if "," in y else [y]
            my = []
            for yy in y:
                my.append(getattr(self, yy))

            my = pd.concat(my, axis=1)
            if y == ["sessions"]:
                my.columns = y

        elif isinstance(y, list):
            my = self[y]
        else:
            my = y

        return mX, my

    ###   Class Methods   ###
    def read_feat(self, filename=None, *args, **kwargs):
        """Reads facial expression detection results from Feat Detector

        Args:
            filename (string, optional): Path to file. Defaults to None.

        Returns:
            Fex
        """
        # Check if filename exists in metadata.
        if filename is None:
            if self.filename:
                filename = self.filename
            else:
                raise ValueError("filename must be specified.")
        result = read_feat(filename, *args, **kwargs)
        return result

    def read_openface(self, filename=None, *args, **kwargs):
        """Reads facial expression detection results from OpenFace

        Args:
            filename (string, optional): Path to file. Defaults to None.

        Returns:
            Fex
        """
        if filename is None:
            if self.filename:
                filename = self.filename
            else:
                raise ValueError("filename must be specified.")
        result = read_openface(filename, *args, **kwargs)
        for name in self._metadata:
            attr_value = getattr(self, name, None)
            if attr_value and getattr(result, name, None) == None:
                setattr(result, name, attr_value)
        return result

    def itersessions(self):
        """Iterate over Fex sessions as (session, series) pairs.

        Returns:
            it: a generator that iterates over the sessions of the fex instance

        """
        for x in np.unique(self.sessions):
            yield x, self.loc[self.sessions == x, :]

    def append(self, data, session_id=None, axis=0):
        """Append a new Fex object to an existing object

        Args:
            data: (Fex) Fex instance to append
            session_id: session label
            axis: ([0,1]) Axis to append. Rows=0, Cols=1
        Returns:
            Fex instance
        """
        if not isinstance(data, self.__class__):
            raise ValueError("Make sure data is a Fex instance.")

        if self.empty:
            out = data.copy()
            if session_id is not None:
                out.sessions = np.repeat(session_id, len(data))
        else:
            if self.sampling_freq != data.sampling_freq:
                raise ValueError(
                    "Make sure Fex objects have the same " "sampling frequency"
                )
            if axis == 0:
                out = self.__class__(
                    pd.concat([self, data], axis=axis, ignore_index=True),
                    sampling_freq=self.sampling_freq,
                ).__finalize__(self)
                if session_id is not None:
                    out.sessions = np.hstack(
                        [self.sessions, np.repeat(session_id, len(data))]
                    )
                if self.features is not None:
                    if data.features is not None:
                        if self.features.shape[1] == data.features.shape[1]:
                            out.features = self.features.append(
                                data.features, ignore_index=True
                            )
                        else:
                            raise ValueError(
                                "Different number of features in new dataset."
                            )
                    else:
                        out.features = self.features
                elif data.features is not None:
                    out = data.features
            elif axis == 1:
                out = self.__class__(
                    pd.concat([self, data], axis=axis), sampling_freq=self.sampling_freq
                ).__finalize__(self)
                if self.sessions is not None:
                    if data.sessions is not None:
                        if np.array_equal(self.sessions, data.sessions):
                            out.sessions = self.sessions
                        else:
                            raise ValueError("Both sessions must be identical.")
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
                raise ValueError("Axis must be 1 or 0.")
        return out

    def regress(self, X, y, fit_intercept=True, *args, **kwargs):
        """Regress using nltools.stats.regress.

        fMRI-like regression to predict Fex activity (y) from set of regressors (X).

        Args:
            X (list or str): Independent variable to predict.
            y (list or str): Dependent variable to be predicted.
            fit_intercept (bool): Whether to add intercept before fitting. Defaults to True.

        Returns:
            Dataframe of betas, ses, t-stats, p-values, df, residuals
        """

        mX, my = self._parse_features_labels(X, y)

        if fit_intercept:
            mX["intercept"] = 1

        b, se, t, p, df, res = regress(mX.to_numpy(), my.to_numpy(), *args, **kwargs)
        b_df = pd.DataFrame(b, index=mX.columns, columns=my.columns)
        se_df = pd.DataFrame(se, index=mX.columns, columns=my.columns)
        t_df = pd.DataFrame(t, index=mX.columns, columns=my.columns)
        p_df = pd.DataFrame(p, index=mX.columns, columns=my.columns)
        df_df = pd.DataFrame(np.tile(df, (2, 1)), index=mX.columns, columns=my.columns)
        res_df = pd.DataFrame(res, columns=my.columns)
        return b_df, se_df, t_df, p_df, df_df, res_df

    def ttest_1samp(self, popmean=0):
        """Conducts 1 sample ttest.

        Uses scipy.stats.ttest_1samp to conduct 1 sample ttest

        Args:
            popmean (int, optional): Population mean to test against. Defaults to 0.
            threshold_dict ([type], optional): Dictonary for thresholding. Defaults to None. [NOT IMPLEMENTED]

        Returns:
            t, p: t-statistics and p-values
        """
        return ttest_1samp(self, popmean)

    def ttest_ind(self, col, sessions=None):
        """Conducts 2 sample ttest.

        Uses scipy.stats.ttest_ind to conduct 2 sample ttest on column col between sessions.

        Args:
            col (str): Column names to compare in a t-test between sessions
            session (array-like): session name to query Fex.sessions, otherwise uses the
            unique values in Fex.sessions.

        Returns:
            t, p: t-statistics and p-values
        """

        if sessions is None:
            sessions = pd.Series(self.sessions).unique()

        if len(sessions) != 2:
            raise ValueError(
                f"There must be exactly 2 session types to perform an independent t-test but {len(sessions)} were found."
            )

        sess1, sess2 = sessions
        a_mask = self.sessions == sess1
        a_mask = a_mask.values if isinstance(self.sessions, pd.Series) else a_mask
        b_mask = self.sessions == sess2
        b_mask = b_mask.values if isinstance(self.sessions, pd.Series) else b_mask
        a = self.loc[a_mask, col]
        b = self.loc[b_mask, col]

        return ttest_ind(a, b)

    def predict(
        self, X, y, model=LinearRegression, cv_kwargs={"cv": 5}, *args, **kwargs
    ):
        """Predicts y from X using a sklearn model.

        Predict a variable of interest y using your model of choice from X, which can be a list of columns of the Fex instance or a dataframe.

        Args:
            X (list or DataFrame): List of column names or dataframe to be used as features for prediction
            y (string or array): y values to be predicted
            model (class, optional): Any sklearn model. Defaults to LinearRegression.
            args, kwargs: Model arguments

        Returns:
            model: Fit model instance.
        """

        mX, my = self._parse_features_labels(X, y)

        # user passes an unintialized class, e.g. LogisticRegression
        if isinstance(model, type):
            clf = model(*args, **kwargs)
        else:
            # user passes an initialized estimator or pipeline, e.g. LogisticRegression()
            clf = model
        scores = cross_val_score(clf, mX, my, **cv_kwargs)
        _ = clf.fit(mX, my)
        return clf, scores

    def downsample(self, target, **kwargs):
        """Downsample Fex columns. Relies on nltools.stats.downsample,
           but ensures that returned object is a Fex object.

        Args:
            target(float): downsampling target, typically in samples not seconds
            kwargs: additional inputs to nltools.stats.downsample

        """
        df_ds = downsample(
            self, sampling_freq=self.sampling_freq, target=target, **kwargs
        ).__finalize__(self)
        df_ds.sampling_freq = target

        if self.features is not None:
            ds_features = downsample(
                self.features, sampling_freq=self.sampling_freq, target=target, **kwargs
            )
        else:
            ds_features = self.features
        df_ds.features = ds_features
        return df_ds

    def isc(self, col, index="frame", columns="input", method="pearson"):
        """[summary]

        Args:
            col (str]): Column name to compute the ISC for.
            index (str, optional): Column to be used in computing ISC. Usually this would be the column identifying the time such as the number of the frame. Defaults to "frame".
            columns (str, optional): Column to be used for ISC. Usually this would be the column identifying the video or subject. Defaults to "input".
            method (str, optional): Method to use for correlation pearson, kendall, or spearman. Defaults to "pearson".

        Returns:
            DataFrame: Correlation matrix with index as colmns
        """
        if index is None:
            index = "frame"
        if columns is None:
            columns = "input"
        mat = pd.pivot_table(self, index=index, columns=columns, values=col).corr(
            method=method
        )
        return mat

    def upsample(self, target, target_type="hz", **kwargs):
        """Upsample Fex columns. Relies on nltools.stats.upsample,
            but ensures that returned object is a Fex object.

        Args:
            target(float): upsampling target, default 'hz' (also 'samples', 'seconds')
            kwargs: additional inputs to nltools.stats.upsample

        """
        df_us = upsample(
            self,
            sampling_freq=self.sampling_freq,
            target=target,
            target_type=target_type,
            **kwargs,
        )
        if self.features is not None:
            us_features = upsample(
                self.features,
                sampling_freq=self.sampling_freq,
                target=target,
                target_type=target_type,
                **kwargs,
            )
        else:
            us_features = self.features
        return self.__class__(df_us, sampling_freq=target, features=us_features)

    def distance(self, method="euclidean", **kwargs):
        """Calculate distance between rows within a Fex() instance.

        Args:
            method: type of distance metric (can use any scikit learn or
                    sciypy metric)

        Returns:
            dist: Outputs a 2D distance matrix.

        """
        return Adjacency(
            pairwise_distances(self, metric=method, **kwargs), matrix_type="Distance"
        )

    def rectification(self, std=3):
        """Removes time points when the face position moved
        more than N standard deviations from the mean.

        Args:
            std (default 3): standard deviation from mean to remove outlier face locations
        Returns:
            data: cleaned FEX object

        """

        if self.facebox_columns and self.au_columns and self.emotion_columns:
            cleaned = deepcopy(self)
            face_columns = self.facebox_columns
            x_m = self.FaceRectX.mean()
            x_std = self.FaceRectX.std()
            y_m = self.FaceRectY.mean()
            y_std = self.FaceRectY.std()
            x_bool = (self.FaceRectX > std * x_std + x_m) | (
                self.FaceRectX < x_m - std * x_std
            )
            y_bool = (self.FaceRectY > std * y_std + y_m) | (
                self.FaceRectY < y_m - std * y_std
            )
            xy_bool = x_bool | y_bool
            cleaned.loc[
                xy_bool, face_columns + self.au_columns + self.emotion_columns
            ] = np.nan
            return cleaned
        else:
            raise ValueError("Facebox columns need to be defined.")

    def baseline(self, baseline="median", normalize=None, ignore_sessions=False):
        """Reference a Fex object to a baseline.

        Args:
            method: {'median', 'mean', 'begin', FexSeries instance}. Will subtract baseline from Fex object (e.g., mean, median).  If passing a Fex object, it will treat that as the baseline.
            normalize: (str). Can normalize results of baseline. Values can be [None, 'db','pct']; default None.
            ignore_sessions: (bool) If True, will ignore Fex.sessions information. Otherwise, method will be applied separately to each unique session.

        Returns:
            Fex object
        """
        if self.sessions is None or ignore_sessions:
            out = self.copy()
            if type(baseline) == str:
                if baseline == "median":
                    baseline_values = out.median()
                elif baseline == "mean":
                    baseline_values = out.mean()
                elif baseline == "begin":
                    baseline_values = out.iloc[0, :]
                else:
                    raise ValueError(
                        "%s is not implemented please use {mean, median, Fex}"
                        % baseline
                    )
            elif isinstance(baseline, (Series, FexSeries)):
                baseline_values = baseline
            elif isinstance(baseline, (Fex, DataFrame)):
                raise ValueError("Must pass in a FexSeries not a FexSeries Instance.")

            if normalize == "db":
                out = 10 * np.log10(out - baseline_values) / baseline_values
            if normalize == "pct":
                out = 100 * (out - baseline_values) / baseline_values
            else:
                out = out - baseline_values
        else:
            out = self.__class__(sampling_freq=self.sampling_freq)
            for k, v in self.itersessions():
                if type(baseline) == str:
                    if baseline == "median":
                        baseline_values = v.median()
                    elif baseline == "mean":
                        baseline_values = v.mean()
                    elif baseline == "begin":
                        baseline_values = v.iloc[0, :]
                    else:
                        raise ValueError(
                            "%s is not implemented please use {mean, median, Fex}"
                            % baseline
                        )
                elif isinstance(baseline, (Series, FexSeries)):
                    baseline_values = baseline
                elif isinstance(baseline, (Fex, DataFrame)):
                    raise ValueError(
                        "Must pass in a FexSeries not a FexSeries Instance."
                    )

                if normalize == "db":
                    out = out.append(
                        10 * np.log10(v - baseline_values) / baseline_values,
                        session_id=k,
                    )
                if normalize == "pct":
                    out = out.append(
                        100 * (v - baseline_values) / baseline_values, session_id=k
                    )
                else:
                    out = out.append(v - baseline_values, session_id=k)
        return out.__finalize__(self)

    def clean(
        self,
        detrend=True,
        standardize=True,
        confounds=None,
        low_pass=None,
        high_pass=None,
        ensure_finite=False,
        ignore_sessions=False,
        *args,
        **kwargs,
    ):
        """Clean Time Series signal

        This function wraps nilearn functionality and can filter, denoise,
        detrend, etc.

        See http://nilearn.github.io/modules/generated/nilearn.signal.clean.html

        This function can do several things on the input signals, in
        the following order: detrend, standardize, remove confounds, low and high-pass filter

        If Fex.sessions is not None, sessions will be cleaned separately.

        Args:
            confounds: (numpy.ndarray, str or list of Confounds timeseries) Shape must be (instant number, confound number), or just (instant number,). The number of time instants in signals and confounds must be identical (i.e. signals.shape[0] == confounds.shape[0]). If a string is provided, it is assumed to be the name of a csv file containing signals as columns, with an optional one-line header. If a list is provided, all confounds are removed from the input signal, as if all were in the same array.
            low_pass: (float) low pass cutoff frequencies in Hz.
            high_pass: (float) high pass cutoff frequencies in Hz.
            detrend: (bool) If detrending should be applied on timeseries (before confound removal)
            standardize: (bool) If True, returned signals are set to unit variance.
            ensure_finite: (bool) If True, the non-finite values (NANs and infs) found in the data will be replaced by zeros.
            ignore_sessions: (bool) If True, will ignore Fex.sessions information. Otherwise, method will be applied separately to each unique session.

        Returns:
            cleaned Fex instance
        """
        if self.sessions is not None:
            if ignore_sessions:
                sessions = None
            else:
                sessions = self.sessions
        else:
            sessions = None
        return self.__class__(
            pd.DataFrame(
                clean(
                    self.values,
                    detrend=detrend,
                    standardize=standardize,
                    confounds=confounds,
                    low_pass=low_pass,
                    high_pass=high_pass,
                    ensure_finite=ensure_finite,
                    t_r=1.0 / np.float64(self.sampling_freq),
                    runs=sessions,
                    *args,
                    **kwargs,
                ),
                columns=self.columns,
            ),
            sampling_freq=self.sampling_freq,
            features=self.features,
            sessions=self.sessions,
        )

    def decompose(self, algorithm="pca", axis=1, n_components=None, *args, **kwargs):
        """Decompose Fex instance

        Args:
            algorithm: (str) Algorithm to perform decomposition types=['pca','ica','nnmf','fa']
            axis: dimension to decompose [0,1]
            n_components: (int) number of components. If None then retain as many as possible.

        Returns:
            output: a dictionary of decomposition parameters
        """
        out = {}
        out["decomposition_object"] = set_decomposition_algorithm(
            algorithm=algorithm, n_components=n_components, *args, **kwargs
        )
        com_names = ["c%s" % str(x + 1) for x in range(n_components)]
        if axis == 0:
            out["decomposition_object"].fit(self.T)
            out["components"] = self.__class__(
                pd.DataFrame(
                    out["decomposition_object"].transform(self.T),
                    index=self.columns,
                    columns=com_names,
                ),
                sampling_freq=None,
            )
            out["weights"] = self.__class__(
                pd.DataFrame(
                    out["decomposition_object"].components_.T,
                    index=self.index,
                    columns=com_names,
                ),
                sampling_freq=self.sampling_freq,
                features=self.features,
                sessions=self.sessions,
            )
        if axis == 1:
            out["decomposition_object"].fit(self)
            out["components"] = self.__class__(
                pd.DataFrame(
                    out["decomposition_object"].transform(self), columns=com_names
                ),
                sampling_freq=self.sampling_freq,
                features=self.features,
                sessions=self.sessions,
            )
            out["weights"] = self.__class__(
                pd.DataFrame(
                    out["decomposition_object"].components_,
                    index=com_names,
                    columns=self.columns,
                ),
                sampling_freq=None,
            ).T
        return out

    def extract_mean(self, ignore_sessions=False):
        """Extract mean of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            Fex: mean values for each feature
        """

        prefix = "mean"
        if self.sessions is None or ignore_sessions:
            feats = pd.DataFrame(self.mean(numeric_only=True)).T
        else:
            feats = []
            for k, v in self.itersessions():
                feats.append(pd.Series(v.mean(numeric_only=True), name=k))
            feats = pd.concat(feats, axis=1).T
        feats = self.__class__(feats)
        feats.columns = f"{prefix}_" + feats.columns
        feats = feats.__finalize__(self)
        if ignore_sessions is False:
            feats.sessions = np.unique(self.sessions)
        feats._update_extracted_colnames(prefix)
        return feats

    def extract_std(self, ignore_sessions=False):
        """Extract std of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            Fex: mean values for each feature
        """

        prefix = "std"
        if self.sessions is None or ignore_sessions:
            feats = pd.DataFrame(self.std(numeric_only=True)).T
        else:
            feats = []
            for k, v in self.itersessions():
                feats.append(pd.Series(v.std(numeric_only=True), name=k))
            feats = pd.concat(feats, axis=1).T
        feats = self.__class__(feats)
        feats.columns = f"{prefix}_" + feats.columns
        feats = feats.__finalize__(self)
        if ignore_sessions is False:
            feats.sessions = np.unique(self.sessions)
        feats._update_extracted_colnames(prefix)
        return feats

    def extract_sem(self, ignore_sessions=False):
        """Extract std of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            Fex: mean values for each feature
        """

        prefix = "sem"
        if self.sessions is None or ignore_sessions:
            feats = pd.DataFrame(self.sem(numeric_only=True)).T
        else:
            feats = []
            for k, v in self.itersessions():
                feats.append(pd.Series(v.sem(numeric_only=True), name=k))
            feats = pd.concat(feats, axis=1).T
        feats = self.__class__(feats)
        feats.columns = f"{prefix}_" + feats.columns
        feats = feats.__finalize__(self)
        if ignore_sessions is False:
            feats.sessions = np.unique(self.sessions)
        feats._update_extracted_colnames(prefix)
        return feats

    def extract_min(self, ignore_sessions=False):
        """Extract minimum of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            Fex: (Fex) minimum values for each feature
        """

        prefix = "min"
        if self.sessions is None or ignore_sessions:
            feats = pd.DataFrame(self.min(numeric_only=True)).T
        else:
            feats = []
            for k, v in self.itersessions():
                feats.append(pd.Series(v.min(numeric_only=True), name=k))
            feats = pd.concat(feats, axis=1).T
        feats = self.__class__(feats)
        feats.columns = f"{prefix}_" + feats.columns
        feats = feats.__finalize__(self)
        if ignore_sessions is False:
            feats.sessions = np.unique(self.sessions)
        feats._update_extracted_colnames(prefix)
        return feats

    def extract_max(self, ignore_sessions=False):
        """Extract maximum of each feature

        Args:
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            fex: (Fex) maximum values for each feature
        """
        prefix = "max"
        if self.sessions is None or ignore_sessions:
            feats = pd.DataFrame(self.max(numeric_only=True)).T
        else:
            feats = []
            for k, v in self.itersessions():
                feats.append(pd.Series(v.max(numeric_only=True), name=k))
            feats = pd.concat(feats, axis=1).T
        feats = self.__class__(feats)
        feats.columns = f"{prefix}_" + feats.columns
        feats = feats.__finalize__(self)
        if ignore_sessions is False:
            feats.sessions = np.unique(self.sessions)
        feats._update_extracted_colnames(prefix)
        return feats

    def extract_summary(
        self,
        mean=True,
        std=True,
        sem=True,
        max=True,
        min=True,
        ignore_sessions=False,
        *args,
        **kwargs,
    ):
        """Extract summary of multiple features

        Args:
            mean: (bool) extract mean of features
            std: (bool) extract std of features
            sem: (bool) extract sem of features
            max: (bool) extract max of features
            min: (bool) extract min of features
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            fex: (Fex)
        """

        if mean is max is min is False:
            raise ValueError("At least one of min, max, mean must be True")

        out = self.__class__().__finalize__(self)
        if ignore_sessions is False:
            out.sessions = np.unique(self.sessions)

        col_updates = []

        if mean:
            new = self.extract_mean(ignore_sessions=ignore_sessions, *args, **kwargs)
            out = out.append(new, axis=1)
            col_updates.append("mean")
        if sem:
            new = self.extract_sem(ignore_sessions=ignore_sessions, *args, **kwargs)
            out = out.append(new, axis=1)
            col_updates.append("sem")
        if std:
            new = self.extract_std(ignore_sessions=ignore_sessions, *args, **kwargs)
            out = out.append(new, axis=1)
            col_updates.append("std")
        if max:
            new = self.extract_max(ignore_sessions=ignore_sessions, *args, **kwargs)
            out = out.append(new, axis=1)
            col_updates.append("max")
        if min:
            new = self.extract_min(ignore_sessions=ignore_sessions, *args, **kwargs)
            out = out.append(new, axis=1)
            col_updates.append("min")

        out._update_extracted_colnames(mode="reset")
        out._update_extracted_colnames(col_updates)

        return out

    def extract_wavelet(self, freq, num_cyc=3, mode="complex", ignore_sessions=False):
        """Perform feature extraction by convolving with a complex morlet wavelet

        Args:
            freq: (float) frequency to extract
            num_cyc: (float) number of cycles for wavelet
            mode: (str) feature to extract, e.g., 'complex','filtered','phase','magnitude','power']
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            convolved: (Fex instance)

        """
        wav = wavelet(freq, sampling_freq=self.sampling_freq, num_cyc=num_cyc)
        if self.sessions is None or ignore_sessions:
            convolved = self.__class__(
                pd.DataFrame(
                    {x: convolve(y, wav, mode="same") for x, y in self.iteritems()}
                ),
                sampling_freq=self.sampling_freq,
            )
        else:
            convolved = self.__class__(sampling_freq=self.sampling_freq)
            for k, v in self.itersessions():
                session = self.__class__(
                    pd.DataFrame(
                        {x: convolve(y, wav, mode="same") for x, y in v.items()}
                    ),
                    sampling_freq=self.sampling_freq,
                )
                convolved = convolved.append(session, session_id=k)
        if mode == "complex":
            convolved = convolved
        elif mode == "filtered":
            convolved = np.real(convolved)
        elif mode == "phase":
            convolved = np.angle(convolved)
        elif mode == "magnitude":
            convolved = np.abs(convolved)
        elif mode == "power":
            convolved = np.abs(convolved) ** 2
        else:
            raise ValueError(
                "Mode must be ['complex','filtered','phase'," "'magnitude','power']"
            )
        convolved = self.__class__(
            convolved,
            sampling_freq=self.sampling_freq,
            features=self.features,
            sessions=self.sessions,
        )
        convolved.columns = (
            "f" + "%s" % round(freq, 2) + "_" + mode + "_" + self.columns
        )
        return convolved

    def extract_multi_wavelet(
        self, min_freq=0.06, max_freq=0.66, bank=8, *args, **kwargs
    ):
        """Convolve with a bank of morlet wavelets.

        Wavelets are equally spaced from min to max frequency. See extract_wavelet for more information and options.

        Args:
            min_freq: (float) minimum frequency to extract
            max_freq: (float) maximum frequency to extract
            bank: (int) size of wavelet bank
            num_cyc: (float) number of cycles for wavelet
            mode: (str) feature to extract, e.g., ['complex','filtered','phase','magnitude','power']
            ignore_sessions: (bool) ignore sessions or extract separately by sessions if available.

        Returns:
            convolved: (Fex instance)
        """
        out = []
        for f in np.geomspace(min_freq, max_freq, bank):
            out.append(self.extract_wavelet(f, *args, **kwargs))
        return self.__class__(
            pd.concat(out, axis=1),
            sampling_freq=self.sampling_freq,
            features=self.features,
            sessions=self.sessions,
        )

    def extract_boft(self, min_freq=0.06, max_freq=0.66, bank=8, *args, **kwargs):
        """Extract Bag of Temporal features

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
        freqs = np.geomspace(min_freq, max_freq, bank)
        wavs, hzs = [], []
        for i, f in enumerate(freqs):
            wav = np.real(wavelet(f, sampling_freq=target_hz))
            wavs.append(wav)
            hzs.append(str(np.round(freqs[i], 2)))
        wavs = np.array(wavs)[::-1]
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
            assert _d.isnull().sum().any() == 0, "Data contains NaNs. Cannot convolve. "
            for iw, cm in enumerate(wavs):
                convolved = np.apply_along_axis(
                    lambda m: np.convolve(m, cm, mode="full"), axis=1, arr=_d.values
                )
                # Extract bin features.
                out = pd.DataFrame(convolved.T).apply(calc_hist_auc, args=(None))
                # 6 bins hardcoded from calc_hist_auc
                colnames = [
                    "pos" + str(i) + "_hz_" + hzs[iw] + "_" + feat for i in range(6)
                ]
                colnames.extend(
                    ["neg" + str(i) + "_hz_" + hzs[iw] + "_" + feat for i in range(6)]
                )
                out = out.T
                out.columns = colnames
                feats = pd.concat([feats, out], axis=1)
        return self.__class__(
            feats, sampling_freq=self.sampling_freq, features=self.features
        )

    def _prepare_plot_aus(self, row, muscles, gaze):
        """
        Plot one or more faces based on their AU representation. This method is just a
        convenient wrapper for feat.plotting.plot_face. See that function for additional
        plotting args and kwargs.

        Args:
            force_separate_plot_per_detection (bool, optional): Whether to create a new
            figure for each detected face or plot to the same figure for multiple
            detections. Useful when you're know you're plotting multiple detections of a
           *single* face from multiple video frames. Default False

        """

        if self.au_model in ["svm", "xgb"]:
            au_lookup = "pyfeat"
            model = None
            feats = AU_LANDMARK_MAP["Feat"]
        else:
            au_lookup = self.au_model
            try:
                model = load_viz_model(f"{au_lookup}_aus_to_landmarks")
            except ValueError as _:
                raise NotImplementedError(
                    f"The AU model used for detection '{self.au_model}' has no corresponding AU visualization model. To fallback to plotting detections with facial landmarks, set faces='landmarks' in your call to .plot_detections. Otherwise, you can either use one of Py-Feat's custom AU detectors ('svm' or 'xgb') or train your own visualization model by following the tutorial at:\n\nhttps://py-feat.org/extra_tutorials/trainAUvisModel.html"
                )

            feats = AU_LANDMARK_MAP[au_lookup]

        rrow = row.copy()
        au = rrow[feats].to_numpy().squeeze()

        gaze = None if isinstance(gaze, bool) else gaze
        return au, gaze, muscles, model

    def plot_singleframe_detections(
        self,
        image_opacity=0.9,
        facebox_color="cyan",
        facebox_width=3,
        pose_width=2,
        landmark_color="white",
        landmark_width=2,
        emotions_position="right",
        emotions_opacity=0.9,
        emotions_color="white",
        emotions_size=12,
        au_heatmap_resolution=1000,
        au_opacity=0.9,
        au_cmap="Blues",
        *args,
        **kwargs,
    ):
        """
        Function to generate interactive plotly figure to interactively visualize py-feat detectors on a single image frame.

        Args:
            image_opacity (float): opacity of image overlay (default=.9)
            emotions_position (str): position around facebox to plot emotion annotations. default='right'
            emotions_opacity (float): opacity of emotion annotation text (default=.9)
            emotions_color (str): color of emotion annotation text (default='pink')
            emotions_size (int): size of emotion annotations (default=14)
            frame_duration (int): duration in milliseconds to play each frame if plotting multiple frames (default=1000)
            facebox_color (str): color of facebox bounding box (default="cyan")
            facebox_width (int): line width of facebox bounding box (default=3)
            pose_width (int): line width of pose rotation plot (default=2)
            landmark_color (str): color of landmark detectors (default="white")
            landmark_width (int): line width of landmark detectors (default=2)
            au_cmap (str): colormap to use for AU heatmap (default='Blues')
            au_heatmap_resolution (int): resolution of heatmap values (default=1000)
            au_opacity (float): opacity of AU heatmaps (default=0.9)

        Returns:
            a plotly figure instance
        """

        n_frames = len(self["frame"].unique())

        if n_frames > 1:
            raise ValueError(
                "This function can only plot a single frame. Try using plot_multipleframe_detections() instead."
            )

        # Initialize Image
        frame_id = self["frame"].unique()[0]
        frame_fex = self.query("frame==@frame_id")
        frame_img = load_pil_img(frame_fex["input"].unique()[0], frame_id)
        img_width = frame_img.width
        img_height = frame_img.height

        # Create figure
        fig = go.Figure()

        # Add invisible scatter trace.
        # This trace is added to help the autoresize logic work.
        fig.add_trace(
            go.Scatter(
                x=[0, img_width],
                y=[0, img_height],
                mode="markers",
                marker_opacity=0,
            )
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width,
                y=img_height,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=image_opacity,
                layer="below",
                sizing="stretch",
                source=frame_img,
            )
        )

        # Add Face Bounding Box
        faceboxes_path = [
            dict(
                type="rect",
                x0=row["FaceRectX"],
                y0=img_height - row["FaceRectY"],
                x1=row["FaceRectX"] + row["FaceRectWidth"],
                y1=img_height - row["FaceRectY"] - row["FaceRectHeight"],
                line=dict(color=facebox_color, width=facebox_width),
            )
            for i, row in frame_fex.iterrows()
        ]

        # Add Landmarks
        landmarks_path = [
            draw_plotly_landmark(
                row,
                img_height,
                fig,
                line_color=landmark_color,
                line_width=landmark_width,
            )
            for i, row in frame_fex.iterrows()
        ]

        # Add Pose
        poses_path = flatten_list(
            [
                draw_plotly_pose(row, img_height, fig, line_width=pose_width)
                for i, row in frame_fex.iterrows()
            ]
        )

        # Add Emotions
        emotions_annotations = []
        for i, row in frame_fex.iterrows():
            emotion_dict = (
                row[
                    [
                        "anger",
                        "disgust",
                        "fear",
                        "happiness",
                        "sadness",
                        "surprise",
                        "neutral",
                    ]
                ]
                .sort_values(ascending=False)
                .to_dict()
            )

            x_position, y_position, align, valign = emotion_annotation_position(
                row,
                img_height,
                img_width,
                emotions_size=emotions_size,
                emotions_position=emotions_position,
            )

            emotion_text = ""
            for emotion in emotion_dict:
                emotion_text += f"{emotion}: <i>{emotion_dict[emotion]:.2f}</i><br>"

            emotions_annotations.append(
                dict(
                    text=emotion_text,
                    x=x_position,
                    y=y_position,
                    opacity=emotions_opacity,
                    showarrow=False,
                    align=align,
                    valign=valign,
                    bgcolor="black",
                    font=dict(color=emotions_color, size=emotions_size),
                )
            )

        # Add AU Heatmaps
        aus_path = flatten_list(
            [
                draw_plotly_au(
                    row,
                    img_height,
                    fig,
                    cmap=au_cmap,
                    au_opacity=au_opacity,
                    heatmap_resolution=au_heatmap_resolution,
                )
                for i, row in frame_fex.iterrows()
            ]
        )

        # Configure other layout
        fig.update_layout(
            width=img_width,
            height=img_height,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        # Configure axes
        fig.update_xaxes(visible=False, range=[0, img_width])

        fig.update_yaxes(
            visible=False,
            range=[0, img_height],
            scaleanchor="x",  # the scaleanchor attribute ensures that the aspect ratio stays constant
        )

        # Add a button to the figure
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                method="relayout",
                                label="Bounding Box",
                                args=["shapes", faceboxes_path],
                                args2=["shapes", []],
                            ),
                            dict(
                                method="relayout",
                                label="Landmarks",
                                args=["shapes", landmarks_path],
                                args2=["shapes", []],
                            ),
                            dict(
                                method="relayout",
                                label="Poses",
                                args=["shapes", poses_path],
                                args2=["shapes", []],
                            ),
                            dict(
                                method="relayout",
                                label="Emotion",
                                args=["annotations", emotions_annotations],
                                args2=["annotations", []],
                            ),
                            dict(
                                method="relayout",
                                label="AU",
                                args=["shapes", aus_path],
                                args2=["shapes", []],
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.12,
                    yanchor="top",
                )
            ]
        )

        # Add annotation
        fig.update_layout(
            annotations=[
                dict(
                    text="Detector:",
                    showarrow=False,
                    x=0,
                    y=1.09,
                    yref="paper",
                    align="left",
                )
            ]
        )

        # fig.show()
        return fig

    def plot_multipleframes_detections(
        self,
        faceboxes=True,
        landmarks=False,
        aus=True,
        poses=False,
        emotions=False,
        emotions_position="right",
        emotions_opacity=1.0,
        emotions_color="white",
        emotions_size=14,
        frame_duration=1000,
        facebox_color="cyan",
        facebox_width=3,
        pose_width=2,
        landmark_color="white",
        landmark_width=2,
        au_heatmap_resolution=1000,
        au_opacity=0.9,
        au_cmap="Blues",
        *args,
        **kwargs,
    ):
        """
        Function to generate interactive plotly figure to visualize py-feat detectors on a series of image frames (e.g., videos).

        Args:
            faceboxes (bool): will include faceboxes when plotting detector output for multiple frames.
            landmarks (bool): will include face landmarks when plotting detector output for multiple frames.
            poses (bool): will include 3 axis line plot indicating x,y,z rotation information when plotting detector output for multiple frames.
            aus (bool): will include action unit heatmaps when plotting detector output for multiple frames.
            emotions (bool): will add text annotations indicating probability of discrete emotion when plotting detector output for multiple frames.
            image_opacity (float): opacity of image overlay (default=.9)
            emotions_position (str): position around facebox to plot emotion annotations. default='right'
            emotions_opacity (float): opacity of emotion annotation text (default=1.)
            emotions_color (str): color of emotion annotation text (default='white')
            emotions_size (int): size of emotion annotations (default=14)
            frame_duration (int): duration in milliseconds to play each frame if plotting multiple frames (default=1000)
            facebox_color (str): color of facebox bounding box (default="cyan")
            facebox_width (int): line width of facebox bounding box (default=3)
            pose_width (int): line width of pose rotation plot (default=2)
            landmark_color (str): color of landmark detectors (default="white")
            landmark_width (int): line width of landmark detectors (default=2)
            au_cmap (str): colormap to use for AU heatmap (default='Blues')
            au_heatmap_resolution (int): resolution of heatmap values (default=1000)
            au_opacity (float): opacity of AU heatmaps (default=0.9)

        Returns:
            a plotly figure instance
        """
        n_frames = len(self["frame"].unique())

        if n_frames <= 1:
            raise ValueError(
                "This function plots multiple frames. Try using plot_singleframe_detections() instead."
            )

        # Initialize Image
        frame_id = 0
        frame_fex = self.query("frame==@frame_id")
        frame_img = load_pil_img(frame_fex["input"].unique()[0], frame_id)
        img_width = frame_img.width
        img_height = frame_img.height

        # Initialize Figure
        fig = go.Figure(
            go.Scatter(
                x=[0, img_width], y=[0, img_height], mode="markers", marker_opacity=0
            )
        )
        fig.update_layout(
            xaxis_visible=False, yaxis_visible=False, width=img_width, height=img_height
        )

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }

        # Create Frames
        frames = []
        for frame_id in self["frame"].unique():
            # Load new image
            frame_fex = self.query("frame==@frame_id")
            frame_img = load_pil_img(frame_fex["input"].unique()[0], frame_id)
            img_width = frame_img.width
            img_height = frame_img.height

            # Add detector paths
            shapes = []
            if faceboxes:
                shapes += [
                    dict(
                        type="rect",
                        x0=row["FaceRectX"],
                        y0=img_height - row["FaceRectY"],
                        x1=row["FaceRectX"] + row["FaceRectWidth"],
                        y1=img_height - row["FaceRectY"] - row["FaceRectHeight"],
                        line=dict(color=facebox_color, width=facebox_width),
                    )
                    for i, row in frame_fex.iterrows()
                ]

            if poses:
                shapes += flatten_list(
                    [
                        draw_plotly_pose(row, img_height, fig, line_width=pose_width)
                        for i, row in frame_fex.iterrows()
                    ]
                )

            if landmarks:
                shapes += [
                    draw_plotly_landmark(
                        row,
                        img_height,
                        fig,
                        line_color=landmark_color,
                        line_width=landmark_width,
                    )
                    for i, row in frame_fex.iterrows()
                ]

            if aus:
                shapes += flatten_list(
                    [
                        draw_plotly_au(
                            row,
                            img_height,
                            fig,
                            cmap=au_cmap,
                            au_opacity=au_opacity,
                            heatmap_resolution=au_heatmap_resolution,
                        )
                        for i, row in frame_fex.iterrows()
                    ]
                )

            # Add emotion annotations
            emotions_annotations = []
            if emotions:
                for i, row in frame_fex.iterrows():
                    emotion_dict = (
                        row[
                            [
                                "anger",
                                "disgust",
                                "fear",
                                "happiness",
                                "sadness",
                                "surprise",
                                "neutral",
                            ]
                        ]
                        .sort_values(ascending=False)
                        .to_dict()
                    )

                    x_position, y_position, align, valign = emotion_annotation_position(
                        row,
                        img_height,
                        img_width,
                        emotions_size=emotions_size,
                        emotions_position=emotions_position,
                    )

                    emotion_text = ""
                    for emotion in emotion_dict:
                        emotion_text += (
                            f"{emotion}: <i>{emotion_dict[emotion]:.2f}</i><br>"
                        )

                    emotions_annotations.append(
                        dict(
                            text=emotion_text,
                            x=x_position,
                            y=y_position,
                            opacity=emotions_opacity,
                            showarrow=False,
                            align=align,
                            valign=valign,
                            bgcolor="black",
                            font=dict(color=emotions_color, size=emotions_size),
                        )
                    )

            sliders_dict["steps"].append(
                {
                    "args": [
                        [frame_id],
                        {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": frame_duration},
                        },
                    ],
                    "label": str(frame_id),
                    "method": "animate",
                }
            )
            frame = go.Frame(
                data=[],
                layout=dict(
                    images=[
                        dict(
                            source=frame_img,
                            opacity=0.9,
                            layer="below",
                            sizing="stretch",
                            x=0,
                            sizex=img_width,
                            y=img_height,
                            sizey=img_height,
                            xref="x",
                            yref="y",
                        )
                    ],
                    xaxis_visible=False,
                    yaxis_visible=False,
                    width=img_width,
                    height=img_height,
                    shapes=shapes,
                    annotations=emotions_annotations,
                    updatemenus=[
                        dict(
                            buttons=[
                                dict(
                                    args=[
                                        None,
                                        {
                                            "frame": {
                                                "duration": frame_duration,
                                                "redraw": True,
                                            },
                                            "fromcurrent": True,
                                        },
                                    ],
                                    label="Play",
                                    method="animate",
                                ),
                                dict(
                                    args=[
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    label="Pause",
                                    method="animate",
                                ),
                            ],
                            direction="left",
                            pad={"r": 10, "t": 87},
                            showactive=True,
                            type="buttons",
                            x=0.1,
                            xanchor="right",
                            y=0,
                            yanchor="top",
                        )
                    ],
                ),
                name=str(frame_id),
            )

            frames.append(frame)

        fig.update(frames=frames)
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                            label="Play",
                            method="animate",
                        ),
                        dict(
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            label="Pause",
                            method="animate",
                        ),
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    type="buttons",
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top",
                )
            ],
            sliders=[sliders_dict],
        )
        # fig.show()

        return fig

    def iplot_detections(
        self,
        faceboxes=True,
        landmarks=True,
        aus=False,
        poses=False,
        emotions=False,
        emotions_position="right",
        emotions_opacity=1.0,
        emotions_color="white",
        emotions_size=14,
        frame_duration=1000,
        facebox_color="cyan",
        facebox_width=3,
        pose_width=2,
        landmark_color="white",
        landmark_width=2,
        au_cmap="Blues",
        au_heatmap_resolution=1000,
        au_opacity=0.9,
        *args,
        **kwargs,
    ):
        """Plot Py-FEAT detection results using plotly backend. There are currently two different types of plots implemented. For single Frames, uses plot_singleframe_detections() to create an interactive plot where different detector outputs can be toggled on or off.  For multiple frames, uses plot_multipleframes_detections() to create a plotly animation to scroll through multiple frames. However, we currently are unable to interactively toggle on and off the detectors, so the detector output must be prespecified when generating the plot.

        Args:
            faceboxes (bool): will include faceboxes when plotting detector output for multiple frames.
            landmarks (bool): will include face landmarks when plotting detector output for multiple frames.
            poses (bool): will include 3 axis line plot indicating x,y,z rotation information when plotting detector output for multiple frames.
            aus (bool): will include action unit heatmaps when plotting detector output for multiple frames.
            emotions (bool): will add text annotations indicating probability of discrete emotion when plotting detector output for multiple frames.
            emotions_position (str): position around facebox to plot emotion annotations. default='right'
            emotions_opacity (float): opacity of emotion annotation text (default=1.)
            emotions_color (str): color of emotion annotation text (default='white')
            emotions_size (int): size of emotion annotations (default=14)
            frame_duration (int): duration in milliseconds to play each frame if plotting multiple frames (default=1000)
            facebox_color (str): color of facebox bounding box (default="cyan")
            facebox_width (int): line width of facebox bounding box (default=3)
            pose_width (int): line width of pose rotation plot (default=2)
            landmark_color (str): color of landmark detectors (default="white")
            landmark_width (int): line width of landmark detectors (default=2)
            au_cmap (str): colormap to use for AU heatmap (default='Blues')
            au_heatmap_resolution (int): resolution of heatmap values (default=1000)
            au_opacity (float): opacity of AU heatmaps (default=0.9)

        Returns:
            a plotly figure instance
        """

        n_frames = len(self["frame"].unique())

        if n_frames > 1:
            plot_single_frame = False
        else:
            plot_single_frame = True

        if plot_single_frame:
            fig = self.plot_singleframe_detections(
                facebox_color=facebox_color,
                facebox_width=facebox_width,
                pose_width=pose_width,
                landmark_color=landmark_color,
                landmark_width=landmark_width,
                emotions_position=emotions_position,
                emotions_opacity=emotions_opacity,
                emotions_color=emotions_color,
                emotions_size=emotions_size,
                au_cmap=au_cmap,
                au_heatmap_resolution=au_heatmap_resolution,
                au_opacity=au_opacity,
                *args,
                **kwargs,
            )

        else:
            fig = self.plot_multipleframes_detections(
                faceboxes=faceboxes,
                landmarks=landmarks,
                aus=aus,
                pose=poses,
                emotions=emotions,
                emotions_position=emotions_position,
                emotions_opacity=emotions_opacity,
                emotions_color=emotions_color,
                emotions_size=emotions_size,
                frame_duration=frame_duration,
                facebox_color=facebox_color,
                facebox_width=facebox_width,
                pose_width=pose_width,
                landmark_color=landmark_color,
                landmark_width=landmark_width,
                au_cmap=au_cmap,
                au_heatmap_resolution=au_heatmap_resolution,
                au_opacity=au_opacity,
                *args,
                **kwargs,
            )

        return fig

    def plot_detections(
        self,
        faces="landmarks",
        faceboxes=True,
        muscles=False,
        poses=False,
        gazes=False,
        add_titles=True,
        au_barplot=True,
        emotion_barplot=True,
        plot_original_image=True,
    ):
        """
        Plots detection results by Feat. Can control plotting of face, AU barplot and
        Emotion barplot. The faces kwarg controls whether facial landmarks are draw on
        top of input images or whether faces are visualized using Py-Feat's AU
        visualization model using detected AUs. If detection was performed on a video an
        faces = 'landmarks', only an outline of the face will be draw without loading
        the underlying vidoe frame to save memory.

        Args:
            faces (str, optional): 'landmarks' to draw detected landmarks or 'aus' to
            generate a face from AU detections using Py-Feat's AU landmark model.
            Defaults to 'landmarks'.
            faceboxes (bool, optional): Whether to draw the bounding box around detected
            faces. Only applies if faces='landmarks'. Defaults to True.
            muscles (bool, optional): Whether to draw muscles from AU activity. Only
            applies if faces='aus'. Defaults to False.
            poses (bool, optional): Whether to draw facial poses. Only applies if
            faces='landmarks'. Defaults to False.
            gazes (bool, optional): Whether to draw gaze vectors. Only applies if faces='aus'. Defaults to False.
            add_titles (bool, optional): Whether to add the file name as a title above
            the face. Defaults to True.
            au_barplot (bool, optional): Whether to include a subplot for au detections. Defaults to True.
            emotion_barplot (bool, optional): Whether to include a subplot for emotion detections. Defaults to True.

        Returns:
            list: list of matplotlib figures
        """

        # Plotting logic, eventually refactor me!:
        # Possible detections:
        # 1. Single image - single-face
        # 2. Single image - multi-face
        # 3. Multi image - single-face per image
        # 4. Multi image - multi-face per image
        # 5. Multi image - single and multi-face mix per image
        # 6. Video - single-face for all frames
        # 7. Video - multi-face for all frames
        # 8. Video - single and multi-face mix across frames

        sns.set_context("paper", font_scale=2.0)

        num_subplots = bool(faces) + au_barplot + emotion_barplot

        all_figs = []
        for _, frame in enumerate(self.frame.unique()):
            # Determine figure width based on how many subplots we have
            f = plt.figure(figsize=(5 * num_subplots, 7))
            spec = f.add_gridspec(ncols=num_subplots, nrows=1)
            col_count = 0
            plot_data = self.query("frame == @frame")
            face_ax, au_ax, emo_ax = None, None, None
            if faces is not False:
                face_ax = f.add_subplot(spec[0, col_count])
                col_count += 1
            if au_barplot:
                au_ax = f.add_subplot(spec[0, col_count])
                col_count += 1
            if emotion_barplot:
                emo_ax = f.add_subplot(spec[0, col_count])
                col_count += 1

            for _, row in plot_data.iterrows():
                # DRAW LANDMARKS ON IMAGE OR AU FACE
                if face_ax is not None:
                    facebox = row[self.facebox_columns].values

                    if not faces == "aus" and plot_original_image:
                        file_extension = os.path.basename(row["input"]).split(".")[-1]
                        if file_extension.lower() in [
                            "jpg",
                            "jpeg",
                            "png",
                            "bmp",
                            "tiff",
                            "pdf",
                        ]:
                            img = read_image(row["input"])
                        else:
                            # Ignore UserWarning: The pts_unit 'pts' gives wrong results. Please use
                            # pts_unit 'sec'. See why it's ok in this issue:
                            # https://github.com/pytorch/vision/issues/1931
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                video, audio, info = read_video(
                                    row["input"], output_format="TCHW"
                                )
                            img = video[row["frame"], :, :]
                        color = "w"
                        face_ax.imshow(img.permute([1, 2, 0]))
                    else:
                        color = "k"  # drawing lineface but not on photo

                    if faceboxes:
                        rect = Rectangle(
                            (facebox[0], facebox[1]),
                            facebox[2],
                            facebox[3],
                            linewidth=2,
                            edgecolor="cyan",
                            fill=False,
                        )
                        face_ax.add_patch(rect)

                    if poses:
                        face_ax = draw_facepose(
                            pose=row[self.facepose_columns].values,
                            facebox=facebox,
                            ax=face_ax,
                        )

                    if faces == "landmarks":
                        landmark = row[self.landmark_columns].values
                        currx = landmark[:68]
                        curry = landmark[68:]

                        # facelines
                        face_ax = draw_lineface(
                            currx, curry, ax=face_ax, color=color, linewidth=3
                        )
                        if not plot_original_image:
                            face_ax.invert_yaxis()

                    elif faces == "aus":
                        # Generate face from AU landmark model
                        if any(self.groupby("frame").size() > 1):
                            raise NotImplementedError(
                                "Plotting using AU landmark model is not currently supported for detections that contain multiple faces"
                            )
                        if muscles:
                            muscles = {"all": "heatmap"}
                        else:
                            muscles = {}
                        aus, gaze, muscles, model = self._prepare_plot_aus(
                            row, muscles=muscles, gaze=gazes
                        )
                        title = row["input"] if add_titles else None
                        face_ax = plot_face(
                            model=model,
                            au=aus,
                            gaze=gaze,
                            ax=face_ax,
                            muscles=muscles,
                            title=None,
                        )
                    else:
                        raise ValueError(
                            f"faces={type(faces)} is not currently supported try ['False','landmarks','aus']."
                        )

                    if add_titles:
                        _ = face_ax.set_title(
                            "\n".join(wrap(os.path.basename(row["input"]))),
                            loc="center",
                            wrap=True,
                            fontsize=14,
                        )

                    face_ax.axes.get_xaxis().set_visible(False)
                    face_ax.axes.get_yaxis().set_visible(False)

            # DRAW AU BARPLOT
            if au_ax is not None:
                _ = plot_data.aus.T.plot(kind="barh", ax=au_ax)
                _ = au_ax.invert_yaxis()
                _ = au_ax.legend().remove()
                _ = au_ax.set(xlim=[0, 1.1], title="Action Units")

            # DRAW EMOTION BARPLOT
            if emo_ax is not None:
                _ = plot_data.emotions.T.plot(kind="barh", ax=emo_ax)
                _ = emo_ax.invert_yaxis()
                _ = emo_ax.legend().remove()
                _ = emo_ax.set(xlim=[0, 1.1], title="Emotions")

            f.tight_layout()
            all_figs.append(f)
        return all_figs

    def compute_identities(self, threshold=0.8, inplace=False):
        """Compute Identities using face embeddings from identity detector using threshold"""
        if inplace:
            self["Identity"] = cluster_identities(
                self.identity_embeddings, threshold=threshold
            )
        else:
            out = self.copy()
            out["Identity"] = cluster_identities(
                out.identity_embeddings, threshold=threshold
            )
            return out

    # TODO: turn this into a property using a @property and @sessions.settr decorators
    # Tried it but was running into maximum recursion depth errors. Maybe some
    # interaction with how pandas sub-classing works?? - ejolly
    def update_sessions(self, new_sessions):
        """
        Returns a copy of the Fex dataframe with a new sessions attribute after
        validation. `new_sessions` should be a dictionary mapping old to new names or an iterable with the same number of rows as the Fex dataframe

        Args:
            new_sessions (dict, Iterable): map or list of new session names

        Returns:
            Fex: self
        """

        out = deepcopy(self)

        if isinstance(new_sessions, dict):
            if not isinstance(out.sessions, pd.Series):
                out.sessions = pd.Series(out.sessions)

            out.sessions = out.sessions.map(new_sessions)

        elif isinstance(new_sessions, Iterable):
            if len(new_sessions) != out.shape[0]:
                raise ValueError(
                    f"When new_sessions are not a dictionary then they must but an iterable with length == the number of rows of this Fex dataframe {out.shape[0]}, but they have length {len(new_sessions)}."
                )

            out.sessions = new_sessions

        else:
            raise TypeError(
                f"new_sessions must be either be a dictionary mapping between old and new session values or an iterable with the same number of rows as this Fex dataframe {out.shape[0]}, but was type: {type(new_sessions)}"
            )

        return out


class ImageDataset(Dataset):
    """Torch Image Dataset

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to
        output_size. If int, will set largest edge to output_size if target size is
        bigger, or smallest edge if target size is smaller to keep aspect ratio the
        same.
        preserve_aspect_ratio (bool): Output size is matched to preserve aspect ratio. Note that longest edge of output size is preserved, but actual output may differ from intended output_size.
        padding (bool): Transform image to exact output_size. If tuple, will preserve
        aspect ratio by adding padding. If int, will set both sides to the same size.

    Returns:
        Dataset: dataset of [batch, channels, height, width] that can be passed to DataLoader
    """

    def __init__(
        self, images, output_size=None, preserve_aspect_ratio=True, padding=False
    ):
        if isinstance(images, str):
            images = [images]
        self.images = images
        self.output_size = output_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.padding = padding

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Dimensions are [channels, height, width]
        try:
            img = read_image(self.images[idx])
        except Exception:
            img = Image.open(self.images[idx])
            img = transforms.PILToTensor()(img)

        # Drop alpha channel
        if img.shape[0] == 4:
            img = img[:3, ...]

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], dim=0)

        if self.output_size is not None:
            logging.info(
                f"ImageDataSet: RESCALING WARNING: from {img.shape} to output_size={self.output_size}"
            )
            transform = Compose(
                [
                    Rescale(
                        self.output_size,
                        preserve_aspect_ratio=self.preserve_aspect_ratio,
                        padding=self.padding,
                    )
                ]
            )
            transformed_img = transform(img)
            return {
                "Image": transformed_img["Image"],
                "Scale": transformed_img["Scale"],
                "Padding": transformed_img["Padding"],
                "FileName": self.images[idx],
            }

        else:
            return {
                "Image": img,
                "Scale": 1.0,
                "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
                "FileName": self.images[idx],
            }


def _inverse_face_transform(faces, batch_data):
    """Helper function to invert the Image Data batch transforms on the face bounding boxes

    Args:
        faces (list): list of lists from face_detector
        batch_data (dict): batch data from Image Data Class

    Returns:
        transformed list of lists
    """

    logging.info("inverting face transform...")

    out_frame = []
    for frame, left, top, scale in zip(
        faces,
        batch_data["Padding"]["Left"].detach().numpy(),
        batch_data["Padding"]["Top"].detach().numpy(),
        batch_data["Scale"].detach().numpy(),
    ):
        out_face = []
        for face in frame:
            out_face.append(
                list(
                    np.append(
                        (
                            np.array(
                                [
                                    face[0] - left,
                                    face[1] - top,
                                    face[2] - left,
                                    face[3] - top,
                                ]
                            )
                            / scale
                        ),
                        face[4],
                    )
                )
            )
        out_frame.append(out_face)
    return out_frame


class imageLoader_DISFAPlus(ImageDataset):
    """
    Loading images from DISFA dataset. Assuming that the user has just unzipped the downloaded DISFAPlus data
    """

    def __init__(
        self,
        data_dir="/Storage/Data/DISFAPlusDataset/",
        output_size=None,
        preserve_aspect_ratio=True,
        padding=False,
        sample=None,
    ):
        super().__init__(
            images=None,
            output_size=output_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            padding=padding,
        )

        # Load all dir info for DISFA+
        self.avail_AUs = [
            "AU1",
            "AU2",
            "AU4",
            "AU5",
            "AU6",
            "AU9",
            "AU12",
            "AU15",
            "AU17",
            "AU20",
            "AU25",
            "AU26",
        ]

        self.data_dir = data_dir
        self.sample = sample
        self.main_file = self._load_data()

        self.output_size = output_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.padding = padding

    def _load_data(self):
        print("data loading in progress")
        all_subjects = os.listdir(os.path.join(self.data_dir, "Labels"))
        if self.sample:
            all_subjects = np.random.choice(
                all_subjects, size=int(self.sample * len(all_subjects)), replace=False
            )

        sessions = [
            os.listdir(os.path.join(self.data_dir, "Labels", subj))
            for subj in all_subjects
        ]
        # all image directory
        dfs = []
        for i, subj in enumerate(all_subjects):
            for sess in sessions[i]:
                AU_f = []
                for au_added in self.avail_AUs:
                    AU_file = pd.read_csv(
                        os.path.join(self.data_dir, "Labels", subj, sess)
                        + "/"
                        + au_added
                        + ".txt",
                        skiprows=2,
                        header=None,
                        names=["intensity"],
                        index_col=0,
                        sep=r"\s{2,}",
                    )
                    AU_file.rename({"intensity": f"{au_added}"}, axis=1, inplace=True)
                    AU_f.append(AU_file)
                AU_pd = pd.concat(AU_f, axis=1)
                AU_pd = AU_pd.reset_index(level=0)
                AU_pd["session"] = sess
                AU_pd["subject"] = subj
                dfs.append(AU_pd)
        df = pd.concat(dfs, ignore_index=True)
        df["image_path"] = [
            os.path.join(self.data_dir, "Images", df["subject"][i], df["session"][i])
            + "/"
            + df["index"][i]
            for i in range(df.shape[0])
        ]
        return df

    def __len__(self):
        return self.main_file.shape[0]

    def __getitem__(self, idx):
        # Dimensions are [channels, height, width]
        img = read_image(self.main_file["image_path"].iloc[idx])
        label = self.main_file.loc[idx, self.avail_AUs].to_numpy().astype(np.int16)

        if self.output_size is not None:
            logging.info(
                f"imageLoader_DISFAPlus: RESCALING WARNING: from {img.shape} to output_size={self.output_size}"
            )
            transform = Compose(
                [
                    Rescale(
                        self.output_size,
                        preserve_aspect_ratio=self.preserve_aspect_ratio,
                        padding=self.padding,
                    )
                ]
            )
            transformed_img = transform(img)
            return {
                "Image": transformed_img["Image"],
                "label": torch.from_numpy(label),
                "Scale": transformed_img["Scale"],
                "Padding": transformed_img["Padding"],
                "FileName": self.main_file["image_path"][idx],
            }

        else:
            return {
                "Image": img,
                "label": torch.from_numpy(label),
                "Scale": 1.0,
                "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
            }


def _inverse_landmark_transform(landmarks, batch_data):
    """Helper function to invert the Image Data batch transforms on the facial landmarks

    Args:
        landmarks (list): list of lists from landmark_detector
        batch_data (dict): batch data from Image Data Class

    Returns:
        transformed list of lists
    """

    logging.info("inverting landmark transform...")

    out_frame = []
    for frame, left, top, scale in zip(
        landmarks,
        batch_data["Padding"]["Left"].detach().numpy(),
        batch_data["Padding"]["Top"].detach().numpy(),
        batch_data["Scale"].detach().numpy(),
    ):
        out_landmark = []
        for landmark in frame:
            out_landmark.append(
                (landmark - np.ones(landmark.shape) * [left, top]) / scale
            )
        out_frame.append(out_landmark)
    return out_frame

class TensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        # Return the number of samples in the dataset
        return self.tensor.size(0)

    def __getitem__(self, idx):
        # Return the sample at the given index
        return self.tensor[idx, ...]
    
class VideoDataset(Dataset):
    """Torch Video Dataset

    Args:
        skip_frames (int): number of frames to skip

    Returns:
        Dataset: dataset of [batch, channels, height, width] that can be passed to DataLoader
    """

    def __init__(self, video_file, skip_frames=None, output_size=None):
        self.file_name = video_file
        self.skip_frames = skip_frames
        self.output_size = output_size
        self.get_video_metadata(video_file)
        # This is the list of frame ids used to slice the video not video_frames
        self.video_frames = np.arange(
            0, self.metadata["num_frames"], 1 if skip_frames is None else skip_frames
        )

    def __len__(self):
        # Number of frames respective skip_frames
        return len(self.video_frames)

    def __getitem__(self, idx):
        # Get the frame data and frame number respective skip_frames
        frame_data, frame_idx = self.load_frame(idx)

        # Swap frame dims to match output of read_image: [time, channels, height, width]
        # Otherwise detectors face on tensor dimension mismatch
        frame_data = swapaxes(swapaxes(frame_data, 0, -1), 1, 2)

        # Rescale if needed like in ImageDataset
        if self.output_size is not None:
            logging.info(
                f"VideoDataset: RESCALING WARNING: from {self.metadata['shape']} to output_size={self.output_size}"
            )
            transform = Compose(
                [Rescale(self.output_size, preserve_aspect_ratio=True, padding=False)]
            )
            transformed_frame_data = transform(frame_data)

            return {
                "Image": transformed_frame_data["Image"],
                "Frame": frame_idx,
                "FileName": self.file_name,
                "Scale": transformed_frame_data["Scale"],
                "Padding": transformed_frame_data["Padding"],
            }
        else:
            return {
                "Image": frame_data,
                "Frame": frame_idx,
                "FileName": self.file_name,
                "Scale": 1.0,
                "Padding": {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0},
            }

    def get_video_metadata(self, video_file):
        container = av.open(video_file)
        stream = container.streams.video[0]
        fps = stream.average_rate
        height = stream.height
        width = stream.width
        num_frames = stream.frames
        container.close()
        self.metadata = {
            "fps": float(fps),
            "fps_frac": fps,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "shape": (height, width),
        }

    def load_frame(self, idx):
        """Load in a single frame from the video using a lazy generator"""

        # Get frame number respecting skip_frames
        frame_idx = int(self.video_frames[idx])

        # Use a py-av generator to load in just this frame
        container = av.open(self.file_name)
        stream = container.streams.video[0]
        frame = next(islice(container.decode(stream), frame_idx, None))
        frame_data = torch.from_numpy(frame.to_ndarray(format="rgb24"))
        container.close()

        return frame_data, frame_idx

    def calc_approx_frame_time(self, idx):
        """Calculate the approximate time of a frame in a video

        Args:
            frame_idx (int): frame number

        Returns:
            float: time in seconds
        """
        frame_time = idx / self.metadata["fps"]
        total_time = self.metadata["num_frames"] / self.metadata["fps"]
        time = total_time if idx >= self.metadata["num_frames"] else frame_time
        return self.convert_sec_to_min_sec(time)

    @staticmethod
    def convert_sec_to_min_sec(duration):
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes:02d}:{seconds:02d}"
