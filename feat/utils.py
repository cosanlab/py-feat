from __future__ import division
'''
    FEAT Utils Class
    ==========================================
    read_facet: read in iMotions-FACET formatted files
    read_affdex: read in iMotions-affdex formatted files
    read_affectiva: read in affectiva-api formatted files
    read_openface: read in openface formatted files

'''

__all__ = ['read_facet','read_affdex','read_affectiva','read_openface']
__author__ = ["Jin Hyun Cheong"]



import os
import numpy as np
import pandas as pd

def read_facet(facetfile, features=None):
    '''
    This function reads in an iMotions-FACET exported facial expression file.
    Args:
        features: If a list of column names are passed, those are returned. Otherwise, default returns the following features:
        ['Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                  'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                  'Neutral Evidence','Positive Evidence','Negative Evidence','AU1 Evidence','AU2 Evidence',
                  'AU4 Evidence','AU5 Evidence','AU6 Evidence','AU7 Evidence','AU9 Evidence','AU10 Evidence',
                  'AU12 Evidence','AU14 Evidence','AU15 Evidence','AU17 Evidence','AU18 Evidence','AU20 Evidence',
                  'AU23 Evidence','AU24 Evidence','AU25 Evidence','AU26 Evidence','AU28 Evidence','AU43 Evidence',
                  'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']

    Returns:
        dataframe of processed facial expressions

    '''

    d = pd.read_csv(facetfile, skiprows=5, sep='\t')

    # Check if features argument is passed and return only those features, else return basic emotion/AU features
    if isinstance(features,list):
        try:
            d = d[features]
        except:
            raise KeyError([features,'not in facetfile'])
    elif isinstance(features, type(None)):
        features = ['Joy Evidence','Anger Evidence','Surprise Evidence','Fear Evidence','Contempt Evidence',
                  'Disgust Evidence','Sadness Evidence','Confusion Evidence','Frustration Evidence',
                  'Neutral Evidence','Positive Evidence','Negative Evidence','AU1 Evidence','AU2 Evidence',
                  'AU4 Evidence','AU5 Evidence','AU6 Evidence','AU7 Evidence','AU9 Evidence','AU10 Evidence',
                  'AU12 Evidence','AU14 Evidence','AU15 Evidence','AU17 Evidence','AU18 Evidence','AU20 Evidence',
                  'AU23 Evidence','AU24 Evidence','AU25 Evidence','AU26 Evidence','AU28 Evidence','AU43 Evidence',
                  'Yaw Degrees', 'Pitch Degrees', 'Roll Degrees']
        d = d[features]
    return d
