from __future__ import division
'''
    FEAT Utils Class
    ==========================================
    read_facet: read in iMotions-FACET formatted files
    read_affdex: read in iMotions-affdex formatted files
    read_affectiva: read in affectiva-api formatted files
    read_openface: read in openface formatted files

'''

__all__ = ['get_resource_path','load_pickled_model','read_facet',
           'read_affdex','read_affectiva','read_openface', 'softmax',
           'registration','neutral','load_h5']
__author__ = ["Jin Hyun Cheong"]



import os, math, pywt, pickle, h5py
from sklearn.cross_decomposition.pls_ import PLSRegression
import numpy as np, pandas as pd
from scipy import signal
from scipy.integrate import simps
import feat

def get_resource_path():
    """ Get path to feat resource directory. """
    return os.path.join(feat.__path__[0], 'resources') # points to the package folder.
    # return os.path.join(os.path.dirname(__file__), 'resources')

def load_pickled_model(file_name=None):
    if file_name is None:
        file_name = os.path.join(get_resource_path(), 'pls_python27.pkl')
    try:
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file_name, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file_name, ':', e)
        raise
    return model

def load_h5(file_name='blue.h5'):
    try:
        hf = h5py.File(os.path.join(get_resource_path(), file_name), 'r')
        d1 = hf.get('coef')
        d2 = hf.get('x_mean')
        d3 = hf.get('y_mean')
        d4 = hf.get('x_std')
        model = PLSRegression(len(d1))
        model.coef_ = np.array(d1)
        model.x_mean_ = np.array(d2)
        model.y_mean_ = np.array(d3)
        model.x_std_ = np.array(d4)
        hf.close()
    except Exception as e:
        print('Unable to load data ', file_name, ':', e)
    return model

def read_facet(facetfile, features=None, raw=False):
    '''
    This function reads in an iMotions-FACET exported facial expression file.
    By default

    Args:
        facetfile: iMotions-FACET file. Files from iMotions 5, 6, and 7 have been tested and supported
        features: If a list of iMotion-FACET column names are passed, those are returned.
                Otherwise, default columns are returned in the following format:['Timestamp','FaceRectX','FaceRectY','FaceRectWidth','FaceRectHeight',
                'Joy','Anger','Surprise','Fear','Contempt', 'Disgust','Sadness','Confusion','Frustration',
                'Neutral','Positive','Negative','AU1','AU2', 'AU4','AU5','AU6','AU7','AU9','AU10',
                'AU12','AU14','AU15','AU17','AU18','AU20', 'AU23','AU24','AU25','AU26','AU28','AU43',
                'Yaw', 'Pitch', 'Roll'].
                Note that these column names are different from the original files which has ' Evidence', ' Degrees' appended to each column.
        raw (default=False): Set to True to return all columns without processing.
    Returns:
        dataframe of processed facial expressions
    '''
    # Check iMotions Version
    versionstr = ''
    try:
        with open(facetfile,'r') as f:
            studyname = f.readline().replace('\t','').replace('\n','')
            studydate = f.readline().replace('\t','').replace('\n','')
            versionstr = f.readline().replace('\t','').replace('\n','')
        versionnum = int(versionstr.split(' ')[-1].split('.')[0])
    except:
        raise TypeError("Cannot infer version of iMotions-FACET file. Check to make sure this is the raw iMotions-FACET file.")

    d = pd.read_csv(facetfile, skiprows=5, sep='\t')
    # Check if features argument is passed and return only those features, else return all columns
    if isinstance(features,list):
        try:
            d = d[features]
            if raw:
                return d
        except:
            raise KeyError([features,'not in facetfile'])
    elif isinstance(features, type(None)):
        if raw:
            return d
        else:
            fex_columns = [col.replace(' Evidence','').replace(' Degrees','') for col in d.columns if 'Evidence' in col or 'Degrees' in col]
            # Remove Intensity as this has been deprecated
            cols2drop = [col for col in d.columns if "Intensity" in col]
            d = d.drop(columns=cols2drop)
            d.columns = [col.replace(' Evidence','') for col in d.columns]
            d.columns = [col.replace(' Degrees','') for col in d.columns]
            d.columns = [col.replace(' ','') for col in d.columns]
            d._metadata = fex_columns
    return d

def read_openface(openfacefile, features=None):
    '''
    This function reads in an OpenFace exported facial expression file.
    Args:
        features: If a list of column names are passed, those are returned. Otherwise, default returns the following features:
        ['frame', 'timestamp', 'confidence', 'success', 'gaze_0_x',
       'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
       'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
       'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8',
       'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16',
       'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24',
       'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32',
       'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40',
       'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48',
       'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56',
       'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64',
       'x_65', 'x_66', 'x_67', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5',
       'y_6', 'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12', 'y_13', 'y_14',
       'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22',
       'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 'y_30',
       'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36', 'y_37', 'y_38',
       'y_39', 'y_40', 'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46',
       'y_47', 'y_48', 'y_49', 'y_50', 'y_51', 'y_52', 'y_53', 'y_54',
       'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 'y_61', 'y_62',
       'y_63', 'y_64', 'y_65', 'y_66', 'y_67', 'X_0', 'X_1', 'X_2', 'X_3',
       'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12',
       'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20',
       'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28',
       'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36',
       'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44',
       'X_45', 'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52',
       'X_53', 'X_54', 'X_55', 'X_56', 'X_57', 'X_58', 'X_59', 'X_60',
       'X_61', 'X_62', 'X_63', 'X_64', 'X_65', 'X_66', 'X_67', 'Y_0',
       'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9',
       'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16', 'Y_17',
       'Y_18', 'Y_19', 'Y_20', 'Y_21', 'Y_22', 'Y_23', 'Y_24', 'Y_25',
       'Y_26', 'Y_27', 'Y_28', 'Y_29', 'Y_30', 'Y_31', 'Y_32', 'Y_33',
       'Y_34', 'Y_35', 'Y_36', 'Y_37', 'Y_38', 'Y_39', 'Y_40', 'Y_41',
       'Y_42', 'Y_43', 'Y_44', 'Y_45', 'Y_46', 'Y_47', 'Y_48', 'Y_49',
       'Y_50', 'Y_51', 'Y_52', 'Y_53', 'Y_54', 'Y_55', 'Y_56', 'Y_57',
       'Y_58', 'Y_59', 'Y_60', 'Y_61', 'Y_62', 'Y_63', 'Y_64', 'Y_65',
       'Y_66', 'Y_67', 'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6',
       'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13', 'Z_14', 'Z_15',
       'Z_16', 'Z_17', 'Z_18', 'Z_19', 'Z_20', 'Z_21', 'Z_22', 'Z_23',
       'Z_24', 'Z_25', 'Z_26', 'Z_27', 'Z_28', 'Z_29', 'Z_30', 'Z_31',
       'Z_32', 'Z_33', 'Z_34', 'Z_35', 'Z_36', 'Z_37', 'Z_38', 'Z_39',
       'Z_40', 'Z_41', 'Z_42', 'Z_43', 'Z_44', 'Z_45', 'Z_46', 'Z_47',
       'Z_48', 'Z_49', 'Z_50', 'Z_51', 'Z_52', 'Z_53', 'Z_54', 'Z_55',
       'Z_56', 'Z_57', 'Z_58', 'Z_59', 'Z_60', 'Z_61', 'Z_62', 'Z_63',
       'Z_64', 'Z_65', 'Z_66', 'Z_67', 'p_scale', 'p_rx', 'p_ry', 'p_rz',
       'p_tx', 'p_ty', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6',
       'p_7', 'p_8', 'p_9', 'p_10', 'p_11', 'p_12', 'p_13', 'p_14', 'p_15',
       'p_16', 'p_17', 'p_18', 'p_19', 'p_20', 'p_21', 'p_22', 'p_23',
       'p_24', 'p_25', 'p_26', 'p_27', 'p_28', 'p_29', 'p_30', 'p_31',
       'p_32', 'p_33', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
       'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
       'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
       'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
       'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
       'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

    Returns:
        dataframe of processed facial expressions

    '''

    d = pd.read_csv(openfacefile, sep=',')
    d.columns = d.columns.str.strip(' ')

    # Check if features argument is passed and return only those features, else return basic emotion/AU features
    if isinstance(features,list):
        try:
            d = d[features]
        except:
            raise KeyError([features,'not in openfacefile'])
    elif isinstance(features, type(None)):
        features = ['frame', 'timestamp', 'confidence', 'success', 'gaze_0_x',
       'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
       'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
       'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8',
       'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16',
       'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24',
       'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32',
       'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40',
       'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48',
       'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56',
       'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64',
       'x_65', 'x_66', 'x_67', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5',
       'y_6', 'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12', 'y_13', 'y_14',
       'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22',
       'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 'y_30',
       'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36', 'y_37', 'y_38',
       'y_39', 'y_40', 'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46',
       'y_47', 'y_48', 'y_49', 'y_50', 'y_51', 'y_52', 'y_53', 'y_54',
       'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 'y_61', 'y_62',
       'y_63', 'y_64', 'y_65', 'y_66', 'y_67', 'X_0', 'X_1', 'X_2', 'X_3',
       'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12',
       'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20',
       'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28',
       'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36',
       'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44',
       'X_45', 'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52',
       'X_53', 'X_54', 'X_55', 'X_56', 'X_57', 'X_58', 'X_59', 'X_60',
       'X_61', 'X_62', 'X_63', 'X_64', 'X_65', 'X_66', 'X_67', 'Y_0',
       'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9',
       'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16', 'Y_17',
       'Y_18', 'Y_19', 'Y_20', 'Y_21', 'Y_22', 'Y_23', 'Y_24', 'Y_25',
       'Y_26', 'Y_27', 'Y_28', 'Y_29', 'Y_30', 'Y_31', 'Y_32', 'Y_33',
       'Y_34', 'Y_35', 'Y_36', 'Y_37', 'Y_38', 'Y_39', 'Y_40', 'Y_41',
       'Y_42', 'Y_43', 'Y_44', 'Y_45', 'Y_46', 'Y_47', 'Y_48', 'Y_49',
       'Y_50', 'Y_51', 'Y_52', 'Y_53', 'Y_54', 'Y_55', 'Y_56', 'Y_57',
       'Y_58', 'Y_59', 'Y_60', 'Y_61', 'Y_62', 'Y_63', 'Y_64', 'Y_65',
       'Y_66', 'Y_67', 'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6',
       'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13', 'Z_14', 'Z_15',
       'Z_16', 'Z_17', 'Z_18', 'Z_19', 'Z_20', 'Z_21', 'Z_22', 'Z_23',
       'Z_24', 'Z_25', 'Z_26', 'Z_27', 'Z_28', 'Z_29', 'Z_30', 'Z_31',
       'Z_32', 'Z_33', 'Z_34', 'Z_35', 'Z_36', 'Z_37', 'Z_38', 'Z_39',
       'Z_40', 'Z_41', 'Z_42', 'Z_43', 'Z_44', 'Z_45', 'Z_46', 'Z_47',
       'Z_48', 'Z_49', 'Z_50', 'Z_51', 'Z_52', 'Z_53', 'Z_54', 'Z_55',
       'Z_56', 'Z_57', 'Z_58', 'Z_59', 'Z_60', 'Z_61', 'Z_62', 'Z_63',
       'Z_64', 'Z_65', 'Z_66', 'Z_67', 'p_scale', 'p_rx', 'p_ry', 'p_rz',
       'p_tx', 'p_ty', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6',
       'p_7', 'p_8', 'p_9', 'p_10', 'p_11', 'p_12', 'p_13', 'p_14', 'p_15',
       'p_16', 'p_17', 'p_18', 'p_19', 'p_20', 'p_21', 'p_22', 'p_23',
       'p_24', 'p_25', 'p_26', 'p_27', 'p_28', 'p_29', 'p_30', 'p_31',
       'p_32', 'p_33', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
       'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
       'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
       'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
       'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
       'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
        try:
            d = d[features]
        except:
            pass
    return d

def read_affectiva(affectivafile,orig_cols = False):
    '''
    This function reads in affectiva file processed through
    the https://github.com/cosanlab/affectiva-api-app.
    Args:
        affectivafile: file to read
        orig_cols: If True, convert original colnames to FACS names
    '''
    d = pd.read_json(affectivafile, lines=True)
    rep_dict = { 'anger':'Anger','attention':'Attention','contempt':'Contempt','disgust':'Disgust','engagement':'Engagement',
            'fear':'Fear','joy':'Joy','sadness':'Sadness','smirk':'Smirk','surprise':'Surprise','valence':'Valence',
    'browFurrow': "AU04", "smile":'AU12',
       'browRaise':'AU02', 'cheekRaise':'AU06', 'chinRaise':'AU17', 'dimpler':'AU14',
        'eyeClosure':'AU43', 'eyeWiden':'AU05',
       'innerBrowRaise':'AU01', 'jawDrop': 'AU26',
      'lidTighten':'AU07', 'lipCornerDepressor':'AU15',
       'lipPress':'AU24', 'lipPucker':'AU18', 'lipStretch':'AU20', 'lipSuck':'AU28', 'mouthOpen':'AU25',
       'noseWrinkle':'AU9', 'upperLipRaise':'AU10'}
    if not orig_cols:
        new_cols = []
        for col in d.columns:
            try:
                new_cols.append(rep_dict[col])
            except:
                new_cols.append(col)
        d.columns = new_cols
    return d


def wavelet(freq, num_cyc=3, sampling_freq=30.):
    """ Create a complex Morlet wavelet by windowing a cosine function by a
        Gaussian. All formulae taken from Cohen, 2014 Chaps 12 + 13

        Args:
            freq: (float) desired frequence of wavelet
            num_cyc: (float) number of wavelet cycles/gaussian taper. Note that
                     smaller cycles give greater temporal precision and that larger
                     values give greater frequency precision; (default: 3)
            sampling_freq: (float) sampling frequency of original signal.

        Returns:
            wav: (ndarray) complex wavelet
    """
    dur = (1/freq)*num_cyc
    time = np.arange(-dur, dur, 1./sampling_freq)

    #Cosine component
    sin = np.exp(2 * np.pi * 1j * freq * time)

    #Gaussian component
    sd = num_cyc/(2 * np.pi * freq) #standard deviation
    gaus = np.exp(-time**2./(2. * sd**2.))

    return sin * gaus

def calc_hist_auc(vals, hist_range=None):
    """
    This function follows the bag of temporal feature analysis as described in
    Bartlett, M. S., Littlewort, G. C., Frank, M. G., & Lee, K. (2014).
    Automatic decoding of facial movements reveals deceptive pain expressions.
    Current Biology, 24(7), 738-743.
    The function receives convolved data, squares the values,
    finds 0 crossings to calculate the AUC(area under the curve)
    and generates a 6 exponentially-spaced-bin histogram for each data.

    Args:

    Returns:


    """
    #Square values
    vals = [elem**2 if elem > 0 else -1*elem**2 for elem in vals]
    #Get 0 crossings
    crossings = np.where(np.diff(np.sign(vals)))[0]
    pos,neg = [],[]
    for i in range(len(crossings)):
        if i == 0:
            cross = vals[:crossings[i]]
        elif i == len(crossings)-1:
            cross = vals[crossings[i]:]
        else:
            cross = vals[crossings[i]:crossings[i+1]]
        if cross:
            auc = simps(cross)
            if auc > 0:
                pos.append(auc)
            elif auc < 0:
                neg.append(np.abs(auc))
    if not hist_range:
        hist_range = np.logspace(0,5,7) # bartlett 10**0~ 10**5

    out = pd.Series(np.hstack([np.histogram(pos,hist_range)[0], np.histogram(neg,hist_range)[0]]))
    return out

def softmax(x):
    '''
    Softmax function to change log likelihood evidence values to probabilities.
    Use with Evidence values from FACET.

    Args:
        x: value to softmax
    '''
    return 1./(1+10.**-(x))

### Functions for face registration ###
neutral = np.array([[37.514994071403564, 118.99554304280198], [38.347467261268164, 135.93119298564565], [40.77550102890035, 152.83280452855092],
           [44.109285817364565, 169.1279402172728], [49.982831719005134, 184.53328583541997], [59.18894827224358, 198.01613609507382],
           [70.41509055106278, 209.2829929016551], [83.65962787515429, 217.8257797774197], [98.6747861407431, 220.00636721799012],
           [113.36502269321642, 217.35622273914575], [126.09720342342395, 208.61554139570768], [137.37278216681938, 197.26636201144768],
           [146.15109522110836, 183.95054534968338], [151.7203254679301, 168.70328047716666], [154.90171533762026, 152.54959546525106],
           [157.01705755745184, 136.0791940145902], [157.81240022435486, 119.28714731581948], [45.87342275811805, 109.05187535227455],
           [53.83702202368147, 101.43275042887998], [65.61231530318975, 99.44649503101734], [77.49003981781006, 101.34627038289048],
           [88.31833069100318, 105.66229287035226], [108.80512997829634, 105.18583248508406], [120.180518838434, 100.84850879934683],
           [131.6712265255873, 99.22426247038426], [142.8040873694427, 101.39810664193074], [150.0927107560624, 108.74640334130906],
           [98.93955016899183, 117.16643104438056], [99.01139789533919, 128.44882090731443], [99.09059391932496, 139.71335180079416],
           [99.22411612922204, 151.32196734885628], [85.97238779261347, 158.19140086045783], [92.2064468619444, 160.61659751751895],
           [98.67862473703293, 162.56437315387998], [105.26853264390792, 160.62509055875418], [111.14227856687825, 158.32687793454852],
           [59.22833204018989, 118.63189570941351], [66.08746862218415, 114.39263501569359], [74.66886627073309, 114.59919005618073],
           [81.80683310969295, 120.00819188630955], [74.34426159695313, 121.7055175900537], [65.7237769427475, 121.82223252349223],
           [114.7522889881524, 119.90654628204749], [122.29832379941683, 114.26349216485505], [130.61954432603773, 114.38399042631573],
           [137.03708638863128, 118.48489574810866], [131.21518765419418, 121.51217888800802], [122.97461037812238, 121.56526096419978],
           [75.39827955150834, 179.4070640864827], [84.55991401346533, 176.2145796986134], [92.90235587470646, 174.4243211955652],
           [98.56534031739243, 176.0653659731581], [104.97777372929698, 174.45766843787231], [113.1125749468363, 176.39970964033202],
           [121.19973608809934, 179.19790184992027], [113.16310623913299, 185.69051008752652], [105.26365304952049, 188.31443911070232],
           [98.41771871303214, 188.9656394139811], [92.2240282658315, 188.38538897373022], [84.05109731022314, 185.74954657843966],
           [79.18422925303048, 179.8065722186372], [92.7172317110304, 179.5201781895618], [98.52973444977067, 180.1630365496041],
           [105.05932172975814, 179.42368920844928], [117.43706438358437, 179.7109259873213], [104.90869094557993, 180.32984591574524],
           [98.35933953480642, 181.15981769637827], [92.49485174856926, 180.48994809345996]])

def registration(face_lms, neutral= neutral, method = 'fullface'):
    '''
    Affine registration of face landmarks to neutral face.

    Args:
        face_lms(array): face landmarks to register with shape (n,136).
                         Columns 0~67 are x coordinates and 68~136 are y coordinates
        neutral(array): target neutral face array that face_lm will be registered
        method(str or list): If string, register to all landmarks ('fullface', default),
                    or inner parts of face nose,mouth,eyes, and brows ('inner').
                    If list, pass landmarks to register to e.g. [27, 28, 29, 30, 36, 39, 42, 45]
    Return:
        registered_lms: registered landmarks in shape (n,136)
    '''
    assert(type(face_lms)==np.ndarray), TypeError('face_lms must be type np.ndarray')
    assert(face_lms.ndim==2), ValueError('face_lms must be shape (n, 136)')
    assert(face_lms.shape[1]==136), ValueError('Must have 136 landmarks')
    registered_lms =[]
    for row in face_lms:
        face = [row[:68],row[68:]]
        face = np.array(face).T
    #   Rotate face
        primary = np.array(face)
        secondary = np.array(neutral)
        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X1, Y1 = pad(primary), pad(secondary)
        if type(method)==str:
            if method == 'fullface':
                A, res, rank, s = np.linalg.lstsq(X1, Y1,rcond=None)
            elif method == 'inner':
                A, res, rank, s = np.linalg.lstsq(X1[17:, :], Y1[17:, :],rcond=None)
            else:
                raise ValueError("method is either 'fullface' or 'inner'")
        elif type(method)==list:
            A, res, rank, s = np.linalg.lstsq(X1[method], Y1[method],rcond=None)
        else:
            raise TypeError("method is string ('fullface','inner') or list of landmarks")
        transform = lambda x: unpad(np.dot(pad(x), A))
        registered_lms.append(transform(primary).T.reshape(1,136).ravel())
    return np.array(registered_lms)
