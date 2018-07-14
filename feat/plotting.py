from __future__ import division

"""Plotting Functions"""

import numpy as np
from sklearn.cross_decomposition.pls_ import PLSRegression
import matplotlib.pyplot as plt

__all__ = ['draw_lineface', 'plot_face', 'draw_vectorfield', '_predict']
__author__ = ["Sophie Byrne", "Luke Chang"]

def draw_lineface(currx, curry, ax=None, color='k', linestyle="-", linewidth=1,
                  *args, **kwargs):
    ''' Plot Line Face

        Args:
            currx: vector (len(68)) of x coordinates
            curry: vector (len(68)) of y coordinates
            ax: matplotlib axis to add
            color: matplotlib line color
            linestyle: matplotlib linestyle
            linewidth: matplotlib linewidth
    '''

    face_outline = plt.Line2D([currx[0], currx[1], currx[2], currx[3], currx[4],
                              currx[5], currx[6], currx[7], currx[8], currx[9],
                              currx[10], currx[11], currx[12], currx[13],
                              currx[14], currx[15], currx[16]], [curry[0],
                              curry[1], curry[2], curry[3], curry[4], curry[5],
                              curry[6],curry[7], curry[8], curry[9], curry[10],
                              curry[11], curry[12], curry[13], curry[14],
                              curry[15], curry[16]],
                              color=color, linestyle=linestyle,
                              linewidth=linewidth, *args, **kwargs)

    eye_l = plt.Line2D([currx[36], currx[37], currx[38], currx[39], currx[40],
                        currx[41], currx[36]],
                       [curry[36], curry[37], curry[38],curry[39], curry[40],
                       curry[41], curry[36]],
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       *args, **kwargs)

    eye_r = plt.Line2D([currx[42], currx[43], currx[44], currx[45], currx[46],
                        currx[47], currx[42]],
                       [curry[42], curry[43], curry[44], curry[45], curry[46],
                        curry[47], curry[42]],
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       *args, **kwargs)

    eyebrow_l = plt.Line2D([currx[17], currx[18], currx[19], currx[20],
                           currx[21]], [curry[17], curry[18], curry[19],
                           curry[20], curry[21]],
                           color=color, linestyle=linestyle,
                           linewidth=linewidth, *args, **kwargs)

    eyebrow_r = plt.Line2D([currx[22], currx[23], currx[24], currx[25],
                           currx[26]],
                          [curry[22], curry[23], curry[24], curry[25],
                           curry[26]],
                           color=color, linestyle=linestyle,
                           linewidth=linewidth, *args, **kwargs)

    lips1 = plt.Line2D([currx[48], currx[49], currx[50], currx[51], currx[52],
                        currx[53], currx[54], currx[64], currx[63], currx[62],
                        currx[61], currx[60], currx[48]],
                        [curry[48], curry[49], curry[50], curry[51], curry[52],
                        curry[53], curry[54], curry[64], curry[63], curry[62],
                        curry[61], curry[60], curry[48]],
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        *args, **kwargs)

    lips2 = plt.Line2D([currx[48], currx[60], currx[67], currx[66], currx[65],
                        currx[64], currx[54], currx[55], currx[56], currx[57],
                        currx[58], currx[59], currx[48]],
                        [curry[48], curry[60], curry[67], curry[66], curry[65],
                        curry[64], curry[54], curry[55], curry[56], curry[57],
                        curry[58], curry[59], curry[48]],
                        color=color, linestyle=linestyle, linewidth=linewidth,
                        *args, **kwargs)

    nose1 = plt.Line2D([currx[27],currx[28],currx[29],currx[30]],
                       [curry[27], curry[28],curry[29],curry[30]],
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       *args, **kwargs)

    nose2= plt.Line2D([currx[31], currx[32], currx[33], currx[34], currx[35]],
                      [curry[31], curry[32], curry[33], curry[34], curry[35]],
                      color=color, linestyle=linestyle, linewidth=linewidth,
                      *args, **kwargs)

    if ax is None:
        plt.figure(figsize=(4,5))
        ax = plt.gca()

    ax.add_line(face_outline)
    ax.add_line(eye_l)
    ax.add_line(eye_r)
    ax.add_line(eyebrow_l)
    ax.add_line(eyebrow_r)
    ax.add_line(lips1)
    ax.add_line(lips2)
    ax.add_line(nose1)
    ax.add_line(nose2)

def draw_vectorfield(reference, target, color='r', scale=1, width=.007,
                     ax=None, *args, **kwargs):
    ''' Draw vectorfield from reference to target

        Args:
            reference: reference landmarks (2,68)
            target: target landmarks (2,68)
            ax: matplotlib axis instance

    '''
    if reference.shape != (2,68):
        raise ValueError('shape of reference must be (2,68)')
    if target.shape != (2,68):
        raise ValueError('shape of target must be (2,68)')

    currx = []; curry = [];
    for i in range(68):
        currx.append(target[0, i] - reference[0, i])
        curry.append(target[1, i] - reference[1, i])

        if ax is None:
            plt.figure(figsize=(4,5))
            ax = plt.gca()

    ax.quiver(reference[0,:],reference[1,:],currx, curry,
              color=color, width=width, angles='xy',
              scale_units='xy', scale=scale, *args, **kwargs)

def plot_face(model=None, au=None, vectorfield=None, ax=None, color='k', linewidth=1,
              linestyle='-', *args, **kwargs):
    ''' Function to plot facesself

        Args:
            model: sklearn PLSRegression instance
            au: vector of action units (len(17))
            vectorfield: (dict) {'target':target_array,'reference':reference_array}
            ax: matplotlib axis handle
            color: matplotlib color
            linewidth: matplotlib linewidth
            linestyle: matplotlib linestyle

        Returns:
    '''

    if model is None:
        raise ValueError('make sure that model is a PLSRegression instance')

    if au is None:
        raise ValueError('au vector must be len(17).')

    landmarks = _predict(au, model)
    currx, curry = ([landmarks[x,:] for x in range(2)])

    if ax is None:
        plt.figure(figsize=(4,5))
        ax = plt.gca()

    draw_lineface(currx, curry, color=color, linewidth=linewidth,
                  linestyle=linestyle, ax=ax, *args, **kwargs)

    if vectorfield is not None:
        if not isinstance(vectorfield, dict):
            raise ValueError('vectorfield must be a dictionary ')
        if 'target' not in vectorfield:
            raise ValueError("vectorfield must contain 'target' key")
        target = vectorfield['target']
        del vectorfield['target']
        draw_vectorfield(landmarks, target, ax=ax, **vectorfield)
    ax.set_xlim([25,172])
    ax.set_ylim((-240,-50))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def _predict(au, model):
    ''' Helper function to predict landmarks from au given a sklearn model

        Args:
            au: vector of 17 action unit intensities
            model: sklearn pls object

        Returns:
            landmarks: Array of landmarks (2,68)

    '''

    if not isinstance(pls, PLSRegression):
        raise ValueError('make sure that model is a PLSRegression instance')

    if len(au) != 17:
        raise ValueError('au vector must be len(17).')

    if len(au.shape) == 1:
        au = np.reshape(au, (1, -1))

    landmarks = np.reshape(model.predict(au), (2,68))
    landmarks[1,:] = -1*landmarks[1,:] # this might not generalize to other models
    return landmarks
