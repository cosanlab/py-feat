from __future__ import division

"""Plotting Functions"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from feat.utils import load_h5
#from utils import load_h5
import warnings
import seaborn as sns
import matplotlib.colors as colors
from collections import OrderedDict

__all__ = ['draw_lineface', 'plot_face', 'draw_vectorfield', 'draw_muscles', 'get_heat', 'predict']
__author__ = ["Sophie Byrne", "Jin Hyun Cheong", "Luke Chang"]

def draw_lineface(currx, curry, ax=None, color='k', linestyle="-", linewidth=1, gaze = None, *args, **kwargs):
    ''' Plot Line Face

        Args:
            currx: vector (len(68)) of x coordinates
            curry: vector (len(68)) of y coordinates
            ax: matplotlib axis to add
            color: matplotlib line color
            linestyle: matplotlib linestyle
            linewidth: matplotlib linewidth
            gaze: array (len(4)) of gaze vectors (fifth value is whether to draw vectors)
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
    if gaze is None:
        gaze = [0,0,0,0]

    else:
        if len(gaze) != 4:
            raise ValueError('gaze must be len(4).')
        gaze = [gaze[0], gaze[1]/2, gaze[2], gaze[3]/2]#, gaze[4]]

    x = (currx[37] + currx[38] + currx[41] + currx[40]) / 4
    y = (curry[37] + curry[38] + curry[40] + curry[41]) / 4
    width = (- curry[37] - curry[38] + curry[40] + curry[41]) / 5
    pupil_l = plt.Circle([x + gaze[0], y - gaze[1]], width, color='k')
    x1 = (currx[43] + currx[46] + currx[44] + currx[47]) / 4
    y1 = (curry[43] + curry[44] + curry[46] + curry[47]) / 4
    width = (-curry[43] - curry[44] + curry[46] + curry[47]) / 5
    pupil_r = plt.Circle([x1 + gaze[2], y1 - gaze[3]], width, color='k')


    if ax is None:
        ax = _create_empty_figure()

    ax.add_patch(pupil_l)
    ax.add_patch(pupil_r)
    ax.add_line(face_outline)
    ax.add_line(eye_l)
    ax.add_line(eye_r)
    ax.add_line(eyebrow_l)
    ax.add_line(eyebrow_r)
    ax.add_line(lips1)
    ax.add_line(lips2)
    ax.add_line(nose1)
    ax.add_line(nose2)
    if gaze:
        ax.quiver([x, x1], [y, y1], [10*gaze[0], 10*gaze[2]], [-10*gaze[1], -10*gaze[3]],
                  color='r', width=.005, angles='xy',
                  scale_units='xy', scale=1)

def draw_vectorfield(reference, target, color='r', scale=1, width=.007,
                     ax=None, *args, **kwargs):
    ''' Draw vectorfield from reference to target

        Args:
            reference: reference landmarks (2,68)
            target: target landmarks (2,68)
            ax: matplotlib axis instance
            au: vector of action units (len(17))

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
            ax = _create_empty_figure()

    ax.quiver(reference[0,:],reference[1,:],currx, curry,
              color=color, width=width, angles='xy',
              scale_units='xy', scale=scale, *args, **kwargs)


def draw_muscles(currx, curry, au=None, ax=None, *args, **kwargs):
    ''' Draw Muscles

        Args:
            currx: vector (len(68)) of x coordinates
            curry: vector (len(68)) of y coordinates
            ax: matplotlib axis to add
    '''
    masseter_l = plt.Polygon([[currx[2], curry[2]], [currx[3], curry[3]],
                              [currx[4], curry[4]], [currx[5], curry[5]],
                              [currx[6], curry[6]], [currx[5], curry[33]]])

    masseter_r = plt.Polygon([[currx[14], curry[14]], [currx[13], curry[13]],
                              [currx[12], curry[12]], [currx[11], curry[11]],
                              [currx[10], curry[10]], [currx[11], curry[33]]])

    temporalis_l = plt.Polygon([[currx[2], curry[2]], [currx[1], curry[1]],
                                [currx[0], curry[0]], [currx[17], curry[17]],
                                [currx[36], curry[36]]])

    temporalis_r = plt.Polygon([[currx[14], curry[14]], [currx[15], curry[15]],
                                [currx[16], curry[16]], [currx[26], curry[26]],
                                [currx[45], curry[45]]])

    dep_lab_inf_l = plt.Polygon([[currx[57], curry[57]], [currx[58], curry[58]],
                                 [currx[59], curry[59]], [currx[6], curry[6]],
                                 [currx[7], curry[7]]], fill=True)

    dep_lab_inf_r = plt.Polygon([[currx[57], curry[57]], [currx[56], curry[56]],
                                 [currx[55], curry[55]], [currx[10], curry[10]],
                                 [currx[9], curry[9]]], fill=True)

    dep_ang_or_r = plt.Polygon([[currx[54], curry[54]], [currx[9], curry[9]],
                                [currx[10], curry[10]]], fill=True)

    dep_ang_or_l = plt.Polygon([[currx[48], curry[48]], [currx[7], curry[7]],
                                [currx[6], curry[6]]], fill=True)

    mentalis_l = plt.Polygon([[currx[58], curry[58]], [currx[7], curry[7]],
                              [currx[8], curry[8]]], fill=True)

    mentalis_r = plt.Polygon([[currx[56], curry[56]], [currx[9], curry[9]],
                              [currx[8], curry[8]]], fill=True)

    risorius_l = plt.Polygon([[currx[4], curry[4]], [currx[5], curry[5]],
                              [currx[48], curry[48]]], fill=True)

    risorius_r = plt.Polygon([[currx[11], curry[11]], [currx[12], curry[12]],
                              [currx[54], curry[54]]], fill=True)

    bottom = (curry[8] - curry[57]) / 2
    orb_oris_l = plt.Polygon([[currx[48], curry[48]], [currx[59], curry[59]],
                             [currx[58], curry[58]], [currx[57], curry[57]],
                             [currx[56], curry[56]], [currx[55], curry[55]],
                             [currx[54], curry[54]], [currx[55], curry[55] + bottom],
                             [currx[56], curry[56] + bottom], [currx[57], curry[57] + bottom],
                             [currx[58], curry[58] + bottom], [currx[59], curry[59] + bottom]])

    orb_oris_u = plt.Polygon([[currx[48], curry[48]], [currx[49], curry[49]],
                             [currx[50], curry[50]], [currx[51], curry[51]],
                             [currx[52], curry[52]], [currx[53], curry[53]],
                             [currx[54], curry[54]], [currx[33], curry[33]]], fill=True)


    frontalis_l = plt.Polygon([[currx[27], curry[27]], [currx[39], curry[39]],
                               [currx[38], curry[38]], [currx[37], curry[37]],
                               [currx[36], curry[36]], [currx[17], curry[17]],
                               [currx[18], curry[18]], [currx[19], curry[19]],
                               [currx[20], curry[20]], [currx[21], curry[21]]])

    frontalis_r = plt.Polygon([[currx[27], curry[27]], [currx[22], curry[22]],
                               [currx[23], curry[23]], [currx[24], curry[24]],
                               [currx[25], curry[25]], [currx[26], curry[26]],
                               [currx[45], curry[45]], [currx[44], curry[44]],
                               [currx[43], curry[43]], [currx[42], curry[42]]])

    frontalis_inner_l = plt.Polygon([[currx[27], curry[27]], [currx[39], curry[39]],
                                     [currx[21], curry[21]]])

    frontalis_inner_r = plt.Polygon([[currx[27], curry[27]], [currx[42], curry[42]],
                                     [currx[22], curry[22]]])

    cor_sup_l = plt.Polygon([[currx[28], curry[28]], [currx[19], curry[19]],
                             [currx[20], curry[20]]])

    cor_sup_r = plt.Polygon([[currx[28], curry[28]], [currx[23], curry[23]],
                             [currx[24], curry[24]]])

    lev_lab_sup_l = plt.Polygon([[currx[41], curry[41]], [currx[40], curry[40]],
                                 [currx[49], curry[49]]])

    lev_lab_sup_r = plt.Polygon([[currx[47], curry[47]], [currx[46], curry[46]],
                                 [currx[53], curry[53]]])

    lev_lab_sup_an_l = plt.Polygon([[currx[39], curry[39]], [currx[49], curry[49]],
                                    [currx[31], curry[31]]])

    lev_lab_sup_an_r = plt.Polygon([[currx[35], curry[35]], [currx[42], curry[42]],
                                    [currx[53], curry[53]]])

    zyg_maj_l = plt.Polygon([[currx[48], curry[48]], [currx[3], curry[3]], [currx[2], curry[2]]], color='r')

    zyg_maj_r = plt.Polygon([[currx[54], curry[54]], [currx[13], curry[13]], [currx[14], curry[14]]], color='r')

    width = (curry[21] - curry[39]) / 2
    orb_oc_l = plt.Polygon([[currx[36] - width / 3, curry[36] + width / 2], [currx[36], curry[36] + width],
                            [currx[37], curry[37] + width], [currx[38], curry[38] + width],
                            [currx[39], curry[39] + width], [currx[39] + width / 3, curry[39] + width / 2],
                            [currx[39] + width / 2, curry[39]], [currx[39] + width / 3, curry[39] - width / 2],
                            [currx[39], curry[39] - width], [currx[40], curry[40] - width],
                            [currx[41], curry[41] - width], [currx[36], curry[36] - width],
                            [currx[36] - width / 3, curry[36] - width / 2], [currx[36] - width / 2, curry[36]]])

    orb_oc_l_inner = plt.Polygon([[currx[36] - width / 6, curry[36] + width / 5], [currx[36], curry[36] + width/2],
                            [currx[37], curry[37] + width/2], [currx[38], curry[38] + width/2],
                            [currx[39], curry[39] + width/2], [currx[39] + width / 6, curry[39] + width / 5],
                            [currx[39] + width / 5, curry[39]], [currx[39] + width / 6, curry[39] - width / 5],
                            [currx[39], curry[39] - width/2], [currx[40], curry[40] - width/2],
                            [currx[41], curry[41] - width/2], [currx[36], curry[36] - width/2],
                            [currx[36] - width / 6, curry[36] - width / 5], [currx[36] - width / 5, curry[36]]], color = 'r')

    width2 = (curry[38] - curry[2])/1.5
    orb_oc_l_outer = plt.Polygon([[currx[39] + width / 2, curry[39]],
                            [currx[39], curry[39] - width], [currx[40], curry[40] - width2],
                            [currx[41], curry[41] - width2], [currx[36], curry[36] - width2],
                            [currx[36] - width2 / 3, curry[36] - width2 / 2], [currx[36] - width / 2, curry[36]]])

    width = (curry[23] - curry[43]) / 2
    orb_oc_r = plt.Polygon([[currx[42] - width / 3, curry[42] + width / 2], [currx[42], curry[42] + width],
                            [currx[43], curry[43] + width], [currx[44], curry[44] + width],
                            [currx[45], curry[45] + width], [currx[45] + width / 3, curry[45] + width / 2],
                            [currx[45] + width / 2, curry[45]], [currx[45] + width / 3, curry[45] - width / 2],
                            [currx[45], curry[45] - width], [currx[46], curry[46] - width],
                            [currx[47], curry[47] - width], [currx[42], curry[42] - width],
                            [currx[42] - width / 3, curry[42] - width / 2], [currx[42] - width / 2, curry[42]]])

    orb_oc_r_inner = plt.Polygon([[currx[42] - width / 6, curry[42] + width / 5], [currx[42], curry[42] + width/2],
                            [currx[43], curry[43] + width/2], [currx[44], curry[44] + width/2],
                            [currx[45], curry[45] + width/2], [currx[45] + width / 6, curry[45] + width / 5],
                            [currx[45] + width / 5, curry[45]], [currx[45] + width / 6, curry[45] - width / 5],
                            [currx[45], curry[45] - width/2], [currx[46], curry[46] - width/2],
                            [currx[47], curry[47] - width/2], [currx[42], curry[42] - width/2],
                            [currx[42] - width / 6, curry[42] - width / 5], [currx[42] - width / 5, curry[42]]])

    width2 = (curry[44] - curry[14]) / 1.5
    orb_oc_r_outer = plt.Polygon([[currx[42] - width / 2, curry[42]], [currx[47], curry[47] - width2],
                            [currx[46], curry[46] - width2], [currx[45], curry[45] - width2],
                            [currx[45] + width2 / 3, curry[45] - width2 / 2], [currx[45] + width / 2, curry[45]]])
    bucc_l = plt.Polygon([[currx[48], curry[48]], [currx[5], curry[50]],
                            [currx[5], curry[57]]], color = 'r')
    bucc_r = plt.Polygon([[currx[54], curry[54]], [currx[11], curry[52]],
                            [currx[11], curry[57]]], color = 'r')
    muscles = {'bucc_l': bucc_l, 'bucc_r': bucc_r,'masseter_l': masseter_l,
               'masseter_r': masseter_r, 'temporalis_l': temporalis_l,'temporalis_r': temporalis_r,
               'dep_lab_inf_l': dep_lab_inf_l, 'dep_lab_inf_r': dep_lab_inf_r, 'dep_ang_or_l': dep_ang_or_l,
               'dep_ang_or_r': dep_ang_or_r, 'mentalis_l': mentalis_l, 'mentalis_r': mentalis_r,
               'risorius_l': risorius_l, 'risorius_r': risorius_r, 'frontalis_l': frontalis_l,
               'frontalis_inner_l': frontalis_inner_l, 'frontalis_r': frontalis_r, 'frontalis_inner_r': frontalis_inner_r,
                'cor_sup_r': cor_sup_r, 'orb_oc_l_outer': orb_oc_l_outer,
               'orb_oc_r_outer': orb_oc_r_outer,'lev_lab_sup_l': lev_lab_sup_l, 'lev_lab_sup_r': lev_lab_sup_r,
               'lev_lab_sup_an_l': lev_lab_sup_an_l, 'lev_lab_sup_an_r': lev_lab_sup_an_r, 'zyg_maj_l': zyg_maj_l,
               'zyg_maj_r': zyg_maj_r, 'orb_oc_l': orb_oc_l, 'orb_oc_r': orb_oc_r, 'orb_oc_l_inner': orb_oc_l_inner,
               'orb_oc_r_inner': orb_oc_r_inner, 'orb_oris_l': orb_oris_l, 'orb_oris_u': orb_oris_u,
               'orb_oc_l': orb_oc_l, 'cor_sup_l': cor_sup_l}

    muscle_names = ['bucc_l', 'bucc_r', 'masseter_l', 'masseter_r', 'temporalis_l', 'temporalis_r',
               'dep_lab_inf_l', 'dep_lab_inf_r', 'dep_ang_or_l', 'dep_ang_or_r', 'mentalis_l', 'mentalis_r',
               'risorius_l', 'risorius_r', 'frontalis_l', 'frontalis_inner_l', 'frontalis_r', 'frontalis_inner_r',
               'cor_sup_r','orb_oc_l_outer', 'orb_oc_r_outer','lev_lab_sup_l', 'lev_lab_sup_r',
               'lev_lab_sup_an_l','lev_lab_sup_an_r', 'zyg_maj_l', 'zyg_maj_r','orb_oc_l', 'orb_oc_r', 'orb_oc_l',
               'orb_oc_l_inner', 'orb_oc_r_inner','orb_oris_l','orb_oris_u', 'cor_sup_l']
    todraw = {}
    facet = False

    if 'facet' in kwargs and au is not None:
        aus = []
        for i in range(12):
            aus.append(au[i])
        aus.append(au[13])
        aus.append(max(au[12],au[14],au[15], au[18], key=abs))
        aus.append(au[16])
        aus.append(au[17])
        aus.append(au[19])
        au = aus
        facet = True
        del kwargs['facet']
    if au is None:
        au = np.zeros(20)
    if 'all' in kwargs:
        for muscle in muscle_names:
            todraw[muscle] = kwargs['all']
        del kwargs['all']
    else:
        for muscle in muscle_names:
            if muscle in kwargs:
                todraw[muscle] = kwargs[muscle]
                del kwargs[muscle]
    for muscle in todraw.keys():
        if todraw[muscle] == 'heatmap':
            # muscles[muscle].set_color(get_heat(muscle, au, facet))
            muscles[muscle].set_color(get_heat(muscle, au, facet))
        else:
            muscles[muscle].set_color(todraw[muscle])
        ax.add_patch(muscles[muscle], *args, **kwargs)

    eye_l = plt.Polygon([[currx[36], curry[36]], [currx[37], curry[37]],
                        [currx[38], curry[38]], [currx[39], curry[39]],
                        [currx[40], curry[40]], [currx[41], curry[41]]], color='w')

    eye_r = plt.Polygon([[currx[42], curry[42]], [currx[43], curry[43]],
                        [currx[44], curry[44]], [currx[45], curry[45]],
                        [currx[46], curry[46]], [currx[47], curry[47]]], color='w')

    mouth = plt.Polygon([[currx[60], curry[60]], [currx[61], curry[61]],
                        [currx[62], curry[62]], [currx[63], curry[63]],
                        [currx[64], curry[64]], [currx[65], curry[65]],
                            [currx[66], curry[66]], [currx[67], curry[67]]], color='w')

    # ax.add_patch(eye_l)
    # ax.add_patch(eye_r)
    # ax.add_patch(mouth)
    return ax


def get_heat(muscle, au, log):
    ''' Function to create heatmap from au vector

        Args:
            au: vector of action units
            muscle: string representation of a muscle
            boolean: whether the action unit values are on a log scale


        Returns:
            color of muscle according to its au value
    '''
    q = sns.color_palette("Blues", 101)
    unit = 0
    aus = {'masseter_l': 15, 'masseter_r': 15, 'temporalis_l': 15,
           'temporalis_r': 15, 'dep_lab_inf_l': 14, 'dep_lab_inf_r': 14,
           'dep_ang_or_l': 10, 'dep_ang_or_r': 10, 'mentalis_l': 11,
           'mentalis_r': 11, 'risorius_l': 12, 'risorius_r': 12,
           'frontalis_l': 1,'frontalis_r': 1, 'frontalis_inner_l': 0,
           'frontalis_inner_r': 0, 'cor_sup_l': 2, 'cor_sup_r': 2,
           'lev_lab_sup_l': 7,'lev_lab_sup_r': 7, 'lev_lab_sup_an_l': 6,
           'lev_lab_sup_an_r': 6, 'zyg_maj_l': 8, 'zyg_maj_r': 8,
           'bucc_l': 9, 'bucc_r': 9,'orb_oc_l_outer': 4, 'orb_oc_r_outer': 4,
           'orb_oc_l': 5, 'orb_oc_r': 5, "orb_oc_l_inner": 16,
           "orb_oc_r_inner": 16, 'orb_oris_l': 13, 'orb_oris_u': 13}
    if muscle in aus:
        unit = aus[muscle]
    if log:
        num = int(100*(1. / (1 + 10. ** -(au[unit]))))
    else:
        num = int(au[unit]*20)
    # set alpha (opacity)
    if au[unit] == 0:
        alpha=0
    else: 
        alpha=.5
    # color = colors.to_hex(q[num])
    # return str(color)
    color = colors.to_rgba(q[num], alpha=alpha)
    return color


def plot_face(model=None, au=None, vectorfield=None, muscles = None, ax=None, color='k', linewidth=1,
              linestyle='-', gaze = None, *args, **kwargs):
    ''' Function to plot facesself

        Args:
            model: sklearn PLSRegression instance
            au: vector of action units (same length as model.x_mean_)
            vectorfield: (dict) {'target':target_array,'reference':reference_array}
            muscles: (dict) {'muscle': color}
            ax: matplotlib axis handle
            color: matplotlib color
            linewidth: matplotlib linewidth
            linestyle: matplotlib linestyle
            gaze: array of gaze vectors (len(4))

        Returns:
            ax: plot handle
    '''

    if model is None:
        model = load_h5()
    else:
        if not isinstance(model, PLSRegression):
            raise ValueError('make sure that model is a PLSRegression instance')

    if au is None:
        au = np.zeros(len(model.x_mean_))
        warnings.warn("Don't forget to pass an 'au' vector of len(20), "
                      "using neutral as default")

    landmarks = predict(au, model)
    currx, curry = ([landmarks[x,:] for x in range(2)])

    if ax is None:
        ax = _create_empty_figure()

    if muscles is not None:
        if not isinstance(muscles, dict):
            raise ValueError('muscles must be a dictionary ')
        draw_muscles(currx, curry, ax=ax, au=au, **muscles)

    if gaze is not None and len((gaze)) != 4:
        warnings.warn("Don't forget to pass a 'gaze' vector of len(4), "
                      "using neutral as default")
        gaze = None

    draw_lineface(currx, curry, color=color, linewidth=linewidth,
                  linestyle=linestyle, ax=ax, gaze = gaze, *args, **kwargs)
    if vectorfield is not None:
        if not isinstance(vectorfield, dict):
            raise ValueError('vectorfield must be a dictionary ')
        if 'reference' not in vectorfield:
            raise ValueError("vectorfield must contain 'reference' key")
        if 'target' not in vectorfield.keys():
            vectorfield['target'] = landmarks
        draw_vectorfield(ax=ax, **vectorfield)
    ax.set_xlim([25,172])
    ax.set_ylim((-240,-50))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax

def predict(au, model=None):
    ''' Helper function to predict landmarks from au given a sklearn model

        Args:
            au: vector of action unit intensities
            model: sklearn pls object (uses pretrained model by default)

        Returns:
            landmarks: Array of landmarks (2,68)

    '''

    if model is None:
        model = load_h5()
    elif not isinstance(model, PLSRegression):
        raise ValueError('make sure that model is a PLSRegression instance')

    if len(au) != len(model.x_mean_):
        print(au)
        print(len(model.x_mean_))
        raise ValueError('au vector must be len(', len(model.x_mean_), ').')

    if len(au.shape) == 1:
        au = np.reshape(au, (1, -1))

    landmarks = np.reshape(model.predict(au), (2,68))
    landmarks[1,:] = -1*landmarks[1,:] # this might not generalize to other models
    return landmarks

def _create_empty_figure(figsize=(4,5), xlim=[25,172], ylim=[-240,-50]):
    '''Create an empty figure'''
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax
