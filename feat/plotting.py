"""
Helper functions for plotting
"""

import os
import sys
import h5py
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn import __version__ as skversion
from sklearn.preprocessing import PolynomialFeatures, scale
import matplotlib.pyplot as plt
from feat.pretrained import AU_LANDMARK_MAP
from math import sin, cos
import warnings
import seaborn as sns
import matplotlib.colors as colors
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from pathlib import Path
from PIL import Image
from textwrap import wrap
from joblib import load
from feat.utils import flatten_list
from feat.utils.io import get_resource_path, download_url
import json

__all__ = [
    "draw_lineface",
    "plot_face",
    "draw_vectorfield",
    "draw_muscles",
    "get_heat",
    "predict",
    "imshow",
    "interpolate_aus",
    "animate_face",
    "face_part_path",
    "draw_plotly_landmark",
    "face_polygon_svg",
    "draw_plotly_au",
    "draw_plotly_pose",
    "emotion_annotation_position",
]


def draw_lineface(
    currx,
    curry,
    ax=None,
    color="k",
    linestyle="-",
    linewidth=1,
    gaze=None,
    *args,
    **kwargs,
):
    """Plot Line Face

    Args:
        currx: vector (len(68)) of x coordinates
        curry: vector (len(68)) of y coordinates
        ax: matplotlib axis to add
        color: matplotlib line color
        linestyle: matplotlib linestyle
        linewidth: matplotlib linewidth
        gaze: array (len(4)) of gaze vectors (fifth value is whether to draw vectors)
    """

    face_outline = plt.Line2D(
        [
            currx[0],
            currx[1],
            currx[2],
            currx[3],
            currx[4],
            currx[5],
            currx[6],
            currx[7],
            currx[8],
            currx[9],
            currx[10],
            currx[11],
            currx[12],
            currx[13],
            currx[14],
            currx[15],
            currx[16],
        ],
        [
            curry[0],
            curry[1],
            curry[2],
            curry[3],
            curry[4],
            curry[5],
            curry[6],
            curry[7],
            curry[8],
            curry[9],
            curry[10],
            curry[11],
            curry[12],
            curry[13],
            curry[14],
            curry[15],
            curry[16],
        ],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    eye_l = plt.Line2D(
        [currx[36], currx[37], currx[38], currx[39], currx[40], currx[41], currx[36]],
        [curry[36], curry[37], curry[38], curry[39], curry[40], curry[41], curry[36]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    eye_r = plt.Line2D(
        [currx[42], currx[43], currx[44], currx[45], currx[46], currx[47], currx[42]],
        [curry[42], curry[43], curry[44], curry[45], curry[46], curry[47], curry[42]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    eyebrow_l = plt.Line2D(
        [currx[17], currx[18], currx[19], currx[20], currx[21]],
        [curry[17], curry[18], curry[19], curry[20], curry[21]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    eyebrow_r = plt.Line2D(
        [currx[22], currx[23], currx[24], currx[25], currx[26]],
        [curry[22], curry[23], curry[24], curry[25], curry[26]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    lips1 = plt.Line2D(
        [
            currx[48],
            currx[49],
            currx[50],
            currx[51],
            currx[52],
            currx[53],
            currx[54],
            currx[64],
            currx[63],
            currx[62],
            currx[61],
            currx[60],
            currx[48],
        ],
        [
            curry[48],
            curry[49],
            curry[50],
            curry[51],
            curry[52],
            curry[53],
            curry[54],
            curry[64],
            curry[63],
            curry[62],
            curry[61],
            curry[60],
            curry[48],
        ],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    lips2 = plt.Line2D(
        [
            currx[48],
            currx[60],
            currx[67],
            currx[66],
            currx[65],
            currx[64],
            currx[54],
            currx[55],
            currx[56],
            currx[57],
            currx[58],
            currx[59],
            currx[48],
        ],
        [
            curry[48],
            curry[60],
            curry[67],
            curry[66],
            curry[65],
            curry[64],
            curry[54],
            curry[55],
            curry[56],
            curry[57],
            curry[58],
            curry[59],
            curry[48],
        ],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    nose1 = plt.Line2D(
        [currx[27], currx[28], currx[29], currx[30]],
        [curry[27], curry[28], curry[29], curry[30]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )

    nose2 = plt.Line2D(
        [currx[31], currx[32], currx[33], currx[34], currx[35]],
        [curry[31], curry[32], curry[33], curry[34], curry[35]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        *args,
        **kwargs,
    )
    if gaze is None:
        gaze = [0, 0, 0, 0]

    else:
        if len(gaze) != 4:
            raise ValueError("gaze must be len(4).")
        gaze = [gaze[0], gaze[1] / 2, gaze[2], gaze[3] / 2]  # , gaze[4]]

    x = (currx[37] + currx[38] + currx[41] + currx[40]) / 4
    y = (curry[37] + curry[38] + curry[40] + curry[41]) / 4
    width = (-curry[37] - curry[38] + curry[40] + curry[41]) / 5
    pupil_l = plt.Circle([x + gaze[0], y - gaze[1]], width, color="k")
    x1 = (currx[43] + currx[46] + currx[44] + currx[47]) / 4
    y1 = (curry[43] + curry[44] + curry[46] + curry[47]) / 4
    width = (-curry[43] - curry[44] + curry[46] + curry[47]) / 5
    pupil_r = plt.Circle([x1 + gaze[2], y1 - gaze[3]], width, color="k")

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
        ax.quiver(
            [x, x1],
            [y, y1],
            [10 * gaze[0], 10 * gaze[2]],
            [-10 * gaze[1], -10 * gaze[3]],
            color="r",
            width=0.005,
            angles="xy",
            scale_units="xy",
            scale=1,
        )
    return ax


def draw_vectorfield(
    reference, target, color="r", scale=1, width=0.007, ax=None, *args, **kwargs
):
    """Draw vectorfield from reference to target

    Args:
        reference: reference landmarks (2,68)
        target: target landmarks (2,68)
        ax: matplotlib axis instance
        au: vector of action units (len(17))

    """
    if reference.shape != (2, 68):
        raise ValueError("shape of reference must be (2,68)")
    if target.shape != (2, 68):
        raise ValueError("shape of target must be (2,68)")

    currx = []
    curry = []
    for i in range(68):
        currx.append(target[0, i] - reference[0, i])
        curry.append(target[1, i] - reference[1, i])

        if ax is None:
            ax = _create_empty_figure()

    ax.quiver(
        reference[0, :],
        reference[1, :],
        currx,
        curry,
        color=color,
        width=width,
        angles="xy",
        scale_units="xy",
        scale=scale,
        *args,
        **kwargs,
    )
    return ax


def draw_muscles(currx, curry, au=None, ax=None, *args, **kwargs):
    """Draw Muscles

    Args:
        currx: vector (len(68)) of x coordinates
        curry: vector (len(68)) of y coordinates
        ax: matplotlib axis to add
    """
    masseter_l = plt.Polygon(
        [
            [currx[2], curry[2]],
            [currx[3], curry[3]],
            [currx[4], curry[4]],
            [currx[5], curry[5]],
            [currx[6], curry[6]],
            [currx[5], curry[33]],
        ]
    )

    masseter_r = plt.Polygon(
        [
            [currx[14], curry[14]],
            [currx[13], curry[13]],
            [currx[12], curry[12]],
            [currx[11], curry[11]],
            [currx[10], curry[10]],
            [currx[11], curry[33]],
        ]
    )

    temporalis_l = plt.Polygon(
        [
            [currx[2], curry[2]],
            [currx[1], curry[1]],
            [currx[0], curry[0]],
            [currx[17], curry[17]],
            [currx[36], curry[36]],
        ]
    )

    temporalis_r = plt.Polygon(
        [
            [currx[14], curry[14]],
            [currx[15], curry[15]],
            [currx[16], curry[16]],
            [currx[26], curry[26]],
            [currx[45], curry[45]],
        ]
    )

    dep_lab_inf_l = plt.Polygon(
        [
            [currx[57], curry[57]],
            [currx[58], curry[58]],
            [currx[59], curry[59]],
            [currx[6], curry[6]],
            [currx[7], curry[7]],
        ],
        fill=True,
    )

    dep_lab_inf_r = plt.Polygon(
        [
            [currx[57], curry[57]],
            [currx[56], curry[56]],
            [currx[55], curry[55]],
            [currx[10], curry[10]],
            [currx[9], curry[9]],
        ],
        fill=True,
    )

    dep_ang_or_r = plt.Polygon(
        [[currx[54], curry[54]], [currx[9], curry[9]], [currx[10], curry[10]]],
        fill=True,
    )

    dep_ang_or_l = plt.Polygon(
        [[currx[48], curry[48]], [currx[7], curry[7]], [currx[6], curry[6]]], fill=True
    )

    mentalis_l = plt.Polygon(
        [[currx[58], curry[58]], [currx[7], curry[7]], [currx[8], curry[8]]], fill=True
    )

    mentalis_r = plt.Polygon(
        [[currx[56], curry[56]], [currx[9], curry[9]], [currx[8], curry[8]]], fill=True
    )

    risorius_l = plt.Polygon(
        [[currx[4], curry[4]], [currx[5], curry[5]], [currx[48], curry[48]]], fill=True
    )

    risorius_r = plt.Polygon(
        [[currx[11], curry[11]], [currx[12], curry[12]], [currx[54], curry[54]]],
        fill=True,
    )

    bottom = (curry[8] - curry[57]) / 2
    orb_oris_l = plt.Polygon(
        [
            [currx[48], curry[48]],
            [currx[59], curry[59]],
            [currx[58], curry[58]],
            [currx[57], curry[57]],
            [currx[56], curry[56]],
            [currx[55], curry[55]],
            [currx[54], curry[54]],
            [currx[55], curry[55] + bottom],
            [currx[56], curry[56] + bottom],
            [currx[57], curry[57] + bottom],
            [currx[58], curry[58] + bottom],
            [currx[59], curry[59] + bottom],
        ]
    )

    orb_oris_u = plt.Polygon(
        [
            [currx[48], curry[48]],
            [currx[49], curry[49]],
            [currx[50], curry[50]],
            [currx[51], curry[51]],
            [currx[52], curry[52]],
            [currx[53], curry[53]],
            [currx[54], curry[54]],
            [currx[33], curry[33]],
        ],
        fill=True,
    )

    frontalis_l = plt.Polygon(
        [
            [currx[27], curry[27]],
            [currx[39], curry[39]],
            [currx[38], curry[38]],
            [currx[37], curry[37]],
            [currx[36], curry[36]],
            [currx[17], curry[17]],
            [currx[18], curry[18]],
            [currx[19], curry[19]],
            [currx[20], curry[20]],
            [currx[21], curry[21]],
        ]
    )

    frontalis_r = plt.Polygon(
        [
            [currx[27], curry[27]],
            [currx[22], curry[22]],
            [currx[23], curry[23]],
            [currx[24], curry[24]],
            [currx[25], curry[25]],
            [currx[26], curry[26]],
            [currx[45], curry[45]],
            [currx[44], curry[44]],
            [currx[43], curry[43]],
            [currx[42], curry[42]],
        ]
    )

    frontalis_inner_l = plt.Polygon(
        [[currx[27], curry[27]], [currx[39], curry[39]], [currx[21], curry[21]]]
    )

    frontalis_inner_r = plt.Polygon(
        [[currx[27], curry[27]], [currx[42], curry[42]], [currx[22], curry[22]]]
    )

    cor_sup_l = plt.Polygon(
        [[currx[28], curry[28]], [currx[19], curry[19]], [currx[20], curry[20]]]
    )

    cor_sup_r = plt.Polygon(
        [[currx[28], curry[28]], [currx[23], curry[23]], [currx[24], curry[24]]]
    )

    lev_lab_sup_l = plt.Polygon(
        [[currx[41], curry[41]], [currx[40], curry[40]], [currx[49], curry[49]]]
    )

    lev_lab_sup_r = plt.Polygon(
        [[currx[47], curry[47]], [currx[46], curry[46]], [currx[53], curry[53]]]
    )

    lev_lab_sup_an_l = plt.Polygon(
        [[currx[39], curry[39]], [currx[49], curry[49]], [currx[31], curry[31]]]
    )

    lev_lab_sup_an_r = plt.Polygon(
        [[currx[35], curry[35]], [currx[42], curry[42]], [currx[53], curry[53]]]
    )

    zyg_maj_l = plt.Polygon(
        [[currx[48], curry[48]], [currx[3], curry[3]], [currx[2], curry[2]]], color="r"
    )

    zyg_maj_r = plt.Polygon(
        [[currx[54], curry[54]], [currx[13], curry[13]], [currx[14], curry[14]]],
        color="r",
    )

    width = (curry[21] - curry[39]) / 2
    orb_oc_l = plt.Polygon(
        [
            [currx[36] - width / 3, curry[36] + width / 2],
            [currx[36], curry[36] + width],
            [currx[37], curry[37] + width],
            [currx[38], curry[38] + width],
            [currx[39], curry[39] + width],
            [currx[39] + width / 3, curry[39] + width / 2],
            [currx[39] + width / 2, curry[39]],
            [currx[39] + width / 3, curry[39] - width / 2],
            [currx[39], curry[39] - width],
            [currx[40], curry[40] - width],
            [currx[41], curry[41] - width],
            [currx[36], curry[36] - width],
            [currx[36] - width / 3, curry[36] - width / 2],
            [currx[36] - width / 2, curry[36]],
        ]
    )

    orb_oc_l_inner = plt.Polygon(
        [
            [currx[36] - width / 6, curry[36] + width / 5],
            [currx[36], curry[36] + width / 2],
            [currx[37], curry[37] + width / 2],
            [currx[38], curry[38] + width / 2],
            [currx[39], curry[39] + width / 2],
            [currx[39] + width / 6, curry[39] + width / 5],
            [currx[39] + width / 5, curry[39]],
            [currx[39] + width / 6, curry[39] - width / 5],
            [currx[39], curry[39] - width / 2],
            [currx[40], curry[40] - width / 2],
            [currx[41], curry[41] - width / 2],
            [currx[36], curry[36] - width / 2],
            [currx[36] - width / 6, curry[36] - width / 5],
            [currx[36] - width / 5, curry[36]],
        ],
        color="r",
    )

    width2 = (curry[38] - curry[2]) / 1.5
    orb_oc_l_outer = plt.Polygon(
        [
            [currx[39] + width / 2, curry[39]],
            [currx[39], curry[39] - width],
            [currx[40], curry[40] - width2],
            [currx[41], curry[41] - width2],
            [currx[36], curry[36] - width2],
            [currx[36] - width2 / 3, curry[36] - width2 / 2],
            [currx[36] - width / 2, curry[36]],
        ]
    )

    width = (curry[23] - curry[43]) / 2
    orb_oc_r = plt.Polygon(
        [
            [currx[42] - width / 3, curry[42] + width / 2],
            [currx[42], curry[42] + width],
            [currx[43], curry[43] + width],
            [currx[44], curry[44] + width],
            [currx[45], curry[45] + width],
            [currx[45] + width / 3, curry[45] + width / 2],
            [currx[45] + width / 2, curry[45]],
            [currx[45] + width / 3, curry[45] - width / 2],
            [currx[45], curry[45] - width],
            [currx[46], curry[46] - width],
            [currx[47], curry[47] - width],
            [currx[42], curry[42] - width],
            [currx[42] - width / 3, curry[42] - width / 2],
            [currx[42] - width / 2, curry[42]],
        ]
    )

    orb_oc_r_inner = plt.Polygon(
        [
            [currx[42] - width / 6, curry[42] + width / 5],
            [currx[42], curry[42] + width / 2],
            [currx[43], curry[43] + width / 2],
            [currx[44], curry[44] + width / 2],
            [currx[45], curry[45] + width / 2],
            [currx[45] + width / 6, curry[45] + width / 5],
            [currx[45] + width / 5, curry[45]],
            [currx[45] + width / 6, curry[45] - width / 5],
            [currx[45], curry[45] - width / 2],
            [currx[46], curry[46] - width / 2],
            [currx[47], curry[47] - width / 2],
            [currx[42], curry[42] - width / 2],
            [currx[42] - width / 6, curry[42] - width / 5],
            [currx[42] - width / 5, curry[42]],
        ]
    )

    width2 = (curry[44] - curry[14]) / 1.5
    orb_oc_r_outer = plt.Polygon(
        [
            [currx[42] - width / 2, curry[42]],
            [currx[47], curry[47] - width2],
            [currx[46], curry[46] - width2],
            [currx[45], curry[45] - width2],
            [currx[45] + width2 / 3, curry[45] - width2 / 2],
            [currx[45] + width / 2, curry[45]],
        ]
    )
    bucc_l = plt.Polygon(
        [[currx[48], curry[48]], [currx[5], curry[50]], [currx[5], curry[57]]],
        color="r",
    )
    bucc_r = plt.Polygon(
        [[currx[54], curry[54]], [currx[11], curry[52]], [currx[11], curry[57]]],
        color="r",
    )
    muscles = {
        "bucc_l": bucc_l,
        "bucc_r": bucc_r,
        "masseter_l": masseter_l,
        "masseter_r": masseter_r,
        "temporalis_l": temporalis_l,
        "temporalis_r": temporalis_r,
        "dep_lab_inf_l": dep_lab_inf_l,
        "dep_lab_inf_r": dep_lab_inf_r,
        "dep_ang_or_l": dep_ang_or_l,
        "dep_ang_or_r": dep_ang_or_r,
        "mentalis_l": mentalis_l,
        "mentalis_r": mentalis_r,
        "risorius_l": risorius_l,
        "risorius_r": risorius_r,
        "frontalis_l": frontalis_l,
        "frontalis_inner_l": frontalis_inner_l,
        "frontalis_r": frontalis_r,
        "frontalis_inner_r": frontalis_inner_r,
        "cor_sup_r": cor_sup_r,
        "orb_oc_l_outer": orb_oc_l_outer,
        "orb_oc_r_outer": orb_oc_r_outer,
        "lev_lab_sup_l": lev_lab_sup_l,
        "lev_lab_sup_r": lev_lab_sup_r,
        "lev_lab_sup_an_l": lev_lab_sup_an_l,
        "lev_lab_sup_an_r": lev_lab_sup_an_r,
        "zyg_maj_l": zyg_maj_l,
        "zyg_maj_r": zyg_maj_r,
        "orb_oc_l": orb_oc_l,
        "orb_oc_r": orb_oc_r,
        "orb_oc_l_inner": orb_oc_l_inner,
        "orb_oc_r_inner": orb_oc_r_inner,
        "orb_oris_l": orb_oris_l,
        "orb_oris_u": orb_oris_u,
        "orb_oc_l": orb_oc_l,
        "cor_sup_l": cor_sup_l,
        "pars_palp_l": orb_oc_l_inner,
        "pars_palp_r": orb_oc_r_inner,
        "masseter_l_rel": masseter_l,
        "masseter_r_rel": masseter_r,
        "temporalis_l_rel": temporalis_l,
        "temporalis_r_rel": temporalis_r,
    }

    muscle_names = [
        "bucc_l",
        "bucc_r",
        "masseter_l",
        "masseter_r",
        "temporalis_l",
        "temporalis_r",
        "dep_lab_inf_l",
        "dep_lab_inf_r",
        "dep_ang_or_l",
        "dep_ang_or_r",
        "mentalis_l",
        "mentalis_r",
        "risorius_l",
        "risorius_r",
        "frontalis_l",
        "frontalis_inner_l",
        "frontalis_r",
        "frontalis_inner_r",
        "cor_sup_r",
        "orb_oc_l_outer",
        "orb_oc_r_outer",
        "lev_lab_sup_l",
        "lev_lab_sup_r",
        "lev_lab_sup_an_l",
        "lev_lab_sup_an_r",
        "zyg_maj_l",
        "zyg_maj_r",
        "orb_oc_l",
        "orb_oc_r",
        "orb_oc_l",
        "orb_oc_l_inner",
        "orb_oc_r_inner",
        "orb_oris_l",
        "orb_oris_u",
        "cor_sup_l",
        "pars_palp_l",
        "pars_palp_r",
        "masseter_l_rel",
        "masseter_r_rel",
        "temporalis_l_rel",
        "temporalis_r_rel",
    ]
    todraw = {}
    facet = False

    if "facet" in kwargs and au is not None:
        aus = []
        for i in range(12):
            aus.append(au[i])
        aus.append(au[13])
        aus.append(max(au[12], au[14], au[15], au[18], key=abs))
        aus.append(au[16])
        aus.append(au[17])
        aus.append(au[19])
        au = aus
        facet = True
        del kwargs["facet"]
    if au is None:
        au = np.zeros(20)
    if "all" in kwargs:
        for muscle in muscle_names:
            todraw[muscle] = kwargs["all"]
        del kwargs["all"]
    else:
        for muscle in muscle_names:
            if muscle in kwargs:
                todraw[muscle] = kwargs[muscle]
                del kwargs[muscle]
    for muscle in todraw.keys():
        if todraw[muscle] == "heatmap":
            muscles[muscle].set_color(get_heat(muscle, au, facet))
        else:
            muscles[muscle].set_color(todraw[muscle])
        ax.add_patch(muscles[muscle], *args, **kwargs)

    eye_l = plt.Polygon(
        [
            [currx[36], curry[36]],
            [currx[37], curry[37]],
            [currx[38], curry[38]],
            [currx[39], curry[39]],
            [currx[40], curry[40]],
            [currx[41], curry[41]],
        ],
        color="w",
    )

    eye_r = plt.Polygon(
        [
            [currx[42], curry[42]],
            [currx[43], curry[43]],
            [currx[44], curry[44]],
            [currx[45], curry[45]],
            [currx[46], curry[46]],
            [currx[47], curry[47]],
        ],
        color="w",
    )

    mouth = plt.Polygon(
        [
            [currx[60], curry[60]],
            [currx[61], curry[61]],
            [currx[62], curry[62]],
            [currx[63], curry[63]],
            [currx[64], curry[64]],
            [currx[65], curry[65]],
            [currx[66], curry[66]],
            [currx[67], curry[67]],
        ],
        color="w",
    )

    ax.add_patch(eye_l)
    ax.add_patch(eye_r)
    ax.add_patch(mouth)
    return ax


def get_heat(muscle, au, log):
    """Function to create heatmap from au vector

    Args:
        muscle (string): string representation of a muscle
        au (list): vector of action units
        log (boolean): whether the action unit values are on a log scale


    Returns:
        color of muscle according to its au value
    """
    q = sns.color_palette("Blues", 151)
    unit = 0
    aus = {
        "masseter_l": 15,
        "masseter_r": 15,
        "temporalis_l": 15,
        "temporalis_r": 15,
        "dep_lab_inf_l": 14,
        "dep_lab_inf_r": 14,
        "dep_ang_or_l": 10,
        "dep_ang_or_r": 10,
        "mentalis_l": 11,
        "mentalis_r": 11,
        "risorius_l": 12,
        "risorius_r": 12,
        "frontalis_l": 1,
        "frontalis_r": 1,
        "frontalis_inner_l": 0,
        "frontalis_inner_r": 0,
        "cor_sup_l": 2,
        "cor_sup_r": 2,
        "lev_lab_sup_l": 7,
        "lev_lab_sup_r": 7,
        "lev_lab_sup_an_l": 6,
        "lev_lab_sup_an_r": 6,
        "zyg_maj_l": 8,
        "zyg_maj_r": 8,
        "bucc_l": 9,
        "bucc_r": 9,
        "orb_oc_l_outer": 4,
        "orb_oc_r_outer": 4,
        "orb_oc_l": 5,
        "orb_oc_r": 5,
        "orb_oc_l_inner": 16,
        "orb_oc_r_inner": 16,
        "orb_oris_l": 13,
        "orb_oris_u": 13,
        "pars_palp_l": 19,
        "pars_palp_r": 19,
        "masseter_l_rel": 17,
        "masseter_r_rel": 17,
        "temporalis_l_rel": 17,
        "temporalis_r_rel": 17,
    }
    if muscle in aus:
        unit = aus[muscle]
    if log:
        num = int(100 * (1.0 / (1 + 10.0 ** -(au[unit]))))
    else:
        num = int(au[unit])
    # set alpha (opacity)
    alpha = au[unit] / 100
    # color = colors.to_hex(q[num])
    # return str(color)
    color = colors.to_rgba(q[num], alpha=alpha)
    return color


def plot_face(
    au=None,
    model=None,
    vectorfield=None,
    muscles=None,
    ax=None,
    feature_range=False,
    color="k",
    linewidth=1,
    linestyle="-",
    border=True,
    gaze=None,
    muscle_scaler=None,
    *args,
    **kwargs,
):
    """Core face plotting function

    Args:
        model: (str/PLSRegression instance) Name of AU visualization model to use.
        Default's to Py-Feat's 20 AU landmark AU model
        au: vector of action units (same length as model.n_components)
        vectorfield: (dict) {'target':target_array,'reference':reference_array}
        muscles: (dict) {'muscle': color}
        ax: matplotlib axis handle
        feature_range (tuple, default: None): If a tuple with (min, max),  scale input AU intensities to (min, max) before prediction.
        color: matplotlib color
        linewidth: matplotlib linewidth
        linestyle: matplotlib linestyle
        gaze: array of gaze vectors (len(4))

    Returns:
        ax: plot handle
    """

    if model is None or isinstance(model, str):
        model = load_viz_model(model)
    else:
        if not isinstance(model, PLSRegression):
            raise ValueError("make sure that model is a PLSRegression instance")

    if au is None or isinstance(au, str) and au == "neutral":
        au = np.zeros(model.n_components)

    landmarks = predict(au, model, feature_range=feature_range)
    currx, curry = [landmarks[x, :] for x in range(2)]

    if ax is None:
        ax = _create_empty_figure()

    if muscles is not None:
        if muscles is True:
            muscles = {"all": "heatmap"}
        elif not isinstance(muscles, dict):
            raise ValueError("muscles must be a dictionary ")
        if muscle_scaler is None:
            # Muscles are always scaled 0 - 100 b/c color palette is 0-100
            au = minmax_scale(au, feature_range=(0, 100))
        elif isinstance(muscle_scaler, (int, float)):
            au = minmax_scale(au, feature_range=(0, 100 * muscle_scaler))
        else:
            au = muscle_scaler.transform(np.array(au).reshape(-1, 1)).squeeze()
        ax = draw_muscles(currx, curry, ax=ax, au=au, **muscles)

    if gaze is not None and len((gaze)) != 4:
        warnings.warn(
            "Don't forget to pass a 'gaze' vector of len(4), "
            "using neutral as default"
        )
        gaze = None

    title = kwargs.pop("title", None)
    title_kwargs = kwargs.pop(
        "title_kwargs", dict(wrap=True, fontsize=14, loc="center")
    )
    ax = draw_lineface(
        currx,
        curry,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        ax=ax,
        gaze=gaze,
        *args,
        **kwargs,
    )
    if vectorfield is not None:
        if not isinstance(vectorfield, dict):
            raise ValueError("vectorfield must be a dictionary ")
        if "reference" not in vectorfield:
            raise ValueError("vectorfield must contain 'reference' key")
        if "target" not in vectorfield.keys():
            vectorfield["target"] = landmarks
        ax = draw_vectorfield(ax=ax, **vectorfield)
    ax.set_xlim([25, 172])
    ax.set_ylim((240, 50))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if title is not None:
        if title_kwargs["wrap"]:
            title = "\n".join(wrap(title))
        _ = ax.set_title(title, **title_kwargs)
    if not border:
        sns.despine(left=True, bottom=True, ax=ax)
    return ax


def predict(au, model=None, feature_range=None):
    """Helper function to predict landmarks from au given a sklearn model

    Args:
        au: vector of action unit intensities
        model: sklearn pls object (uses pretrained model by default)
        feature_range (tuple, default: None): If a tuple with (min, max),  scale input AU intensities to (min, max) before prediction.

    Returns:
        landmarks: Array of landmarks (2,68)
    """
    if model is None:
        model = load_viz_model()
    elif not isinstance(model, PLSRegression):
        raise ValueError("make sure that model is a PLSRegression instance")

    if len(au) != model.n_components:
        print(au)
        print(model.n_components)
        raise ValueError(f"au vector must be length {model.n_components}.")

    if len(au.shape) == 1:
        au = np.reshape(au, (1, -1))

    if feature_range:
        au = minmax_scale(au, feature_range=feature_range, axis=1)

    # Handle auto-raveling feature added to PLSRegression in sklearn 1.3
    # because our model was trained in an earlier version where this attribute
    # did not exist
    # https://scikit-learn.org/stable/whats_new/v1.3.html#sklearn-cross-decomposition
    if not hasattr(model, "_predict_1d"):
        model._predict_1d = True
    landmarks = np.reshape(model.predict(au), (2, 68))
    return landmarks


def draw_facepose(pose, facebox, ax):
    """
    Draw the face pose axes on the passed image for the passed facebox.
    Adapted from draw_axis function at:
    https://github.com/natanielruiz/deep-head-pose/blob/master/code/utils.py
    Args:
      img: a PIL image
      pose: [pitch, roll, yaw] array
      facebox: [x1, x2, width, height] array
      ax = pyplot axis to draw on
    """

    # Center axis on facebox
    x1, y1, w, h = facebox[:4]
    x2, y2 = x1 + w, y1 + h
    tdx = (x1 + x2) / 2
    tdy = (y1 + y2) / 2

    # Make rotation axis lines proportional to facebox size
    size = min(x2 - x1, y2 - y1) // 2

    # Get pose axes
    pitch, roll, yaw = pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # Draw face and pose axes
    # ax.imshow(img)
    ax.plot((tdx, x1), (tdy, y1), color="red", linewidth=2)
    ax.plot((tdx, x2), (tdy, y2), color="green", linewidth=2)
    ax.plot((tdx, x3), (tdy, y3), color="blue", linewidth=2)

    return ax


def _create_empty_figure(
    figsize=(4, 5), xlim=[25, 172], ylim=[240, 50], return_fig=False
):
    """Create an empty figure"""
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if return_fig:
        return ax, fig
    return ax


def imshow(obj, figsize=(3, 3), aspect="equal"):
    """
    Convenience wrapper function around matplotlib imshow that creates figure and axis
    boilerplate for single image plotting

    Args:
        obj (str/Path/PIL.Imag): string or Path to image file or pre-loaded PIL.Image instance
        figsize (tuple, optional): matplotlib figure size. Defaults to None.
        aspect (str, optional): passed to matplotlib imshow. Defaults to "equal".
    """
    if isinstance(obj, (str, Path)):
        obj = Image.open(obj)

    _, ax = plt.subplots(figsize=figsize)
    _ = ax.imshow(obj, aspect=aspect)
    _ = ax.axis("off")


def interpolate_aus(
    start,
    end,
    num_frames,
    interp_func=None,
    num_padding_frames=None,
    include_reverse=True,
):
    """
    Helper function to interpolate between starting and ending AU values using
    non-linear easing functions

    Args:
        start (np.ndarray): array of starting intensities
        end (np.ndarray): array of ending intensities
        num_frames (int): number of frames to interpolate over
        interp_func (callable, optional): easing function. Defaults to None.
        num_padding_frames (int, optional): number of additional freeze frames to add
        before the first frame and after the last frame. Defaults to None.
        include_reverse (bool, optional): return the reverse interpolation appended to
        the end of the interpolation. Useful for animating start -> end -> start. Defaults to True.

    Returns:
        np.ndarray: frames x au 2d array
    """

    from easing_functions import CubicEaseInOut

    func = CubicEaseInOut if interp_func is None else func
    # Loop over each AU and generate a cubic bezier style interpolation from its
    # starting intensity to its ending intensity
    au_interpolations = []
    for au_start, au_end in zip(start, end):
        interp_func = func(au_start, au_end)
        intensities = [*map(interp_func, np.linspace(0, 1, num_frames))]
        au_interpolations.append(intensities)

    au_interpolations = np.column_stack(au_interpolations)

    if num_padding_frames is not None:
        begin_padding = np.tile(au_interpolations[0], (num_padding_frames, 1))
        end_padding = np.tile(au_interpolations[-1], (num_padding_frames, 1))
        au_interpolations = np.vstack([begin_padding, au_interpolations, end_padding])

    if include_reverse:
        au_interpolations = np.vstack([au_interpolations, au_interpolations[::-1, :]])
    return au_interpolations


def animate_face(
    AU=None,
    start=None,
    end=None,
    save=None,
    include_reverse=True,
    feature_range=None,
    **kwargs,
):
    """
    Create a matplotlib animation interpolating between a starting and ending face. Can
    either work like `plot_face` by taking an array of AU intensities for `start` and
    `end`, or by animating a single AU using the `AU` keyword argument and setting
    `start` and `end` to a scalar value.

    Args:
        AU (str/int, optional): action unit id (e.g. 12 or 'AU12'). Defaults to None.
        start (float/np.ndarray, optional): AU intensity to start at. Defaults to None
        which a neutral face with all AUs = 0.
        end (float/np.ndarray, optional): AU intensity(s) to end at. We don't recommend
        going beyond 3. Defaults to None.
        save (str, optional): file to save animation to. Defaults to None.
        include_reverse (bool, optional): Whether to also reverse the animation, i.e.
        start -> end -> start. Defaults to True.
        title (str, optional): plot title. Defaults to None.
        fps (int, optional): frame-rate; Defaults to 15fps
        duration (float, optional): length of animation in seconds. Defaults to 0.5
        padding (float, optional): additional time to wait in seconds on the first and
        last frame of the animation. Useful when you plan to loop the animation.
        Defaults to 0.25
        interp_func (callable, optional): interpolation function that takes a start and
        end keyword argument and returns a function that will be applied to values
        np.linspace(0, 1, num_frames); Defaults to CubicEaseInOut. See
        https://github.com/semitable/easing-functions for other options.


    Returns:
        matplotlib Animation
    """
    from celluloid import Camera

    color = kwargs.pop("color", "k")
    linewidth = kwargs.pop("linewidth", 1)
    linestyle = kwargs.pop("linestyle", "-")
    fps = kwargs.pop("fps", 15)
    duration = kwargs.pop("duration", 0.5)
    padding = kwargs.pop("padding", 0.25)
    num_frames = int(np.ceil(fps * duration))
    interp_func = kwargs.pop("interp_func", None)

    if start is None:
        start = np.zeros(20)

    if feature_range is not None:
        start = minmax_scale(start, feature_range=feature_range)
        end = minmax_scale(end, feature_range=feature_range)

    if AU is not None:
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise TypeError("If AU is specified start and end should be single numbers")

        if isinstance(AU, int):
            AU = f"AU{str(AU).zfill(2)}"
        au_map = dict(zip(AU_LANDMARK_MAP["Feat"], list(range(20))))
        au_idx = au_map[AU.upper()]
        _start, _end = np.zeros(20), np.zeros(20)
        _start[au_idx] = start
        _end[au_idx] = end
        start, end = _start, _end

    # To properly animate muscles we need to min-max scale referenced against the ending
    # AU intensities rather than the AUs of any given frame of the animation which is
    # what plot_face does if muscle_scaler is None
    if "muscles" in kwargs:
        muscle_scaler = MinMaxScaler(feature_range=(0, 100))
        # MinMaxScaler defaults to per-feature normalization so reshape like we have
        # multiple observations of a single feature
        _ = muscle_scaler.fit(np.array(end).reshape(-1, 1))
    else:
        muscle_scaler = None

    # Loop over each AU and generate a cubic bezier style interpolation from its
    # starting intensity to its ending intensity
    num_padding_frames = padding if padding is None else int(np.ceil(fps * padding))
    au_interpolations = interpolate_aus(
        start,
        end,
        interp_func=interp_func,
        num_frames=num_frames,
        num_padding_frames=num_padding_frames,
        include_reverse=include_reverse,
    )
    gaze_start = kwargs.pop("gaze_start", np.array([0, 0, 0, 0]))
    gaze_end = kwargs.pop("gaze_end", np.array([0, 0, 0, 0]))
    gaze_interpolations = interpolate_aus(
        gaze_start,
        gaze_end,
        interp_func=interp_func,
        num_frames=num_frames,
        num_padding_frames=num_padding_frames,
        include_reverse=include_reverse,
    )

    ax, fig = _create_empty_figure(return_fig=True)
    camera = Camera(fig)

    for aus, gaze in zip(au_interpolations, gaze_interpolations):
        ax = plot_face(
            model=None,
            ax=ax,
            au=aus,
            gaze=gaze,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            muscle_scaler=muscle_scaler,
            **kwargs,
        )
        # if title is not None:
        #     _ = ax.set(title=title)
        _ = camera.snap()
    animation = camera.animate()
    if save is not None:
        animation.save(save, fps=fps)
    return animation


def load_viz_model(
    file_name=None,
    prefer_joblib_if_version_match=True,
    verbose=False,
):
    """Load the h5 PLS model for plotting. Will try using joblib if python and sklearn
    major and minor versions match those the model was trained with (3.8.x and 1.0.x
    respectively), otherwise will reconstruct the model object using h5 data.

    Args:
        file_name (str, optional): Specify model to load.. Defaults to 'blue.h5'.
        prefer_joblib_if_version_match (bool, optional): If the sklearn and python major.minor versions
        match then return the pickled PLSRegression object. Otherwise build it from
        scratch using .h5 data. Default True

    Returns:
        model: PLS model
    """

    file_name = "pyfeat_aus_to_landmarks" if file_name is None else file_name

    if "." in file_name:
        raise TypeError("Please use a file name with no extension")

    h5_path = os.path.join(get_resource_path(), f"{file_name}.h5")
    joblib_path = os.path.join(get_resource_path(), f"{file_name}.joblib")

    # Make sure saved viz model exists
    if not os.path.exists(h5_path) or not os.path.exists(joblib_path):
        with open(os.path.join(get_resource_path(), "model_list.json"), "r") as f:
            model_urls = json.load(f)
            urls = model_urls["viz_models"][file_name]["urls"]
            for url in urls:
                download_url(url, get_resource_path(), verbose=verbose)

    # Check sklearn and python version to see if we can load joblib
    my_skmajor, my_skminor, *my_skpatch = skversion.split(".")
    my_pymajor, my_pyminor, *my_pymicro = sys.version_info

    # Versions viz models were trained with
    pymajor, pyminor, skmajor, skminor = 3, 8, 1, 1
    if (
        int(my_skmajor) == skmajor
        and int(my_skminor) == skminor
        and int(my_pymajor) == pymajor
        and int(my_pyminor) == pyminor
        and prefer_joblib_if_version_match
    ):
        can_load_joblib = True
    else:
        can_load_joblib = False

    try:
        if can_load_joblib:
            if verbose:
                print("Loading joblib")
            model = load(joblib_path)
            # We need the h5 file for some meta-data even when loading using joblib
            hf = h5py.File(h5_path, mode="r")
            model.__dict__["model_name_"] = hf.attrs["model_name"]
            model.__dict__["skversion"] = hf.attrs["skversion"]
            model.__dict__["pyversion"] = hf.attrs["pyversion"]
            hf.close()
        else:
            if verbose:
                print("Reconstructing from h5")
            hf = h5py.File(h5_path, mode="r")
            x_weights = np.array(hf.get("x_weights"))
            model = PLSRegression(n_components=x_weights.shape[1])
            # PLSRegression in sklearn < 1.1 storex coefs as samples x features, but
            # recent versions transpose this. Check if the user is on Python 3.7 (which
            # only supports sklearn 1.0.x) or < sklearn 1.1.x
            if (my_pymajor == 3 and my_pyminor == 7) or (
                my_skmajor == 1 and my_skminor != 1
            ):
                model.__dict__["coef_"] = np.array(hf.get("coef"))
                model.__dict__["_coef_"] = np.array(hf.get("coef"))
            else:
                model.__dict__["coef_"] = np.array(hf.get("coef")).T
                model.__dict__["_coef_"] = np.array(hf.get("coef")).T
            model.__dict__["x_weights_"] = np.array(hf.get("x_weights"))
            model.__dict__["y_weights_"] = np.array(hf.get("y_weights"))
            model.__dict__["x_loadings"] = np.array(hf.get("x_loadings"))
            model.__dict__["y_loadings"] = np.array(hf.get("y_loadings"))
            model.__dict__["x_scores"] = np.array(hf.get("x_scores"))
            model.__dict__["y_scores"] = np.array(hf.get("y_scores"))
            model.__dict__["x_rotations"] = np.array(hf.get("x_rotations"))
            model.__dict__["y_rotations"] = np.array(hf.get("y_rotations"))
            model.__dict__["intercept"] = np.array(hf.get("intercept"))
            model.__dict__["x_train"] = np.array(hf.get("X_train"))
            model.__dict__["y_train"] = np.array(hf.get("Y_train"))
            model.__dict__["X_train"] = np.array(hf.get("x_train"))
            model.__dict__["Y_train"] = np.array(hf.get("y_train"))
            model.__dict__["intercept_"] = np.array(hf.get("intercept"))
            model.__dict__["model_name_"] = hf.attrs["model_name"]
            model.__dict__["skversion"] = hf.attrs["skversion"]
            model.__dict__["pyversion"] = hf.attrs["pyversion"]

            # Older sklearn version named these attributes differently
            if int(skversion.split(".")[0]) < 1:
                model.__dict__["x_mean_"] = np.array(hf.get("x_mean"))
                model.__dict__["y_mean_"] = np.array(hf.get("y_mean"))
                model.__dict__["x_std_"] = np.array(hf.get("x_std"))
                model.__dict__["y_std_"] = np.array(hf.get("y_std"))
            else:
                model.__dict__["_x_mean"] = np.array(hf.get("x_mean"))
                model.__dict__["_y_mean"] = np.array(hf.get("y_mean"))
                model.__dict__["_x_std"] = np.array(hf.get("x_std"))
                model.__dict__["_y_std"] = np.array(hf.get("y_std"))
            hf.close()
    except Exception as e:
        raise IOError(f"Unable to load data: {e}")
    return model


def face_part_path(row, img_height, line_points):
    """Helper function to draw SVG path for a specific face part. Requires list of landmark point positions (i.e., [0,1,2]). Last coordinate is end point

    Args:
        row: (FexSeries) a row of a Fex object
        img_height (int): the height of the image
        line_points (list): a list of points on a landmark (i.e., [0:68])

    Returns:
        fig (str): an SVG string
    """

    path = ""
    counter = 0
    for i in line_points:
        x = row[f"x_{i}"]
        y = img_height - row[f"y_{i}"]
        if counter == 0:
            path += f"M {x},{y}"
            counter += 1
        else:
            path += f"L {x},{y}"
    path += " Z"
    return path


def draw_plotly_landmark(
    row, img_height, fig, line_width=3, line_color="white", output="dictionary"
):
    """Helper function to draw an SVG path for a plotly figure object

    Args:
        row: (FexSeries) a row of a Fex object
        img_height (int): height of the image to flip the y-coordinates
        fig: a plotly figure instance
        output (str): type of output "figure" for plotly figure object or "dictionary"
        line_width (int): (optional) line width if outputting a plotly figure instance
        line_color (int): (optional) line color if outputting a plotly figure instance

    Returns:
        fig (str): an SVG string
    """

    path = ""

    # Face outline
    path += face_part_path(
        row,
        img_height,
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
        ],
    )

    # Left Eye
    path += face_part_path(row, img_height, [36, 37, 38, 39, 40, 41])

    # Right Eye
    path += face_part_path(row, img_height, [42, 43, 44, 45, 46, 47])

    # Left Eyebrow
    path += face_part_path(row, img_height, [17, 18, 19, 20, 21, 20, 19, 18, 17])

    # Right Eyebrow
    path += face_part_path(row, img_height, [22, 23, 24, 25, 26, 25, 24, 23, 22])

    # Lips1
    path += face_part_path(
        row, img_height, [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60, 48]
    )

    # Lips2
    path += face_part_path(
        row, img_height, [48, 60, 67, 66, 65, 64, 54, 55, 56, 57, 58, 59, 48]
    )

    # Nose 1
    path += face_part_path(row, img_height, [27, 28, 29, 30, 29, 28, 27])

    # Nose 2
    path += face_part_path(row, img_height, [31, 32, 33, 34, 35, 34, 33, 32, 31])

    if output == "figure":
        # Draw figure
        fig.add_shape(
            type="path", path=path, line_color=line_color, line_width=line_width
        )

        return fig

    elif output == "dictionary":
        return dict(
            type="path", path=path, line=dict(color=line_color, width=line_width)
        )

    else:
        raise ValueError('output can only be ["figure","dictionary"]')


def face_polygon_svg(line_points, img_height):
    """Helper function to draw SVG path for a polygon of a specific face part. Requires list of landmark x,y coordinate tuples (i.e., [(2,2),(5,33)]).

    Args:
        line_points (list): a list of tuples of landmark coordinates
        img_height (int): height of the image to flip the y-coordinates

    Returns:
        fig (str): an SVG string
    """

    path = ""
    counter = 0
    for x, y in line_points:
        y = img_height - y
        if counter == 0:
            path += f"M {x},{y}"
            counter += 1
        else:
            path += f"L {x},{y}"
    path += " Z"
    return path


def draw_plotly_au(
    row,
    img_height,
    fig,
    heatmap_resolution=1000,
    au_opacity=0.9,
    cmap="Blues",
    output="dictionary",
):
    """Helper function to draw an SVG path for a plotly figure object

        NOTES:
            Need to clean up muscle ids after looking at face anatomy action units

    Args:
        row (FexSeries): FexSeries instance
        img_height (int): height of image overlay. used to adjust coordinates
        fig: plotly figure handle
        heatmap_resolution (int): precision of cmap
        au_opacity (float): amount of opacity for face muscles
        cmap (str): colormap
        output (str): type of output "figure" for plotly figure object or "dictionary"

    Returns:
        fig: plotly figure handle
    """

    muscle_au_dict = {
        "masseter_l": 15,
        "masseter_r": 15,
        "temporalis_l": 15,
        "temporalis_r": 15,
        "dep_lab_inf_l": 14,
        "dep_lab_inf_r": 14,
        "dep_ang_or_l": 10,
        "dep_ang_or_r": 10,
        "mentalis_l": 11,
        "mentalis_r": 11,
        "risorius_l": 12,
        "risorius_r": 12,
        "frontalis_l": 1,
        "frontalis_r": 1,
        "frontalis_inner_l": 0,
        "frontalis_inner_r": 0,
        "cor_sup_l": 2,
        "cor_sup_r": 2,
        "lev_lab_sup_l": 7,
        "lev_lab_sup_r": 7,
        "lev_lab_sup_an_l": 6,
        "lev_lab_sup_an_r": 6,
        "zyg_maj_l": 8,
        "zyg_maj_r": 8,
        "bucc_l": 9,
        "bucc_r": 9,
        "orb_oc_l_outer": 4,
        "orb_oc_r_outer": 4,
        "orb_oc_l": 5,
        "orb_oc_r": 5,
        "orb_oc_l_inner": 16,
        "orb_oc_r_inner": 16,
        "orb_oris_l": 13,
        "orb_oris_u": 13,
    }

    #     muscle_au_dict = {"masseter_l": 15,
    #                         "masseter_r": 15,
    #                         "temporalis_l": 15,
    #                         "temporalis_r": 15,
    #                         "dep_lab_inf_l": 14,
    #                         "dep_lab_inf_r": 14,
    #                         "dep_ang_or_l": 10,
    #                         "dep_ang_or_r": 10,
    #                         "mentalis_l": 11,
    #                         "mentalis_r": 11,
    #                         "risorius_l": 12,
    #                         "risorius_r": 12,
    #                         "frontalis_l": 1,
    #                         "frontalis_r": 1,
    #                         "frontalis_inner_l": 0,
    #                         "frontalis_inner_r": 0,
    #                         "cor_sup_l": 2,
    #                         "cor_sup_r": 2,
    #                         "lev_lab_sup_l": 7,
    #                         "lev_lab_sup_r": 7,
    #                         "lev_lab_sup_an_l": 6,
    #                         "lev_lab_sup_an_r": 6,
    #                         "zyg_maj_l": 8,
    #                         "zyg_maj_r": 8,
    #                         "bucc_l": 9,
    #                         "bucc_r": 9,
    #                         "orb_oc_l_outer": 4,
    #                         "orb_oc_r_outer": 4,
    #                         "orb_oc_l": 5,
    #                         "orb_oc_r": 5,
    #                         "orb_oc_l_inner": 16,
    #                         "orb_oc_r_inner": 16,
    #                         "orb_oris_l": 13,
    #                         "orb_oris_u": 13}
    # #                         "pars_palp_l": 19,
    # #                         "pars_palp_r": 19,
    # #                         "masseter_l_rel": 17,
    # #                         "masseter_r_rel": 17,
    # #                         "temporalis_l_rel": 17,
    # #                         "temporalis_r_rel": 17}

    aus = [
        "AU01",
        "AU02",
        "AU04",
        "AU05",
        "AU06",
        "AU07",
        "AU09",
        "AU10",
        "AU11",
        "AU12",
        "AU14",
        "AU15",
        "AU17",
        "AU20",
        "AU23",
        "AU24",
        "AU25",
        "AU26",
        "AU28",
        "AU43",
    ]

    masseter_l = face_polygon_svg(
        [
            (row["x_2"], row["y_2"]),
            (row["x_3"], row["y_3"]),
            (row["x_4"], row["y_4"]),
            (row["x_5"], row["y_5"]),
            (row["x_6"], row["y_6"]),
            (row["x_5"], row["y_33"]),
        ],
        img_height,
    )

    masseter_r = face_polygon_svg(
        [
            (row["x_14"], row["y_14"]),
            (row["x_13"], row["y_13"]),
            (row["x_12"], row["y_12"]),
            (row["x_11"], row["y_11"]),
            (row["x_10"], row["y_10"]),
            (row["x_11"], row["y_33"]),
        ],
        img_height,
    )

    temporalis_l = face_polygon_svg(
        [
            (row["x_2"], row["y_2"]),
            (row["x_1"], row["y_1"]),
            (row["x_0"], row["y_0"]),
            (row["x_17"], row["y_17"]),
            (row["x_36"], row["y_36"]),
        ],
        img_height,
    )

    temporalis_r = face_polygon_svg(
        [
            (row["x_14"], row["y_14"]),
            (row["x_15"], row["y_15"]),
            (row["x_16"], row["y_16"]),
            (row["x_26"], row["y_26"]),
            (row["x_45"], row["y_45"]),
        ],
        img_height,
    )

    dep_lab_inf_l = face_polygon_svg(
        [
            (row["x_57"], row["y_57"]),
            (row["x_58"], row["y_58"]),
            (row["x_59"], row["y_59"]),
            (row["x_6"], row["y_6"]),
            (row["x_7"], row["y_7"]),
        ],
        img_height,
    )

    dep_lab_inf_r = face_polygon_svg(
        [
            (row["x_57"], row["y_57"]),
            (row["x_56"], row["y_56"]),
            (row["x_55"], row["y_55"]),
            (row["x_10"], row["y_10"]),
            (row["x_9"], row["y_9"]),
        ],
        img_height,
    )

    dep_ang_or_l = face_polygon_svg(
        [
            (row["x_48"], row["y_48"]),
            (row["x_7"], row["y_7"]),
            (row["x_6"], row["y_6"]),
        ],
        img_height,
    )

    dep_ang_or_r = face_polygon_svg(
        [
            (row["x_54"], row["y_54"]),
            (row["x_9"], row["y_9"]),
            (row["x_10"], row["y_10"]),
        ],
        img_height,
    )

    mentalis_l = face_polygon_svg(
        [
            (row["x_58"], row["y_58"]),
            (row["x_7"], row["y_7"]),
            (row["x_8"], row["y_8"]),
        ],
        img_height,
    )

    mentalis_r = face_polygon_svg(
        [
            (row["x_56"], row["y_56"]),
            (row["x_9"], row["y_9"]),
            (row["x_8"], row["y_8"]),
        ],
        img_height,
    )

    risorius_l = face_polygon_svg(
        [
            (row["x_4"], row["y_4"]),
            (row["x_5"], row["y_5"]),
            (row["x_48"], row["y_48"]),
        ],
        img_height,
    )

    risorius_r = face_polygon_svg(
        [
            (row["x_11"], row["y_11"]),
            (row["x_12"], row["y_12"]),
            (row["x_54"], row["y_54"]),
        ],
        img_height,
    )

    bottom = (row["y_8"] - row["y_57"]) / 2

    orb_oris_l = face_polygon_svg(
        [
            (row["x_48"], row["y_48"]),
            (row["x_59"], row["y_59"]),
            (row["x_58"], row["y_58"]),
            (row["x_57"], row["y_57"]),
            (row["x_56"], row["y_56"]),
            (row["x_55"], row["y_55"] + bottom),
            (row["x_54"], row["y_54"] + bottom),
            (row["x_55"], row["y_55"] + bottom),
            (row["x_56"], row["y_56"] + bottom),
            (row["x_57"], row["y_57"] + bottom),
            (row["x_58"], row["y_58"] + bottom),
            (row["x_59"], row["y_59"] + bottom),
        ],
        img_height,
    )

    orb_oris_u = face_polygon_svg(
        [
            (row["x_48"], row["y_48"]),
            (row["x_49"], row["y_49"]),
            (row["x_50"], row["y_50"]),
            (row["x_51"], row["y_51"]),
            (row["x_52"], row["y_52"]),
            (row["x_53"], row["y_53"]),
            (row["x_54"], row["y_54"]),
            (row["x_33"], row["y_33"]),
        ],
        img_height,
    )

    frontalis_l = face_polygon_svg(
        [
            (row["x_27"], row["y_27"]),
            (row["x_39"], row["y_39"]),
            (row["x_38"], row["y_38"]),
            (row["x_37"], row["y_37"]),
            (row["x_36"], row["y_36"]),
            (row["x_17"], row["y_17"]),
            (row["x_18"], row["y_18"]),
            (row["x_19"], row["y_19"]),
            (row["x_20"], row["y_20"]),
            (row["x_21"], row["y_21"]),
        ],
        img_height,
    )

    frontalis_r = face_polygon_svg(
        [
            (row["x_27"], row["y_27"]),
            (row["x_22"], row["y_22"]),
            (row["x_23"], row["y_23"]),
            (row["x_24"], row["y_24"]),
            (row["x_25"], row["y_25"]),
            (row["x_26"], row["y_26"]),
            (row["x_45"], row["y_45"]),
            (row["x_44"], row["y_44"]),
            (row["x_43"], row["y_43"]),
            (row["x_42"], row["y_42"]),
        ],
        img_height,
    )

    frontalis_inner_l = face_polygon_svg(
        [
            (row["x_27"], row["y_27"]),
            (row["x_39"], row["y_39"]),
            (row["x_21"], row["y_21"]),
        ],
        img_height,
    )

    frontalis_inner_r = face_polygon_svg(
        [
            (row["x_27"], row["y_27"]),
            (row["x_42"], row["y_42"]),
            (row["x_22"], row["y_22"]),
        ],
        img_height,
    )

    cor_sup_l = face_polygon_svg(
        [
            (row["x_28"], row["y_28"]),
            (row["x_19"], row["y_19"]),
            (row["x_20"], row["y_20"]),
        ],
        img_height,
    )

    cor_sup_r = face_polygon_svg(
        [
            (row["x_28"], row["y_28"]),
            (row["x_23"], row["y_23"]),
            (row["x_24"], row["y_24"]),
        ],
        img_height,
    )

    lev_lab_sup_l = face_polygon_svg(
        [
            (row["x_41"], row["y_41"]),
            (row["x_40"], row["y_40"]),
            (row["x_49"], row["y_49"]),
        ],
        img_height,
    )

    lev_lab_sup_r = face_polygon_svg(
        [
            (row["x_47"], row["y_47"]),
            (row["x_46"], row["y_46"]),
            (row["x_53"], row["y_53"]),
        ],
        img_height,
    )

    lev_lab_sup_an_l = face_polygon_svg(
        [
            (row["x_39"], row["y_39"]),
            (row["x_49"], row["y_49"]),
            (row["x_31"], row["y_31"]),
        ],
        img_height,
    )

    lev_lab_sup_an_r = face_polygon_svg(
        [
            (row["x_35"], row["y_35"]),
            (row["x_42"], row["y_42"]),
            (row["x_53"], row["y_53"]),
        ],
        img_height,
    )

    zyg_maj_l = face_polygon_svg(
        [
            (row["x_48"], row["y_48"]),
            (row["x_3"], row["y_3"]),
            (row["x_2"], row["y_2"]),
        ],
        img_height,
    )

    zyg_maj_r = face_polygon_svg(
        [
            (row["x_54"], row["y_54"]),
            (row["x_13"], row["y_13"]),
            (row["x_14"], row["y_14"]),
        ],
        img_height,
    )

    bucc_l = face_polygon_svg(
        [
            (row["x_48"], row["y_48"]),
            (row["x_5"], row["y_50"]),
            (row["x_5"], row["y_57"]),
        ],
        img_height,
    )

    bucc_r = face_polygon_svg(
        [
            (row["x_54"], row["y_54"]),
            (row["x_11"], row["y_52"]),
            (row["x_11"], row["y_57"]),
        ],
        img_height,
    )

    width_l = (row["y_21"] - row["y_39"]) / 2

    orb_oc_l = face_polygon_svg(
        [
            (row["x_36"] - width_l / 3, row["y_36"] + width_l / 2),
            (row["x_36"], row["y_36"] + width_l),
            (row["x_37"], row["y_37"] + width_l),
            (row["x_38"], row["y_38"] + width_l),
            (row["x_39"], row["y_39"] + width_l),
            (row["x_39"] + width_l / 3, row["y_39"] + width_l / 2),
            (row["x_39"] + width_l / 2, row["y_39"]),
            (row["x_39"] + width_l / 3, row["y_39"] - width_l / 2),
            (row["x_39"], row["y_39"] - width_l),
            (row["x_40"], row["y_40"] - width_l),
            (row["x_41"], row["y_41"] - width_l),
            (row["x_36"], row["y_36"] - width_l),
            (row["x_36"] - width_l / 3, row["y_36"] - width_l / 2),
            (row["x_36"] - width_l / 2, row["y_36"]),
        ],
        img_height,
    )

    orb_oc_l_inner = face_polygon_svg(
        [
            (row["x_36"] - width_l / 6, row["y_36"] + width_l / 5),
            (row["x_36"], row["y_36"] + width_l / 2),
            (row["x_37"], row["y_37"] + width_l / 2),
            (row["x_38"], row["y_38"] + width_l / 2),
            (row["x_39"], row["y_39"] + width_l / 2),
            (row["x_39"] + width_l / 6, row["y_39"] + width_l / 5),
            (row["x_39"] + width_l / 5, row["y_39"]),
            (row["x_39"] + width_l / 6, row["y_39"] - width_l / 5),
            (row["x_39"], row["y_39"] - width_l),
            (row["x_40"], row["y_40"] - width_l),
            (row["x_41"], row["y_41"] - width_l),
            (row["x_36"], row["y_36"] - width_l),
            (row["x_36"] - width_l / 6, row["y_36"] - width_l / 5),
            (row["x_36"] - width_l / 5, row["y_36"]),
        ],
        img_height,
    )

    width_l2 = (row["y_38"] - row["y_2"]) / 1.5

    orb_oc_l_outer = face_polygon_svg(
        [
            (row["x_39"] + width_l / 2, row["y_39"] + width_l / 2),
            (row["x_39"], row["y_39"] - width_l),
            (row["x_40"], row["y_40"] - width_l2),
            (row["x_41"], row["y_41"] - width_l2),
            (row["x_36"], row["y_36"] - width_l2),
            (row["x_36"] - width_l2 / 3, row["y_36"] - width_l2 / 2),
            (row["x_36"] - width_l / 2, row["y_36"]),
        ],
        img_height,
    )

    width_r = (row["y_23"] - row["y_43"]) / 2

    orb_oc_r = face_polygon_svg(
        [
            (row["x_42"] - width_r / 3, row["y_42"] + width_r / 2),
            (row["x_42"], row["y_42"] + width_r),
            (row["x_43"], row["y_43"] + width_r),
            (row["x_44"], row["y_44"] + width_r),
            (row["x_45"], row["y_45"] + width_r),
            (row["x_45"] + width_r / 3, row["y_45"] + width_r / 2),
            (row["x_45"] + width_r / 2, row["y_45"]),
            (row["x_45"] + width_r / 3, row["y_45"] - width_r / 2),
            (row["x_45"], row["y_45"] - width_r),
            (row["x_46"], row["y_46"] - width_r),
            (row["x_47"], row["y_47"] - width_r),
            (row["x_42"], row["y_42"] - width_r),
            (row["x_42"] - width_l / 3, row["y_42"] - width_r / 2),
            (row["x_42"] - width_r / 2, row["y_42"]),
        ],
        img_height,
    )

    orb_oc_r_inner = face_polygon_svg(
        [
            (row["x_42"] - width_r / 6, row["y_42"] + width_r / 5),
            (row["x_42"], row["y_42"] + width_r / 2),
            (row["x_43"], row["y_43"] + width_r / 2),
            (row["x_44"], row["y_44"] + width_r / 2),
            (row["x_45"], row["y_45"] + width_r / 2),
            (row["x_45"] + width_r / 6, row["y_45"] + width_r / 5),
            (row["x_45"] + width_r / 5, row["y_45"]),
            (row["x_45"] + width_r / 6, row["y_45"] - width_r / 5),
            (row["x_45"], row["y_45"] - width_r / 2),
            (row["x_46"], row["y_46"] - width_r / 2),
            (row["x_47"], row["y_47"] - width_r / 2),
            (row["x_42"], row["y_42"] - width_r / 2),
            (row["x_42"] - width_l / 6, row["y_42"] - width_r / 5),
            (row["x_42"] - width_r / 5, row["y_42"]),
        ],
        img_height,
    )

    width_r2 = (row["y_44"] - row["y_14"]) / 1.5

    orb_oc_r_outer = face_polygon_svg(
        [
            (row["x_42"] - width_r / 2, row["y_42"]),
            (row["x_47"], row["y_47"] - width_r2),
            (row["x_46"], row["y_46"] - width_r2),
            (row["x_45"], row["y_45"] - width_r2),
            (row["x_45"] + width_r2 / 3, row["y_45"] - width_r2 / 2),
            (row["x_45"] + width_r / 2, row["y_45"]),
        ],
        img_height,
    )

    eye_l = face_polygon_svg(
        [
            (row["x_36"], row["y_36"]),
            (row["x_37"], row["y_37"]),
            (row["x_38"], row["y_38"]),
            (row["x_39"], row["y_39"]),
            (row["x_40"], row["y_40"]),
            (row["x_41"], row["y_41"]),
        ],
        img_height,
    )

    eye_r = face_polygon_svg(
        [
            (row["x_42"], row["y_42"]),
            (row["x_43"], row["y_43"]),
            (row["x_44"], row["y_44"]),
            (row["x_45"], row["y_45"]),
            (row["x_46"], row["y_46"]),
            (row["x_47"], row["y_47"]),
        ],
        img_height,
    )

    # Outside Mouth
    #     mouth = face_polygon_path([(row['x_48'],row['y_48']),
    #                                (row['x_49'],row['y_49']),
    #                                (row['x_50'],row['y_50']),
    #                                (row['x_51'],row['y_51']),
    #                                (row['x_52'],row['y_52']),
    #                                (row['x_53'],row['y_53']),
    #                                (row['x_54'],row['y_54']),
    #                                (row['x_55'],row['y_55']),
    #                                (row['x_56'],row['y_56']),
    #                                (row['x_57'],row['y_57']),
    #                                (row['x_58'],row['y_58']),
    #                                (row['x_59'],row['y_59'])], img_height)
    # Inside Mouth
    mouth = face_polygon_svg(
        [
            (row["x_60"], row["y_60"]),
            (row["x_61"], row["y_61"]),
            (row["x_62"], row["y_62"]),
            (row["x_63"], row["y_63"]),
            (row["x_64"], row["y_64"]),
            (row["x_65"], row["y_65"]),
            (row["x_66"], row["y_66"]),
            (row["x_67"], row["y_67"]),
        ],
        img_height,
    )

    pupil_l = [
        (
            (
                row["x_36"]
                + row["x_37"]
                + row["x_38"]
                + row["x_40"]
                + row["x_41"]
                + row["x_39"]
            )
            / 6,
            (
                img_height
                - (
                    row["y_36"]
                    + row["y_37"]
                    + row["y_38"]
                    + row["y_40"]
                    + row["y_41"]
                    + row["y_39"]
                )
                / 6
            ),
        ),
        (
            (row["x_38"] + row["x_40"]) / 2,
            (img_height - (row["y_37"] + row["y_38"]) / 2),
        ),
    ]
    pupil_r = [
        (
            (row["x_43"] + row["x_44"] + row["x_46"] + row["x_47"]) / 4,
            (img_height - (row["y_43"] + row["y_44"] + row["y_46"] + row["y_47"]) / 4),
        ),
        (
            (row["x_44"] + row["x_46"]) / 2,
            (img_height - (row["y_43"] + row["y_44"]) / 2),
        ),
    ]

    # Build AU heatmap
    cmap = sns.color_palette(cmap, heatmap_resolution + 1)

    if output == "figure":
        for muscle in list(muscle_au_dict.keys()):
            color = cmap.as_hex()[
                int(row[aus[muscle_au_dict[muscle]]] * heatmap_resolution)
            ]
            fig.add_shape(
                type="path",
                path=eval(muscle),
                line_color=color,
                fillcolor=color,
                opacity=au_opacity,
            )

            for region in [eye_l, eye_r, mouth]:
                fig.add_shape(
                    type="path",
                    path=region,
                    line_color="black",
                    line_width=2,
                    fillcolor="white",
                )

            for pupil in [pupil_l, pupil_r]:
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    fillcolor="black",
                    x0=pupil[0][0],
                    y0=pupil[0][1],
                    x1=pupil[1][0],
                    y1=pupil[1][1],
                    line_color="black",
                    line_width=3,
                )

        return fig

    elif output == "dictionary":
        muscles = []
        for muscle in list(muscle_au_dict.keys()):
            if isinstance(row[aus[muscle_au_dict[muscle]]], (np.nan)):
                au_intensity = 0
            else:
                au_intensity = int(row[aus[muscle_au_dict[muscle]]])
            color = cmap.as_hex()[au_intensity * heatmap_resolution]

            muscles.append(
                dict(
                    type="path",
                    path=eval(muscle),
                    fillcolor=color,
                    opacity=au_opacity,
                    line=dict(color=color),
                )
            )

        regions = []
        for region in [eye_l, eye_r, mouth]:
            regions.append(
                dict(
                    type="path",
                    path=region,
                    line_width=2,
                    fillcolor="white",
                    line=dict(color="black"),
                )
            )

        pupils = []
        for pupil in [pupil_l, pupil_r]:
            pupils.append(
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    fillcolor="black",
                    x0=pupil[0][0],
                    y0=pupil[0][1],
                    x1=pupil[1][0],
                    y1=pupil[1][1],
                    line_width=3,
                    line=dict(color="black"),
                )
            )
        return flatten_list([muscles, regions, pupils])

    else:
        raise ValueError('output can only be ["figure","dictionary"]')


def draw_plotly_pose(row, img_height, fig, line_width=2, output="dictionary"):
    """
    Helper function to draw a path indicating the x,y,z pose position.

    Args:
        row (FexSeries): FexSeries instance
        img_height (int): height of image overlay. used to adjust coordinates
        fig: plotly figure handle
        line_width (int): (optional) width of line if outputing a plotly figure instance
        output (str): type of output "figure" for plotly figure object or "dictionary"

    Returns:
        fig: plotly figure handle
    """

    # Center axis on facebox
    x1, y1, w, h = row[["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]]
    x2, y2 = x1 + w, y1 + h
    tdx = (x1 + x2) / 2
    tdy = (y1 + y2) / 2

    # Make rotation axis lines proportional to facebox size
    size = min(x2 - x1, y2 - y1) // 2

    # Get pose axes
    pitch, roll, yaw = row[["Pitch", "Roll", "Yaw"]]
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = (
        size
        * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + tdy
    )

    # Y-Axis | drawn in green
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = (
        size
        * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
        + tdy
    )

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    # Flip y coordinates
    tdy, y1, y2, y3 = [img_height - c for c in [tdy, y1, y2, y3]]

    if output == "figure":
        # Draw face and pose axes
        fig.add_shape(
            type="line",
            x0=tdx,
            y0=tdy,
            x1=x1,
            y1=y1,
            line=dict(color="red", width=line_width),
        )
        fig.add_shape(
            type="line",
            x0=tdx,
            y0=tdy,
            x1=x2,
            y1=y2,
            line=dict(color="green", width=line_width),
        )
        fig.add_shape(
            type="line",
            x0=tdx,
            y0=tdy,
            x1=x3,
            y1=y3,
            line=dict(color="blue", width=line_width),
        )
        return fig

    elif output == "dictionary":
        return [
            dict(type="line", x0=tdx, y0=tdy, x1=x1, y1=y1, line=dict(color="red")),
            dict(type="line", x0=tdx, y0=tdy, x1=x2, y1=y2, line=dict(color="green")),
            dict(type="line", x0=tdx, y0=tdy, x1=x3, y1=y3, line=dict(color="blue")),
        ]
    #     return [dict(type='line', x0=tdx, y0=tdy, x1=x1, y1=y1, line_color="red", width=line_width),
    #                 dict(type='line', x0=tdx, y0=tdy, x1=x2, y1=y2, line_color="green", width=line_width),
    #                 dict(type='line', x0=tdx, y0=tdy, x1=x3, y1=y3, line_color="blue", width=line_width)]

    else:
        raise ValueError('output can only be ["figure","dictionary"]')


def emotion_annotation_position(
    row, img_height, img_width, emotions_size=12, emotions_position="bottom"
):
    """Helper function to adjust position of emotion annotations

    Args:
        row (FexSeries): FexSeries instance
        img_height (int): height of image overlay. used to adjust coordinates
        img_width (int): width of image overlay. used to adjust coordinates
        emotions_size (int): size of text used to adjust positions
        emotions_position (str): position to place emotion annotations ['left', 'right', 'top', 'bottom']

    Returns:
        x_position (int):
        y_position (int):
        align (str): plotly annotation text alignment ['top','bottom', 'left', 'right ]
        valign (str): plotly annotation vertical alignment ['middle', 'top', 'bottom']
    """

    y_spacing = img_height * 0.01 * emotions_size * 0.5
    x_spacing = img_width * 0.02 * emotions_size * 0.18

    if emotions_position.lower() == "bottom":
        x_position = row["FaceRectX"] + row["FaceRectWidth"] / 2
        y_position = (
            img_height
            - row["FaceRectY"]
            - row["FaceRectHeight"]
            - img_height * 0.04
            - y_spacing
        )
        align = "left"
        valign = "bottom"
    elif emotions_position.lower() == "top":
        x_position = row["FaceRectX"] + row["FaceRectWidth"] / 2
        y_position = (
            img_height
            - row["FaceRectY"]
            + row["FaceRectHeight"] / 2
            + y_spacing
            - img_height * 0.04
        )
        align = "left"
        valign = "bottom"
    elif emotions_position.lower() == "right":
        x_position = (
            row["FaceRectX"] + row["FaceRectWidth"] + img_width * 0.025 + x_spacing
        )
        y_position = img_height - row["FaceRectY"] - row["FaceRectHeight"] / 2
        align = "left"
        valign = "middle"
    elif emotions_position.lower() == "left":
        x_position = (
            row["FaceRectX"] - row["FaceRectWidth"] / 2 - x_spacing + img_width * 0.01
        )
        y_position = img_height - row["FaceRectY"] - row["FaceRectHeight"] / 2
        align = "right"
        valign = "middle"
    else:
        raise ValueError(
            '"emotions_position" must be one of ["bottom","top","left","right"]'
        )

    return (x_position, y_position, align, valign)
