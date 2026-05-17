"""
Helper functions for plotting
"""

import os
import sys
import h5py
import torch
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn import __version__ as skversion
import matplotlib.pyplot as plt
from feat.pretrained import AU_LANDMARK_MAP
from feat.utils.io import get_resource_path, download_url
from feat.utils.image_operations import (
    align_face,
    mask_image,
    procrustes_align_2d_batched,
)
from feat.utils import flatten_list
from huggingface_hub import hf_hub_download
from math import sin, cos
import warnings
import seaborn as sns
import matplotlib.colors as colors
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from pathlib import Path
from PIL import Image
from textwrap import wrap
from joblib import load
import json
from skimage.morphology.convex_hull import grid_points_in_poly
from scipy.spatial import ConvexHull
import torchvision.transforms as transforms
from torchvision.utils import draw_keypoints, draw_bounding_boxes, make_grid

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


# dlib-68 mirror pairs (anatomical left ↔ right). Indices not listed here
# sit on the facial midline and stay put under mirror-averaging.
_DLIB68_MIRROR_PAIRS = (
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    (31, 35), (32, 34),
    (48, 54), (49, 53), (50, 52), (59, 55), (58, 56), (60, 64), (61, 63), (67, 65),
)
_DLIB68_MIDLINE_IDX = (27, 28, 29, 30, 33, 51, 62, 66, 57, 8)


def _symmetrize_dlib68(landmarks):
    """Mirror-average dlib-68 landmarks around the facial midline.

    The v2 ``PLSAULandmarkModel`` (PR #301) was trained on real CelebV-HQ
    faces and absorbs ~5-8% mirror-asymmetry even for symmetric AU input.
    Since none of py-feat's 20 AUs are inherently lateralized, post-hoc
    averaging around the midline restores the geometric symmetry users
    expect from a visualization tool. v1 ``PLSRegression`` output is
    already symmetric to ~0.5%, so this is essentially a no-op for v1.
    """
    out = np.asarray(landmarks, dtype=np.float32).copy()
    x, y = out[0], out[1]
    midline_x = float(np.mean([x[i] for i in _DLIB68_MIDLINE_IDX]))
    for i in _DLIB68_MIDLINE_IDX:
        x[i] = midline_x
    for left, right in _DLIB68_MIRROR_PAIRS:
        avg_x = (x[left] + (2 * midline_x - x[right])) / 2
        avg_y = (y[left] + y[right]) / 2
        x[left] = avg_x
        x[right] = 2 * midline_x - avg_x
        y[left] = avg_y
        y[right] = avg_y
    return out


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
    symmetrize=True,
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
        if not isinstance(model, (PLSRegression, PLSAULandmarkModel)):
            raise ValueError(
                "model must be a PLSRegression instance or PLSAULandmarkModel "
                "(returned by feat.plotting.load_viz_model())"
            )

    if au is None or isinstance(au, str) and au == "neutral":
        au = np.zeros(model.n_components)

    landmarks = predict(au, model, feature_range=feature_range)
    if symmetrize:
        landmarks = _symmetrize_dlib68(landmarks)
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
            "Don't forget to pass a 'gaze' vector of len(4), " "using neutral as default"
        )
        gaze = None

    title = kwargs.pop("title", None)
    title_kwargs = kwargs.pop("title_kwargs", dict(wrap=True, fontsize=14, loc="center"))
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
        # Symmetrize endpoints too so arrows land on the drawn (symmetric)
        # face rather than the raw asymmetric model output — otherwise the
        # tutorial pattern of `predict(au)` → vectorfield draws arrows
        # offset from the visible features by ~5-10 px.
        if symmetrize:
            vectorfield = dict(vectorfield)
            vectorfield["reference"] = _symmetrize_dlib68(vectorfield["reference"])
            vectorfield["target"] = _symmetrize_dlib68(vectorfield["target"])
        ax = draw_vectorfield(ax=ax, **vectorfield)
    # Auto-derive viewport from all drawn artists (landmarks + muscle patches
    # + gaze quivers + vectorfield arrows), then add asymmetric padding so
    # the forehead has room above the topmost landmark (no landmarks live
    # above the eyebrows in the dlib-68 layout). Hardcoded [25,172]/[240,50]
    # was calibrated for the v1 viz model; the v2 PLSAULandmarkModel
    # (PR #301) returns landmarks in a different range, so a hardcoded
    # viewport drew everything off-screen.
    ax.relim(visible_only=True)
    (xmin, xmax) = ax.dataLim.intervalx
    (ymin, ymax) = ax.dataLim.intervaly
    x_pad = (xmax - xmin) * 0.12
    y_top_pad = (ymax - ymin) * 0.25  # forehead room above topmost landmark
    y_bot_pad = (ymax - ymin) * 0.08
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymax + y_bot_pad, ymin - y_top_pad)  # inverted (image coords)
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
    elif not isinstance(model, (PLSRegression, PLSAULandmarkModel)):
        raise ValueError(
            "model must be a PLSRegression instance or PLSAULandmarkModel "
            "(returned by feat.plotting.load_viz_model())"
        )

    if len(au) != model.n_components:
        print(au)
        print(model.n_components)
        raise ValueError(f"au vector must be length {model.n_components}.")

    if len(au.shape) == 1:
        au = np.reshape(au, (1, -1))

    if feature_range:
        au = minmax_scale(au, feature_range=feature_range, axis=1)

    # Handle auto-raveling feature added to PLSRegression in sklearn 1.3
    # because our v1 model was trained in an earlier version where this
    # attribute did not exist. PLSAULandmarkModel ignores this.
    # https://scikit-learn.org/stable/whats_new/v1.3.html#sklearn-cross-decomposition
    if isinstance(model, PLSRegression) and not hasattr(model, "_predict_1d"):
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


def draw_facegaze(pitch_rad, yaw_rad, facebox, ax, color="yellow", linewidth=2):
    """Draw a single gaze arrow from the face-bbox center in the predicted
    gaze direction.

    Convention matches L2CS / Gaze360: +pitch = subject looks up;
    +yaw = subject looks to camera's right (subject's left). The arrow is
    drawn in image coords (y-down), so we flip the y component when
    projecting the gaze unit vector onto the image plane.

    Args:
        pitch_rad: gaze pitch in radians (scalar or len-1 array)
        yaw_rad: gaze yaw in radians
        facebox: [x, y, w, h] of the detected face
        ax: matplotlib axis
        color: arrow color (default 'yellow' to stand out on most images)
        linewidth: arrow line width
    """
    pitch = float(np.asarray(pitch_rad).ravel()[0])
    yaw = float(np.asarray(yaw_rad).ravel()[0])
    x, y, w, h = facebox[:4]
    cx = x + w / 2.0
    cy = y + h / 2.0
    # Length proportional to facebox; full-deflection (~45°) maps to
    # about half the face width.
    length = min(w, h) * 1.1
    # Project the 3D gaze unit vector onto the image plane:
    #   dx_image = sin(yaw) * cos(pitch)   (+x = camera's right)
    #   dy_image = -sin(pitch)             (image y is flipped; +pitch up
    #                                       → arrow points UP, ie -y in pixels)
    dx = length * np.sin(yaw) * np.cos(pitch)
    dy = length * -np.sin(pitch)
    ax.arrow(
        cx, cy, dx, dy,
        color=color, linewidth=linewidth,
        head_width=length * 0.06, head_length=length * 0.08,
        length_includes_head=True,
    )
    return ax


def _gaze_to_pupil_offsets(pitch_rad, yaw_rad, scale=15.0):
    """Convert head-centric (pitch, yaw) to the legacy 4-vector pupil-offset
    format used by ``plot_face`` for synthetic AU-driven faces.

    The synthetic-face renderer draws pupils at
    ``[left_x + gaze[0], left_y - gaze[1]/2, right_x + gaze[2], right_y - gaze[3]/2]``
    where the offsets are in pixel units of the canvas. We map symmetric
    pupil offsets from gaze direction:
      x-offset (both pupils): ``scale * sin(yaw)``
      y-offset (both pupils): ``scale * sin(pitch)`` (+pitch = look up,
        which is +y in the canvas frame the renderer uses for pupils)
    """
    pitch = float(np.asarray(pitch_rad).ravel()[0])
    yaw = float(np.asarray(yaw_rad).ravel()[0])
    if not np.isfinite(pitch) or not np.isfinite(yaw):
        return None
    xoff = scale * np.sin(yaw)
    yoff = scale * np.sin(pitch)
    return [xoff, yoff, xoff, yoff]


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

    interp_func = CubicEaseInOut if interp_func is None else interp_func
    # Loop over each AU and generate a cubic bezier style interpolation from its
    # starting intensity to its ending intensity
    au_interpolations = []
    for au_start, au_end in zip(start, end):
        interpolator = interp_func(au_start, au_end)
        intensities = [*map(interpolator, np.linspace(0, 1, num_frames))]
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
    start,
    end,
    save,
    AU=None,
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
    animation.save(save, fps=fps)
    plt.close("all")


class PLSAULandmarkModel:
    """Lightweight wrapper around the v2 AU → 68-pt landmark PLS weights.

    Mirrors the subset of sklearn's ``PLSRegression`` interface that
    ``feat.plotting.predict()`` actually uses: a ``.predict(au)`` method
    and an ``.n_components`` attribute reporting the AU input dimension
    (20). The underlying model was trained with 23 input features
    (20 AU + 3 pose); pose is implicitly held at zero at inference, which
    by construction zeros out the 60 pose×AU interaction features that
    contributed during training. Inference is a single matmul.

    Loaded by ``load_viz_model()`` (default since this PR). See
    https://huggingface.co/py-feat/au_to_landmarks for the model card,
    training details, and OOS performance (R² = 0.794 ± 0.004 on 3-fold
    GroupKFold-by-video).
    """

    def __init__(self, coef, intercept, au_columns, model_name="au_to_landmarks_pls_v2"):
        # coef shape: (23, 136); the deployed slice with pose absorbed
        self._coef = np.asarray(coef, dtype=np.float32)
        self._intercept = np.asarray(intercept, dtype=np.float32)
        self.au_columns = list(au_columns)
        # Number of AU input features the user passes (pose is implicit-zero)
        self.n_components = 20
        self.model_name_ = model_name

    def predict(self, au):
        """Predict 136-d axis-major landmarks from (n_samples, 20) AU intensities.

        Pose is held at zero by zero-padding the input to 23-d so the
        deployed coef matrix can be used directly.
        """
        au = np.asarray(au, dtype=np.float32)
        if au.ndim == 1:
            au = au.reshape(1, -1)
        n = au.shape[0]
        x = np.empty((n, self._coef.shape[0]), dtype=np.float32)
        x[:, : au.shape[1]] = au
        x[:, au.shape[1] :] = 0.0  # pose covariates → zero
        return x @ self._coef + self._intercept

    def __repr__(self):
        return (
            f"PLSAULandmarkModel(model_name='{self.model_name_}', "
            f"n_components={self.n_components}, "
            f"output_shape=(n_samples, {self._coef.shape[1]}))"
        )


# Module-level cache for the v2 wrapper. Avoids re-deserializing the NPZ
# (and re-allocating the coef/intercept arrays) on every load_viz_model() call.
# The HF download itself is already cached by huggingface_hub.
_PLS_V2_VIZ_MODEL = None


def _load_pls_v2_from_hub(verbose=False):
    """Download (cached) and wrap the v2 AU → 68-pt landmark PLS NPZ from HuggingFace Hub."""
    global _PLS_V2_VIZ_MODEL
    if _PLS_V2_VIZ_MODEL is not None:
        return _PLS_V2_VIZ_MODEL

    if verbose:
        print("Loading v2 PLS landmarks model from HuggingFace Hub")
    path = hf_hub_download(
        repo_id="py-feat/au_to_landmarks",
        filename="au_to_landmarks_pls_v2.npz",
        cache_dir=get_resource_path(),
    )
    z = np.load(path, allow_pickle=False)
    _PLS_V2_VIZ_MODEL = PLSAULandmarkModel(
        coef=z["coef"],
        intercept=z["intercept"],
        au_columns=[str(s) for s in z["au_columns"]],
        model_name="au_to_landmarks_pls_v2",
    )
    return _PLS_V2_VIZ_MODEL


# ---------------------------------------------------------------------
# AU + pose → 478-vertex MediaPipe FaceMesh (opt-in 3D visualization).
# Mirrors the 68-pt landmark wrapper above. Output lives in a pose-canonical
# frame anchored on stable upper-face landmarks (forehead + nose bridge +
# canthi). See https://huggingface.co/py-feat/au_to_mesh for the model card.
# ---------------------------------------------------------------------


class PLSAUMeshModel:
    """Wrapper around the v2 AU + pose → 478-vertex MP mesh PLS weights.

    Mirrors the ``PLSAULandmarkModel`` interface (``.predict(au)``,
    ``.n_components``) but emits the full 3D MediaPipe FaceMesh instead of
    a 68-point dlib-style 2D landmark layout. Pose is held implicit-zero
    at inference; the absorbed pose×AU interaction terms drop out
    automatically. Inference is a single matmul.

    Loaded by ``load_face_mesh_viz_model()``. Underlying weights live in
    the ``py-feat/au_to_mesh`` HF Hub repo.
    """

    def __init__(
        self,
        coef,
        intercept,
        au_columns,
        pose_columns,
        mean_aligned_mesh,
        model_name="au_to_mesh_pls_v2",
    ):
        # coef shape: (23, 1434); the deployed slice with pose absorbed
        self._coef = np.asarray(coef, dtype=np.float32)
        self._intercept = np.asarray(intercept, dtype=np.float32)
        self.au_columns = list(au_columns)
        self.pose_columns = list(pose_columns)
        # Number of AU input features the user passes (pose is implicit-zero)
        self.n_components = len(self.au_columns)
        # Population-mean rest mesh in pose-canonical frame; useful as a
        # reference baseline for visualizing AU deltas.
        self.mean_aligned_mesh = np.asarray(mean_aligned_mesh, dtype=np.float32)
        self.model_name_ = model_name

    def predict(self, au):
        """Predict (n_samples, 1434) flat mesh coords from (n_samples, 20) AU.

        The 1434-d flat layout is **axis-major**: ``[x_0..x_477 | y_0..y_477
        | z_0..z_477]`` — matching how training stacks the per-vertex columns
        ``x_i / y_i / z_i``. Use ``predict_face_mesh()`` for an ``(n, 478, 3)``
        reshape with the right vertex ordering.

        Pose channels are zero-padded so the deployed coef matrix can be
        used directly. Output is in the Procrustes-aligned canonical frame
        (units are GPA-aligned cm, not image pixels).
        """
        au = np.asarray(au, dtype=np.float32)
        if au.ndim == 1:
            au = au.reshape(1, -1)
        if au.shape[1] != self.n_components:
            raise ValueError(
                f"au must have {self.n_components} columns "
                f"(matching {self.au_columns}); got {au.shape[1]}."
            )
        n = au.shape[0]
        x = np.zeros((n, self._coef.shape[0]), dtype=np.float32)
        x[:, : au.shape[1]] = au
        # Pose channels remain zero (already initialized)
        return x @ self._coef + self._intercept

    def __repr__(self):
        return (
            f"PLSAUMeshModel(model_name='{self.model_name_}', "
            f"n_components={self.n_components}, "
            f"output_shape=(n_samples, 478, 3))"
        )


# Module-level cache; the HF download is already cached by huggingface_hub.
_PLS_V2_MESH_MODEL = None


def _load_pls_au_to_mesh_v2_from_hub(verbose=False):
    """Download (cached) and wrap the v2 AU→mesh PLS NPZ from HuggingFace Hub."""
    global _PLS_V2_MESH_MODEL
    if _PLS_V2_MESH_MODEL is not None:
        return _PLS_V2_MESH_MODEL

    if verbose:
        print("Loading v2 PLS AU→mesh model from HuggingFace Hub")
    path = hf_hub_download(
        repo_id="py-feat/au_to_mesh",
        filename="au_to_mesh_pls_v2.npz",
        cache_dir=get_resource_path(),
    )
    z = np.load(path, allow_pickle=False)
    au_columns = [str(s) for s in z["au_columns"]]
    # Guard against silent mislabeling: callers pass `au` as a 20-d vector
    # assuming index i corresponds to AU_LANDMARK_MAP["Feat"][i]. A re-trained
    # npz with a shuffled column order would silently use wrong AU positions
    # and produce subtly wrong mesh deformations. Refuse to load instead.
    if au_columns != AU_LANDMARK_MAP["Feat"]:
        raise RuntimeError(
            "AU→mesh PLS au_columns drifted from AU_LANDMARK_MAP['Feat']. "
            f"NPZ: {au_columns}; canonical: {AU_LANDMARK_MAP['Feat']}. "
            "Re-train or update the canonical AU list."
        )
    _PLS_V2_MESH_MODEL = PLSAUMeshModel(
        coef=z["coef"],
        intercept=z["intercept"],
        au_columns=au_columns,
        pose_columns=[str(s) for s in z["pose_columns"]],
        mean_aligned_mesh=z["mean_aligned_mesh"],
        model_name="au_to_mesh_pls_v2",
    )
    return _PLS_V2_MESH_MODEL


def load_face_mesh_viz_model(verbose=False):
    """Load the AU + pose → 478-pt MediaPipe FaceMesh PLS visualization model.

    Returns a ``PLSAUMeshModel`` whose ``.predict(au)`` produces the 478-vertex
    3D mesh (flattened to 1434-d) in a pose-canonical frame. Use
    ``predict_face_mesh()`` for a (478, 3) reshape, or ``plot_face_mesh()`` for
    a quick 3D wireframe.

    Args:
        verbose: print a status line when first downloading.

    Returns:
        PLSAUMeshModel
    """
    return _load_pls_au_to_mesh_v2_from_hub(verbose=verbose)


def predict_face_mesh(au, model=None):
    """Predict the 3D MediaPipe FaceMesh from AU intensities.

    Args:
        au: AU intensity vector or batch. Shape ``(20,)`` or ``(n, 20)`` — the
            standard ``AU_LANDMARK_MAP['Feat']`` order.
        model: optional ``PLSAUMeshModel``; defaults to the cached v2 model.

    Returns:
        Predicted mesh in pose-canonical-frame coordinates (GPA-aligned cm,
        not image pixels). Shape ``(478, 3)`` for a 1-D AU input,
        ``(n, 478, 3)`` for a batched 2-D input.
    """
    if model is None:
        model = load_face_mesh_viz_model()
    if not isinstance(model, PLSAUMeshModel):
        raise ValueError(
            "model must be a PLSAUMeshModel "
            "(returned by feat.plotting.load_face_mesh_viz_model())"
        )
    au_arr = np.asarray(au)
    is_single = au_arr.ndim == 1
    if au_arr.ndim == 1:
        au_arr = au_arr.reshape(1, -1)
    if au_arr.shape[1] != model.n_components:
        raise ValueError(
            f"au vector must be length {model.n_components}; got {au_arr.shape[1]}."
        )
    flat = model.predict(au_arr)  # (n, 1434), axis-major [x | y | z]
    xs = flat[:, :478]
    ys = flat[:, 478:956]
    zs = flat[:, 956:]
    mesh = np.stack([xs, ys, zs], axis=-1)  # (n, 478, 3)
    return mesh[0] if is_single else mesh


def plot_face_mesh(
    au=None,
    model=None,
    ax=None,
    color="black",
    linewidth=0.6,
    alpha=0.9,
    view_init=(0, -90),
    *,
    mesh=None,
    mode="contours",
    gaze=None,
    gaze_color="gold",
    gaze_length_frac=0.3,
):
    """3D wireframe of the predicted face mesh. Opt-in alternative to ``plot_face``.

    Renders a 478-vertex MediaPipe mesh as a 3D wireframe. The mesh source is
    selected by what the caller passes:

    - ``mesh`` given: draw that mesh directly (e.g., output of
      ``predict_mesh_from_dlib68`` for a Detector Fex).
    - ``au`` given: predict via the AU→mesh PLS model (PR #304).
    - neither: draw the population-mean rest mesh.

    Edge density is controlled by ``mode``:

    - ``'contours'`` (default, ~124 edges): canonical contours — lips, eyes,
      eyebrows, face oval. Fast and uncluttered; weak expressions can look
      similar to rest.
    - ``'tesselation'`` (~2,556 edges): full MediaPipe tessellation. Matches
      what ``plot_face_mesh_plotly`` shows by default. Slower to render in
      matplotlib but reveals nose, cheek, and inner-face structure that's
      invisible in contours.

    Coordinate frame: the trained mesh uses standard math convention with
    ``+X`` lateral, ``+Y`` up (forehead at ~+8, chin at ~-8), ``+Z`` out of
    the face. We map data ``(x, y, z)`` to matplotlib ``(x, z, y)`` so the
    default 3D viewport renders the face standing upright with Z (depth) into
    the screen — matching how a viewer sees the face.

    Args:
        au: AU intensity vector ``(20,)`` in ``AU_LANDMARK_MAP['Feat']`` order.
        mesh: precomputed ``(478, 3)`` mesh array, in the canonical frame.
        model: optional ``PLSAUMeshModel`` for the ``au`` path; defaults to
            the cached v2 model.
        ax: optional matplotlib 3D axis. If ``None``, a new figure is created.
        color, linewidth, alpha: line styling.
        view_init: ``(elev, azim)`` matplotlib view angles. Default frames
            the face front-on; rotate ``azim`` to see profile.
        mode: ``'contours'`` (default) or ``'tesselation'``.

    Returns:
        The matplotlib 3D axis with the wireframe drawn.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    from feat.utils.mp_plotting import FaceLandmarksConnections

    if mesh is not None and au is not None:
        raise ValueError("pass either `au` or `mesh`, not both")

    if mesh is not None:
        verts = np.asarray(mesh, dtype=np.float32)
        if verts.shape != (478, 3):
            raise ValueError(
                f"mesh must have shape (478, 3); got {verts.shape}. "
                "For batched predictions, plot one face at a time."
            )
    elif au is None:
        if model is None:
            model = load_face_mesh_viz_model()
        verts = model.mean_aligned_mesh
    else:
        verts = predict_face_mesh(au, model=model)
        if verts.ndim != 2:
            raise ValueError(
                "plot_face_mesh expects a single AU vector; pass one face at a time. "
                "For batched predictions use predict_face_mesh(au_batch)."
            )

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

    if mode in ("tesselation", "tessellation"):
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    elif mode == "contours":
        connections = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    else:
        raise ValueError(
            f"mode must be 'contours' or 'tesselation' (or 'tessellation'); "
            f"got {mode!r}"
        )

    # Map data (X, Y, Z) → matplotlib (X, Z, Y) so face renders upright with
    # depth into the screen under the default viewport.
    xs = verts[:, 0]
    ys_mpl = verts[:, 2]
    zs_mpl = verts[:, 1]

    # Batch all edges into a single Line3DCollection — ~10-30x faster than
    # looping ax.plot() per edge, which matters for tesselation (~2,556 edges).
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    segments = [
        ((xs[c.start], ys_mpl[c.start], zs_mpl[c.start]),
         (xs[c.end], ys_mpl[c.end], zs_mpl[c.end]))
        for c in connections
    ]
    ax.add_collection3d(Line3DCollection(
        segments, colors=color, linewidths=linewidth, alpha=alpha,
    ))

    if gaze is not None:
        pitch_rad, yaw_rad = float(gaze[0]), float(gaze[1])
        origin, direction, length = _gaze_arrow_in_mesh_frame(
            verts, pitch_rad, yaw_rad, length_frac=gaze_length_frac,
        )
        end_pt = origin + length * direction
        # Same data → mpl-axis swap (x, z, y) used for the mesh.
        ax.quiver(
            origin[0], origin[2], origin[1],
            (end_pt[0] - origin[0]), (end_pt[2] - origin[2]), (end_pt[1] - origin[1]),
            color=gaze_color, linewidth=2.0, arrow_length_ratio=0.25,
        )

    # Equal aspect ratio so the face isn't squashed by mpl3d's default scaling.
    extents = np.array([
        [xs.min(), xs.max()],
        [ys_mpl.min(), ys_mpl.max()],
        [zs_mpl.min(), zs_mpl.max()],
    ])
    centers = extents.mean(axis=1)
    half = (extents[:, 1] - extents[:, 0]).max() / 2
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view_init[0], azim=view_init[1])
    ax.set_axis_off()
    return ax


def plot_face_mesh_plotly(
    au=None,
    model=None,
    color="black",
    line_width=1.5,
    opacity=0.85,
    background="white",
    *,
    mesh=None,
    mode="tesselation",
    gaze=None,
    gaze_color="gold",
    gaze_length_frac=0.3,
):
    """Interactive 3D face mesh as a Plotly figure. Opt-in alternative to ``plot_face_mesh``.

    Renders the 478-vertex MP face mesh as a wireframe in an interactive
    Plotly 3D scene the user can rotate, pan, and zoom. Same data sources
    as ``plot_face_mesh``: pass ``au`` for an AU→mesh PLS prediction, ``mesh``
    for a precomputed ``(478, 3)`` array, or neither for the rest mesh.

    Args:
        au: AU intensity vector ``(20,)`` in ``AU_LANDMARK_MAP['Feat']`` order.
        model: optional ``PLSAUMeshModel`` for the ``au`` path.
        color, line_width, opacity, background: scene styling.
        mesh: keyword-only. Precomputed ``(478, 3)`` mesh (e.g., from
            ``predict_mesh_from_dlib68``). Mutually exclusive with ``au``.
        mode: which connection set to draw —

            - ``'tesselation'`` (or ``'tessellation'``, default): the
              2,556-edge MP tessellation; shows the full mesh structure
              and looks dense / 3D. (MediaPipe's upstream constant uses
              the single-l spelling; both are accepted.)
            - ``'contours'``: the 124-edge canonical contours (lips + eyes
              + eyebrows + face oval); matches ``plot_face_mesh``.

    Returns:
        ``plotly.graph_objects.Figure``. Call ``.show()`` to render in a
        browser/notebook, or ``.write_html(path)`` / ``.write_image(path)``
        to persist.

    Coordinate frame: matches ``plot_face_mesh`` — data ``(X, Y, Z)`` is
    mapped to Plotly ``(X, Z, Y)`` so the face renders upright with depth
    into the screen under the default camera.
    """
    import plotly.graph_objects as go
    from feat.utils.mp_plotting import FaceLandmarksConnections

    if mesh is not None and au is not None:
        raise ValueError("pass either `au` or `mesh`, not both")

    if mesh is not None:
        verts = np.asarray(mesh, dtype=np.float32)
        if verts.shape != (478, 3):
            raise ValueError(
                f"mesh must have shape (478, 3); got {verts.shape}."
            )
    elif au is None:
        if model is None:
            model = load_face_mesh_viz_model()
        verts = model.mean_aligned_mesh
    else:
        verts = predict_face_mesh(au, model=model)
        if verts.ndim != 2:
            raise ValueError(
                "plot_face_mesh_plotly expects a single AU vector; pass one face at a time."
            )

    # MediaPipe's constant uses the single-l "tesselation" spelling; accept
    # the standard English "tessellation" too so users typing either land
    # in the right place.
    if mode in ("tesselation", "tessellation"):
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    elif mode == "contours":
        connections = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    else:
        raise ValueError(
            f"mode must be 'tesselation' (or 'tessellation') or 'contours'; "
            f"got {mode!r}"
        )

    # Apply same data → mpl axis swap as plot_face_mesh so face is upright.
    xs = verts[:, 0]
    ys = verts[:, 2]
    zs = verts[:, 1]

    # Build a single Scatter3d trace with NaN separators between line
    # segments — much faster than emitting one trace per edge for the
    # 2,556-edge tessellation.
    n_edges = len(connections)
    seg_x = np.empty(3 * n_edges, dtype=np.float32)
    seg_y = np.empty(3 * n_edges, dtype=np.float32)
    seg_z = np.empty(3 * n_edges, dtype=np.float32)
    nan = np.float32(np.nan)
    for i, conn in enumerate(connections):
        a, b = conn.start, conn.end
        seg_x[3 * i] = xs[a]; seg_x[3 * i + 1] = xs[b]; seg_x[3 * i + 2] = nan
        seg_y[3 * i] = ys[a]; seg_y[3 * i + 1] = ys[b]; seg_y[3 * i + 2] = nan
        seg_z[3 * i] = zs[a]; seg_z[3 * i + 1] = zs[b]; seg_z[3 * i + 2] = nan

    traces = [go.Scatter3d(
        x=seg_x, y=seg_y, z=seg_z,
        mode="lines",
        line=dict(color=color, width=line_width),
        opacity=opacity,
        showlegend=False,
        hoverinfo="skip",
    )]

    if gaze is not None:
        pitch_rad, yaw_rad = float(gaze[0]), float(gaze[1])
        origin, direction, length = _gaze_arrow_in_mesh_frame(
            verts, pitch_rad, yaw_rad, length_frac=gaze_length_frac,
        )
        end_pt = origin + length * direction
        # Data → plotly axis swap (x, z, y) so arrow lives in same scene as mesh.
        traces.append(go.Scatter3d(
            x=[origin[0], end_pt[0]], y=[origin[2], end_pt[2]], z=[origin[1], end_pt[1]],
            mode="lines+markers",
            line=dict(color=gaze_color, width=line_width * 3),
            marker=dict(size=[1, 5], color=gaze_color, symbol=["circle", "diamond"]),
            showlegend=False, hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)

    # Equal-aspect bounds so the face isn't visually squashed.
    extents = np.array([
        [xs.min(), xs.max()],
        [ys.min(), ys.max()],
        [zs.min(), zs.max()],
    ])
    centers = extents.mean(axis=1)
    half = (extents[:, 1] - extents[:, 0]).max() / 2
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[centers[0] - half, centers[0] + half]),
            yaxis=dict(visible=False, range=[centers[1] - half, centers[1] + half]),
            zaxis=dict(visible=False, range=[centers[2] - half, centers[2] + half]),
            aspectmode="cube",
            bgcolor=background,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=background,
    )
    return fig


# MP-478 mesh: outer-canthi landmark indices (anatomical-left / -right eye
# outer corner). Midpoint is a stable origin for the gaze arrow in the
# mesh's pose-canonical frame.
_MP_OUTER_CANTHI = (33, 263)


def _gaze_arrow_in_mesh_frame(verts, pitch_rad, yaw_rad, length_frac=0.3):
    """Return (origin, direction, length) for a gaze arrow on a 478-vertex MP mesh.

    Origin: midpoint of outer-canthi landmarks (33, 263) — a stable per-frame
    anchor that doesn't drift with eyelid AUs.

    Direction in the mesh's (+X lateral-to-viewer's-left, +Y up, +Z out-of-face)
    frame from L2CS (pitch, yaw) head-centric Euler angles. Positive pitch =
    looking up; positive yaw = turning head/eyes to the subject's right
    (which is the viewer's left, i.e., -X in the mesh frame).

    Length: ``length_frac`` of the face Y-extent (chin-to-forehead) so the
    arrow scales with whatever face size the mesh model produces.
    """
    origin = (verts[_MP_OUTER_CANTHI[0]] + verts[_MP_OUTER_CANTHI[1]]) / 2.0
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    # Subject's-right yaw → viewer's-left → -X direction in mesh frame.
    direction = np.array([-sy * cp, sp, cy * cp], dtype=np.float32)
    face_h = float(verts[:, 1].max() - verts[:, 1].min())
    length = face_h * length_frac
    return origin, direction, length


# dlib-68 line sequences used by draw_lineface(). Lifted here so the plotly
# 2D animation can build the same face geometry without owning matplotlib state.
# Each tuple is a polyline through landmark indices; NaN separates them when
# packed into a single Plotly Scatter trace.
_DLIB68_LINE_PATHS = (
    tuple(range(0, 17)),                                            # jaw / face outline
    (17, 18, 19, 20, 21),                                           # left eyebrow
    (22, 23, 24, 25, 26),                                           # right eyebrow
    (27, 28, 29, 30),                                               # nose bridge
    (31, 32, 33, 34, 35),                                           # nose bottom
    (36, 37, 38, 39, 40, 41, 36),                                   # left eye
    (42, 43, 44, 45, 46, 47, 42),                                   # right eye
    (48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60, 48),           # lips outer top + inner top
    (48, 60, 67, 66, 65, 64, 54, 55, 56, 57, 58, 59, 48),           # lips inner bottom + outer bottom
)


def animate_face_plotly(
    start,
    end,
    num_frames=24,
    fps=15,
    include_reverse=True,
    model=None,
    color="black",
    line_width=1.5,
    background="white",
    symmetrize=True,
):
    """Interactive 2D 68-pt face-landmark animation in Plotly.

    Returns a Plotly ``Figure`` with frames + play/pause + slider, animating
    the legacy ``plot_face`` line geometry between ``start`` and ``end`` AU
    vectors. Complements ``animate_face`` (matplotlib GIF) and
    ``animate_face_mesh_plotly`` (3D mesh).

    The plotly version is handy for live notebook exploration when you
    don't want to write a GIF file — and unlike matplotlib's celluloid
    path, the animation keeps the figure interactive (pan, zoom, scrub
    via the slider).

    Args:
        start: AU intensity vector ``(20,)`` at frame 0.
        end: AU intensity vector ``(20,)`` at the peak.
        num_frames: frames in the start→end half (cubic-eased).
        fps: playback rate (sets per-frame transition duration).
        include_reverse: append end→start so the loop returns to neutral.
        model: optional viz model override (defaults to the cached v2
            ``PLSAULandmarkModel``; same default as ``plot_face``).
        color, line_width, background: scene styling.
        symmetrize: mirror-average each frame's predicted landmarks so the
            animation stays clean even when the underlying model is
            mildly asymmetric (same default as ``plot_face``).

    Returns:
        ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    if model is None:
        model = load_viz_model()

    aus = interpolate_aus(
        start=np.asarray(start, dtype=np.float32),
        end=np.asarray(end, dtype=np.float32),
        num_frames=num_frames,
        include_reverse=include_reverse,
    )

    # Per-frame landmark prediction + optional symmetrize.
    frames_xy = []
    for au in aus:
        lm = predict(au, model=model)
        if symmetrize:
            lm = _symmetrize_dlib68(lm)
        frames_xy.append(lm)

    # Build NaN-separated polyline arrays per frame using the same line
    # paths draw_lineface walks. y-axis is inverted at layout level so
    # we get the image-coord convention used by plot_face.
    def _pack(lm):
        xs, ys = [], []
        for path in _DLIB68_LINE_PATHS:
            xs.extend(lm[0, i] for i in path)
            xs.append(np.nan)
            ys.extend(lm[1, i] for i in path)
            ys.append(np.nan)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    packed = [_pack(lm) for lm in frames_xy]

    # Global extent across all frames → stable camera (no zoom jitter).
    all_xs = np.concatenate([p[0] for p in packed])
    all_ys = np.concatenate([p[1] for p in packed])
    finite_x = all_xs[np.isfinite(all_xs)]
    finite_y = all_ys[np.isfinite(all_ys)]
    x_pad = (finite_x.max() - finite_x.min()) * 0.10
    y_pad = (finite_y.max() - finite_y.min()) * 0.08
    x_range = [finite_x.min() - x_pad, finite_x.max() + x_pad]
    y_range = [finite_y.max() + y_pad, finite_y.min() - y_pad]  # inverted

    def _scatter(seg_x, seg_y):
        return go.Scatter(
            x=seg_x, y=seg_y,
            mode="lines",
            line=dict(color=color, width=line_width),
            showlegend=False, hoverinfo="skip",
        )

    frames = [
        go.Frame(data=[_scatter(*packed[i])], name=str(i))
        for i in range(len(packed))
    ]
    frame_duration_ms = int(1000 / fps)

    fig = go.Figure(
        data=[_scatter(*packed[0])],
        frames=frames,
    )
    fig.update_layout(
        xaxis=dict(visible=False, range=x_range, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=y_range),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=background,
        plot_bgcolor=background,
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05, x=0.0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=frame_duration_ms, redraw=True),
                    fromcurrent=True, transition=dict(duration=0), mode="immediate"
                )]),
                dict(label="❚❚ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate"
                )]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.05, len=0.9, xanchor="left", yanchor="top",
            currentvalue=dict(prefix="frame ", visible=True, xanchor="right"),
            steps=[dict(
                method="animate", label=str(i),
                args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                     mode="immediate", transition=dict(duration=0))]
            ) for i in range(len(packed))],
        )],
    )
    return fig


def animate_face_mesh_plotly(
    start,
    end,
    num_frames=24,
    fps=15,
    include_reverse=True,
    model=None,
    mode="tesselation",
    color="black",
    line_width=1.5,
    opacity=0.85,
    background="white",
):
    """Interactive 3D face-mesh animation between two AU intensity vectors.

    Returns a Plotly ``Figure`` with frames + play/pause buttons + a frame
    slider. The 3D camera stays rotatable while the animation plays — so
    the user can pick a profile angle, hit play, and watch the expression
    morph from that viewpoint. Companion to ``plot_face_mesh_plotly`` (a
    single static frame) and ``animate_face`` (legacy 2D matplotlib GIF).

    Args:
        start: AU intensity vector ``(20,)`` at the animation's first frame.
        end: AU intensity vector ``(20,)`` at the peak.
        num_frames: number of frames in the start→end half (cubic-eased).
            Total frames is 2*num_frames if include_reverse, else num_frames.
        fps: playback rate. Sets the per-frame ``duration`` (ms) in plotly's
            animation transition.
        include_reverse: append end→start so the loop returns to neutral.
        model: optional ``PLSAUMeshModel`` (defaults to the cached v2 model).
        mode: ``'tesselation'`` (default) or ``'contours'``. See
            ``plot_face_mesh_plotly``.
        color, line_width, opacity, background: scene styling, same as
            ``plot_face_mesh_plotly``.

    Returns:
        ``plotly.graph_objects.Figure``. Call ``.show()`` to play in a
        notebook, or ``.write_html(path)`` to save a standalone HTML file
        with the animation controls embedded.
    """
    import plotly.graph_objects as go
    from feat.utils.mp_plotting import FaceLandmarksConnections

    if mode in ("tesselation", "tessellation"):
        connections = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    elif mode == "contours":
        connections = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    else:
        raise ValueError(
            f"mode must be 'tesselation' or 'contours'; got {mode!r}"
        )

    # Cubic-eased AU trajectory; reuses the helper that animate_face uses.
    aus = interpolate_aus(
        start=np.asarray(start, dtype=np.float32),
        end=np.asarray(end, dtype=np.float32),
        num_frames=num_frames,
        include_reverse=include_reverse,
    )

    # Predict every frame's mesh in one batched call when supported, else loop.
    if model is None:
        model = load_face_mesh_viz_model()
    meshes = predict_face_mesh(aus, model=model)  # (n_frames, 478, 3)

    # Build all per-frame NaN-separated segment arrays up front so we can
    # also compute global extents for a stable camera that doesn't rescale
    # mid-animation.
    n_edges = len(connections)
    a_idx = np.array([c.start for c in connections])
    b_idx = np.array([c.end for c in connections])
    nan = np.float32(np.nan)

    all_xs, all_ys, all_zs = [], [], []
    for verts in meshes:
        # Data (X, Y, Z) → plotly (X, Z, Y) for upright face orientation.
        xs = verts[:, 0]
        ys = verts[:, 2]
        zs = verts[:, 1]
        seg_x = np.empty(3 * n_edges, dtype=np.float32)
        seg_y = np.empty(3 * n_edges, dtype=np.float32)
        seg_z = np.empty(3 * n_edges, dtype=np.float32)
        seg_x[0::3] = xs[a_idx]; seg_x[1::3] = xs[b_idx]; seg_x[2::3] = nan
        seg_y[0::3] = ys[a_idx]; seg_y[1::3] = ys[b_idx]; seg_y[2::3] = nan
        seg_z[0::3] = zs[a_idx]; seg_z[1::3] = zs[b_idx]; seg_z[2::3] = nan
        all_xs.append(seg_x); all_ys.append(seg_y); all_zs.append(seg_z)

    # Global extents across all frames so the camera doesn't jitter.
    flat_verts = meshes.reshape(-1, 3)
    glob_xs = flat_verts[:, 0]
    glob_ys = flat_verts[:, 2]
    glob_zs = flat_verts[:, 1]
    extents = np.array([
        [glob_xs.min(), glob_xs.max()],
        [glob_ys.min(), glob_ys.max()],
        [glob_zs.min(), glob_zs.max()],
    ])
    centers = extents.mean(axis=1)
    half = (extents[:, 1] - extents[:, 0]).max() / 2

    def _scatter(seg_x, seg_y, seg_z):
        return go.Scatter3d(
            x=seg_x, y=seg_y, z=seg_z,
            mode="lines",
            line=dict(color=color, width=line_width),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        )

    frames = [
        go.Frame(data=[_scatter(all_xs[i], all_ys[i], all_zs[i])], name=str(i))
        for i in range(len(meshes))
    ]
    frame_duration_ms = int(1000 / fps)

    fig = go.Figure(
        data=[_scatter(all_xs[0], all_ys[0], all_zs[0])],
        frames=frames,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[centers[0] - half, centers[0] + half]),
            yaxis=dict(visible=False, range=[centers[1] - half, centers[1] + half]),
            zaxis=dict(visible=False, range=[centers[2] - half, centers[2] + half]),
            aspectmode="cube",
            bgcolor=background,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=background,
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05, x=0.0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=frame_duration_ms, redraw=True),
                    fromcurrent=True, transition=dict(duration=0), mode="immediate"
                )]),
                dict(label="❚❚ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate"
                )]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.05, len=0.9, xanchor="left", yanchor="top",
            currentvalue=dict(prefix="frame ", visible=True, xanchor="right"),
            steps=[dict(
                method="animate", label=str(i),
                args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                     mode="immediate", transition=dict(duration=0))]
            ) for i in range(len(meshes))],
        )],
    )
    return fig


# ---------------------------------------------------------------------
# 68-pt dlib landmarks → 478-vertex MediaPipe FaceMesh bridge.
# Lets users with a Detector Fex (mobilefacenet 68-pt) reconstruct an
# approximate MP mesh, opening up plot_face_mesh + downstream 3D consumers
# without a separate MPDetector pass. PCA-bottleneck linear regression
# trained on 340K paired (dlib-68, MP-478) frames; OOS R² ≈ 0.48.
# See https://huggingface.co/py-feat/landmarks68_to_mesh478.
# ---------------------------------------------------------------------


class PCALandmarks68ToMeshModel:
    """Wrapper around the v2 dlib-68 → MP-478 PCA-bottleneck bridge weights.

    Inference takes Procrustes-aligned 68-pt landmarks (axis-major flat
    ``[x_0..x_67 | y_0..y_67]``, 136-d) and emits the 478-vertex 3D mesh
    flattened to 1434-d in the same axis-major layout as the AU→mesh model
    (PR #304). Use ``predict_mesh_from_dlib68()`` for the convenient end-to-end
    path that handles raw-landmark alignment + reshape.

    Loaded by ``load_landmarks68_to_mesh478_model()``. Underlying weights live
    in the ``py-feat/landmarks68_to_mesh478`` HF Hub repo. OOS R² ≈ 0.48.
    """

    def __init__(
        self,
        coef,
        intercept,
        input_columns,
        anchor_indices_dlib68,
        reference_dlib_anchors,
        mean_aligned_dlib_landmarks,
        mean_predicted_mesh,
        model_name="landmarks68_to_mesh478_pca_v2",
    ):
        # coef shape: (136, 1434); deployed weights with PCA absorbed
        self._coef = np.asarray(coef, dtype=np.float32)
        self._intercept = np.asarray(intercept, dtype=np.float32)
        self.input_columns = list(input_columns)
        self.anchor_indices_dlib68 = np.asarray(anchor_indices_dlib68, dtype=np.int64)
        self.reference_dlib_anchors = np.asarray(reference_dlib_anchors, dtype=np.float32)
        self.mean_aligned_dlib_landmarks = np.asarray(
            mean_aligned_dlib_landmarks, dtype=np.float32
        )
        self.mean_predicted_mesh = np.asarray(mean_predicted_mesh, dtype=np.float32)
        self.model_name_ = model_name

    def predict(self, aligned_landmarks_136):
        """Predict (n, 1434) flat axis-major mesh from (n, 136) aligned landmarks.

        Inputs must already be Procrustes-aligned to the model's reference
        anchors. Use ``predict_mesh_from_dlib68()`` for raw-landmark inputs
        (it handles alignment + reshape).
        """
        x = np.asarray(aligned_landmarks_136, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self._coef.shape[0]:
            raise ValueError(
                f"input must have {self._coef.shape[0]} columns "
                "(axis-major aligned [x_0..x_67 | y_0..y_67]); "
                f"got {x.shape[1]}."
            )
        return x @ self._coef + self._intercept

    def __repr__(self):
        return (
            f"PCALandmarks68ToMeshModel(model_name='{self.model_name_}', "
            f"input_dim={self._coef.shape[0]}, "
            f"output_shape=(n_samples, 478, 3))"
        )


_LM68_TO_MESH478_MODEL = None


def _load_landmarks68_to_mesh478_v2_from_hub(verbose=False):
    """Download (cached) and wrap the v2 68→478 PCA-bottleneck NPZ from HF Hub."""
    global _LM68_TO_MESH478_MODEL
    if _LM68_TO_MESH478_MODEL is not None:
        return _LM68_TO_MESH478_MODEL

    if verbose:
        print("Loading v2 PCA dlib-68 → MP-478 model from HuggingFace Hub")
    path = hf_hub_download(
        repo_id="py-feat/landmarks68_to_mesh478",
        filename="landmarks68_to_mesh478_pca_v2.npz",
        cache_dir=get_resource_path(),
    )
    z = np.load(path, allow_pickle=False)
    input_columns = [str(s) for s in z["input_columns"]]
    # Guard against silent layout drift: the bridge expects axis-major
    # `lm_x_0..lm_x_67, lm_y_0..lm_y_67`. A future re-trained npz with a
    # different convention (e.g., interleaved per-vertex) would silently
    # produce garbage.
    expected = [f"lm_x_{i}" for i in range(68)] + [f"lm_y_{i}" for i in range(68)]
    if input_columns != expected:
        raise RuntimeError(
            "landmarks68_to_mesh478 input_columns drifted from canonical "
            "axis-major lm_x / lm_y layout. Re-train or update the loader."
        )
    _LM68_TO_MESH478_MODEL = PCALandmarks68ToMeshModel(
        coef=z["coef"],
        intercept=z["intercept"],
        input_columns=input_columns,
        anchor_indices_dlib68=z["anchor_indices_dlib68"],
        reference_dlib_anchors=z["reference_dlib_anchors"],
        mean_aligned_dlib_landmarks=z["mean_aligned_dlib_landmarks"],
        mean_predicted_mesh=z["mean_predicted_mesh"],
        model_name="landmarks68_to_mesh478_pca_v2",
    )
    return _LM68_TO_MESH478_MODEL


def load_landmarks68_to_mesh478_model(verbose=False):
    """Load the dlib-68 → MP-478 PCA-bottleneck bridge model.

    Use ``predict_mesh_from_dlib68()`` for the convenient end-to-end path
    that aligns raw landmarks + predicts + reshapes in one call.

    Args:
        verbose: print a status line when first downloading.

    Returns:
        PCALandmarks68ToMeshModel
    """
    return _load_landmarks68_to_mesh478_v2_from_hub(verbose=verbose)


def predict_mesh_from_dlib68(landmarks_68, model=None):
    """Reconstruct a 478-vertex MP mesh from raw 68-pt dlib landmarks.

    Bridges Detector output (mobilefacenet 68-pt landmarks in image-space
    pixels) to the MediaPipe FaceMesh frame, so users with a Detector Fex
    can feed ``plot_face_mesh()`` or other 3D consumers without running
    MPDetector.

    Steps internally:
        1. Procrustes-align the raw landmarks to the saved reference anchors
           (8 stable upper-face points: nose bridge + canthi).
        2. Flatten axis-major and apply the (136 → 1434) PCA-bottleneck weights.
        3. Reshape the flat output to ``(478, 3)``.

    Args:
        landmarks_68: ``(68, 2)`` for a single face or ``(n, 68, 2)`` for a
            batch. In image-space pixels (e.g., from
            ``Fex.landmarks_dlib68_xy``).
        model: optional ``PCALandmarks68ToMeshModel``; defaults to the cached
            v2 model.

    Returns:
        Predicted mesh in pose-canonical-frame coordinates (GPA-aligned cm,
        not image pixels). Shape ``(478, 3)`` for a 2-D input,
        ``(n, 478, 3)`` for a 3-D batch.
    """
    if model is None:
        model = load_landmarks68_to_mesh478_model()
    if not isinstance(model, PCALandmarks68ToMeshModel):
        raise ValueError(
            "model must be a PCALandmarks68ToMeshModel "
            "(returned by feat.plotting.load_landmarks68_to_mesh478_model())"
        )
    arr = np.asarray(landmarks_68, dtype=np.float32)
    is_single = arr.ndim == 2
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (68, 2):
        raise ValueError(
            f"landmarks_68 must have shape (68, 2) or (n, 68, 2); got {arr.shape}"
        )

    aligned = procrustes_align_2d_batched(
        arr, model.anchor_indices_dlib68, model.reference_dlib_anchors,
    )
    # Axis-major flat input: [x_0..x_67 | y_0..y_67]
    x_flat = np.hstack([aligned[:, :, 0], aligned[:, :, 1]]).astype(np.float32)
    flat = model.predict(x_flat)  # (n, 1434), axis-major [x | y | z]
    xs = flat[:, :478]
    ys = flat[:, 478:956]
    zs = flat[:, 956:]
    mesh = np.stack([xs, ys, zs], axis=-1)  # (n, 478, 3)
    return mesh[0] if is_single else mesh


def load_viz_model(
    file_name=None,
    prefer_joblib_if_version_match=True,
    verbose=False,
):
    """Load the AU → 68-pt facial landmark PLS visualization model.

    Default (``file_name=None``) returns the **v2 PLS model** (
    ``PLSAULandmarkModel``) trained on 350K paired (AU, landmark) frames
    from ~10K CelebV-HQ celebrity videos. OOS R² = 0.794 ± 0.004 on 3-fold
    GroupKFold-by-video; significantly outperforms the v1 Cheong et al. 2023
    model (R² ≈ 0.4-0.5 on 13K class-balanced rows).

    For backward compatibility, passing ``file_name="pyfeat_aus_to_landmarks"``
    loads the legacy v1 model (``sklearn.cross_decomposition.PLSRegression``
    instance, with the ``.h5`` companion for metadata). The v1 weights are
    co-housed in the same HuggingFace repo (https://huggingface.co/py-feat/au_to_landmarks)
    so rollback works without breaking caches.

    Args:
        file_name (str, optional): If ``None`` (default), load v2 PLS NPZ.
            If ``"pyfeat_aus_to_landmarks"``, load legacy v1 joblib + h5.
        prefer_joblib_if_version_match (bool, optional): Only relevant for
            v1 legacy loading. If sklearn / Python major.minor match the
            versions the v1 model was trained with, return the unpickled
            ``PLSRegression``. Otherwise reconstruct from ``.h5``. Default True.
        verbose (bool, optional): Print progress messages. Default False.

    Returns:
        model: ``PLSAULandmarkModel`` (v2, default) or ``PLSRegression`` (v1).
            Both expose ``.predict(au)`` and ``.n_components`` so
            ``feat.plotting.predict()`` works transparently with either.
    """

    # New default: v2 PLS NPZ from HuggingFace Hub.
    if file_name is None:
        return _load_pls_v2_from_hub(verbose=verbose)

    if "." in file_name:
        raise TypeError("Please use a file name with no extension")

    if file_name == "pyfeat_aus_to_landmarks":
        # Legacy v1 model: now mirrored on HuggingFace Hub at
        # py-feat/au_to_landmarks alongside the v2 NPZ. Both files are needed
        # (the .joblib carries the unpickled PLSRegression; the .h5 carries
        # metadata + reconstruction tensors). hf_hub_download caches under
        # ``feat/resources/`` so subsequent calls hit the local cache.
        joblib_path = hf_hub_download(
            repo_id="py-feat/au_to_landmarks",
            filename=f"{file_name}.joblib",
            cache_dir=get_resource_path(),
        )
        h5_path = hf_hub_download(
            repo_id="py-feat/au_to_landmarks",
            filename=f"{file_name}.h5",
            cache_dir=get_resource_path(),
        )
    else:
        # Other custom viz models — legacy URL-based path via model_list.json.
        h5_path = os.path.join(get_resource_path(), f"{file_name}.h5")
        joblib_path = os.path.join(get_resource_path(), f"{file_name}.joblib")
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


def plot_frame(
    frame,
    boxes=None,
    landmarks=None,
    boxes_width=2,
    boxes_colors="cyan",
    landmarks_radius=2,
    landmarks_width=2,
    landmarks_colors="white",
):
    """
    Plot Torch Frames and py-feat output. If multiple frames will create a grid of images

    Args:
        frame (torch.Tensor): Tensor of shape (B, C, H, W) or (C, H, W)
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes
        landmarks (torch.Tensor): Tensor of shape (N, 136) containing flattened 68 point landmark keystones

    Returns:
        PILImage
    """

    if len(frame.shape) == 4:
        B, C, H, W = frame.shape
    elif len(frame.shape) == 3:
        C, H, W = frame.shape
    else:
        raise ValueError("Can only plot (B,C,H,W) or (C,H,W)")
    if B == 1:
        if boxes is not None:
            new_frame = draw_bounding_boxes(
                frame.squeeze(0), boxes, width=boxes_width, colors=boxes_colors
            )

            if landmarks is not None:
                new_frame = draw_keypoints(
                    new_frame,
                    landmarks.reshape(landmarks.shape[0], -1, 2),
                    radius=landmarks_radius,
                    width=landmarks_width,
                    colors=landmarks_colors,
                )
        else:
            if landmarks is not None:
                new_frame = draw_keypoints(
                    frame.squeeze(0),
                    landmarks.reshape(landmarks.shape[0], -1, 2),
                    radius=landmarks_radius,
                    width=landmarks_width,
                    colors=landmarks_colors,
                )
            else:
                new_frame = frame.squeeze(0)
        return transforms.ToPILImage()(new_frame.squeeze(0))
    else:
        if (boxes is not None) & (landmarks is None):
            new_frame = make_grid(
                torch.stack(
                    [
                        draw_bounding_boxes(
                            f, b.unsqueeze(0), width=boxes_width, colors=boxes_colors
                        )
                        for f, b in zip(frame.unbind(dim=0), boxes.unbind(dim=0))
                    ],
                    dim=0,
                )
            )
        elif (landmarks is not None) & (boxes is None):
            new_frame = make_grid(
                torch.stack(
                    [
                        draw_keypoints(
                            f,
                            l.unsqueeze(0),
                            radius=landmarks_radius,
                            width=landmarks_width,
                            colors=landmarks_colors,
                        )
                        for f, l in zip(
                            frame.unbind(dim=0),
                            landmarks.reshape(landmarks.shape[0], -1, 2).unbind(dim=0),
                        )
                    ],
                    dim=0,
                )
            )
        elif (boxes is not None) & (landmarks is not None):
            new_frame = make_grid(
                torch.stack(
                    [
                        draw_keypoints(
                            fr,
                            l.unsqueeze(0),
                            radius=landmarks_radius,
                            width=landmarks_width,
                            colors=landmarks_colors,
                        )
                        for fr, l in zip(
                            [
                                draw_bounding_boxes(
                                    f,
                                    b.unsqueeze(0),
                                    width=boxes_width,
                                    colors=boxes_colors,
                                )
                                for f, b in zip(frame.unbind(dim=0), boxes.unbind(dim=0))
                            ],
                            landmarks.reshape(landmarks.shape[0], -1, 2).unbind(dim=0),
                        )
                    ]
                )
            )
        else:
            new_frame = make_grid(frame)
        return transforms.ToPILImage()(new_frame)


def extract_face_from_landmarks(frame, landmarks, face_size=112):
    """Extract a face in a frame with a convex hull of landmarks.

    This function extracts the faces of the frame with convex hulls and masks out the rest.

    Args:
        frame (array): The original image]
        detected_faces (list): face bounding box
        landmarks (list): the landmark information]
        align (bool): align face to standard position
        size_output (int, optional): [description]. Defaults to 112.

    Returns:
        resized_face_np: resized face as a numpy array
        new_landmarks: landmarks of aligned face
    """

    if not isinstance(frame, torch.Tensor):
        raise ValueError(f"image must be a tensor not {type(frame)}")

    if len(frame.shape) != 4:
        frame = frame.unsqueeze(0)

    landmarks = landmarks.cpu().detach().numpy()

    aligned_img, new_landmarks = align_face(
        frame,
        landmarks.flatten(),
        landmark_type=68,
        box_enlarge=2.5,
        img_size=face_size,
    )

    hull = ConvexHull(new_landmarks)
    mask = grid_points_in_poly(
        shape=aligned_img.shape[-2:],
        # for some reason verts need to be flipped
        verts=list(
            zip(
                new_landmarks[hull.vertices][:, 1],
                new_landmarks[hull.vertices][:, 0],
            )
        ),
    )
    mask[
        0 : np.min([new_landmarks[0][1], new_landmarks[16][1]]),
        new_landmarks[0][0] : new_landmarks[16][0],
    ] = True
    masked_image = mask_image(aligned_img, mask)

    return (masked_image, new_landmarks)


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
        return dict(type="path", path=path, line=dict(color=line_color, width=line_width))

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

    # masseter_l = face_polygon_svg(
    #     [
    #         (row["x_2"], row["y_2"]),
    #         (row["x_3"], row["y_3"]),
    #         (row["x_4"], row["y_4"]),
    #         (row["x_5"], row["y_5"]),
    #         (row["x_6"], row["y_6"]),
    #         (row["x_5"], row["y_33"]),
    #     ],
    #     img_height,
    # )

    # masseter_r = face_polygon_svg(
    #     [
    #         (row["x_14"], row["y_14"]),
    #         (row["x_13"], row["y_13"]),
    #         (row["x_12"], row["y_12"]),
    #         (row["x_11"], row["y_11"]),
    #         (row["x_10"], row["y_10"]),
    #         (row["x_11"], row["y_33"]),
    #     ],
    #     img_height,
    # )

    # temporalis_l = face_polygon_svg(
    #     [
    #         (row["x_2"], row["y_2"]),
    #         (row["x_1"], row["y_1"]),
    #         (row["x_0"], row["y_0"]),
    #         (row["x_17"], row["y_17"]),
    #         (row["x_36"], row["y_36"]),
    #     ],
    #     img_height,
    # )

    # temporalis_r = face_polygon_svg(
    #     [
    #         (row["x_14"], row["y_14"]),
    #         (row["x_15"], row["y_15"]),
    #         (row["x_16"], row["y_16"]),
    #         (row["x_26"], row["y_26"]),
    #         (row["x_45"], row["y_45"]),
    #     ],
    #     img_height,
    # )

    # dep_lab_inf_l = face_polygon_svg(
    #     [
    #         (row["x_57"], row["y_57"]),
    #         (row["x_58"], row["y_58"]),
    #         (row["x_59"], row["y_59"]),
    #         (row["x_6"], row["y_6"]),
    #         (row["x_7"], row["y_7"]),
    #     ],
    #     img_height,
    # )

    # dep_lab_inf_r = face_polygon_svg(
    #     [
    #         (row["x_57"], row["y_57"]),
    #         (row["x_56"], row["y_56"]),
    #         (row["x_55"], row["y_55"]),
    #         (row["x_10"], row["y_10"]),
    #         (row["x_9"], row["y_9"]),
    #     ],
    #     img_height,
    # )

    # dep_ang_or_l = face_polygon_svg(
    #     [
    #         (row["x_48"], row["y_48"]),
    #         (row["x_7"], row["y_7"]),
    #         (row["x_6"], row["y_6"]),
    #     ],
    #     img_height,
    # )

    # dep_ang_or_r = face_polygon_svg(
    #     [
    #         (row["x_54"], row["y_54"]),
    #         (row["x_9"], row["y_9"]),
    #         (row["x_10"], row["y_10"]),
    #     ],
    #     img_height,
    # )

    # mentalis_l = face_polygon_svg(
    #     [
    #         (row["x_58"], row["y_58"]),
    #         (row["x_7"], row["y_7"]),
    #         (row["x_8"], row["y_8"]),
    #     ],
    #     img_height,
    # )

    # mentalis_r = face_polygon_svg(
    #     [
    #         (row["x_56"], row["y_56"]),
    #         (row["x_9"], row["y_9"]),
    #         (row["x_8"], row["y_8"]),
    #     ],
    #     img_height,
    # )

    # risorius_l = face_polygon_svg(
    #     [
    #         (row["x_4"], row["y_4"]),
    #         (row["x_5"], row["y_5"]),
    #         (row["x_48"], row["y_48"]),
    #     ],
    #     img_height,
    # )

    # risorius_r = face_polygon_svg(
    #     [
    #         (row["x_11"], row["y_11"]),
    #         (row["x_12"], row["y_12"]),
    #         (row["x_54"], row["y_54"]),
    #     ],
    #     img_height,
    # )

    # bottom = (row["y_8"] - row["y_57"]) / 2

    # orb_oris_l = face_polygon_svg(
    #     [
    #         (row["x_48"], row["y_48"]),
    #         (row["x_59"], row["y_59"]),
    #         (row["x_58"], row["y_58"]),
    #         (row["x_57"], row["y_57"]),
    #         (row["x_56"], row["y_56"]),
    #         (row["x_55"], row["y_55"] + bottom),
    #         (row["x_54"], row["y_54"] + bottom),
    #         (row["x_55"], row["y_55"] + bottom),
    #         (row["x_56"], row["y_56"] + bottom),
    #         (row["x_57"], row["y_57"] + bottom),
    #         (row["x_58"], row["y_58"] + bottom),
    #         (row["x_59"], row["y_59"] + bottom),
    #     ],
    #     img_height,
    # )

    # orb_oris_u = face_polygon_svg(
    #     [
    #         (row["x_48"], row["y_48"]),
    #         (row["x_49"], row["y_49"]),
    #         (row["x_50"], row["y_50"]),
    #         (row["x_51"], row["y_51"]),
    #         (row["x_52"], row["y_52"]),
    #         (row["x_53"], row["y_53"]),
    #         (row["x_54"], row["y_54"]),
    #         (row["x_33"], row["y_33"]),
    #     ],
    #     img_height,
    # )

    # frontalis_l = face_polygon_svg(
    #     [
    #         (row["x_27"], row["y_27"]),
    #         (row["x_39"], row["y_39"]),
    #         (row["x_38"], row["y_38"]),
    #         (row["x_37"], row["y_37"]),
    #         (row["x_36"], row["y_36"]),
    #         (row["x_17"], row["y_17"]),
    #         (row["x_18"], row["y_18"]),
    #         (row["x_19"], row["y_19"]),
    #         (row["x_20"], row["y_20"]),
    #         (row["x_21"], row["y_21"]),
    #     ],
    #     img_height,
    # )

    # frontalis_r = face_polygon_svg(
    #     [
    #         (row["x_27"], row["y_27"]),
    #         (row["x_22"], row["y_22"]),
    #         (row["x_23"], row["y_23"]),
    #         (row["x_24"], row["y_24"]),
    #         (row["x_25"], row["y_25"]),
    #         (row["x_26"], row["y_26"]),
    #         (row["x_45"], row["y_45"]),
    #         (row["x_44"], row["y_44"]),
    #         (row["x_43"], row["y_43"]),
    #         (row["x_42"], row["y_42"]),
    #     ],
    #     img_height,
    # )

    # frontalis_inner_l = face_polygon_svg(
    #     [
    #         (row["x_27"], row["y_27"]),
    #         (row["x_39"], row["y_39"]),
    #         (row["x_21"], row["y_21"]),
    #     ],
    #     img_height,
    # )

    # frontalis_inner_r = face_polygon_svg(
    #     [
    #         (row["x_27"], row["y_27"]),
    #         (row["x_42"], row["y_42"]),
    #         (row["x_22"], row["y_22"]),
    #     ],
    #     img_height,
    # )

    # cor_sup_l = face_polygon_svg(
    #     [
    #         (row["x_28"], row["y_28"]),
    #         (row["x_19"], row["y_19"]),
    #         (row["x_20"], row["y_20"]),
    #     ],
    #     img_height,
    # )

    # cor_sup_r = face_polygon_svg(
    #     [
    #         (row["x_28"], row["y_28"]),
    #         (row["x_23"], row["y_23"]),
    #         (row["x_24"], row["y_24"]),
    #     ],
    #     img_height,
    # )

    # lev_lab_sup_l = face_polygon_svg(
    #     [
    #         (row["x_41"], row["y_41"]),
    #         (row["x_40"], row["y_40"]),
    #         (row["x_49"], row["y_49"]),
    #     ],
    #     img_height,
    # )

    # lev_lab_sup_r = face_polygon_svg(
    #     [
    #         (row["x_47"], row["y_47"]),
    #         (row["x_46"], row["y_46"]),
    #         (row["x_53"], row["y_53"]),
    #     ],
    #     img_height,
    # )

    # lev_lab_sup_an_l = face_polygon_svg(
    #     [
    #         (row["x_39"], row["y_39"]),
    #         (row["x_49"], row["y_49"]),
    #         (row["x_31"], row["y_31"]),
    #     ],
    #     img_height,
    # )

    # lev_lab_sup_an_r = face_polygon_svg(
    #     [
    #         (row["x_35"], row["y_35"]),
    #         (row["x_42"], row["y_42"]),
    #         (row["x_53"], row["y_53"]),
    #     ],
    #     img_height,
    # )

    # zyg_maj_l = face_polygon_svg(
    #     [
    #         (row["x_48"], row["y_48"]),
    #         (row["x_3"], row["y_3"]),
    #         (row["x_2"], row["y_2"]),
    #     ],
    #     img_height,
    # )

    # zyg_maj_r = face_polygon_svg(
    #     [
    #         (row["x_54"], row["y_54"]),
    #         (row["x_13"], row["y_13"]),
    #         (row["x_14"], row["y_14"]),
    #     ],
    #     img_height,
    # )

    # bucc_l = face_polygon_svg(
    #     [
    #         (row["x_48"], row["y_48"]),
    #         (row["x_5"], row["y_50"]),
    #         (row["x_5"], row["y_57"]),
    #     ],
    #     img_height,
    # )

    # bucc_r = face_polygon_svg(
    #     [
    #         (row["x_54"], row["y_54"]),
    #         (row["x_11"], row["y_52"]),
    #         (row["x_11"], row["y_57"]),
    #     ],
    #     img_height,
    # )

    # width_l = (row["y_21"] - row["y_39"]) / 2

    # orb_oc_l = face_polygon_svg(
    #     [
    #         (row["x_36"] - width_l / 3, row["y_36"] + width_l / 2),
    #         (row["x_36"], row["y_36"] + width_l),
    #         (row["x_37"], row["y_37"] + width_l),
    #         (row["x_38"], row["y_38"] + width_l),
    #         (row["x_39"], row["y_39"] + width_l),
    #         (row["x_39"] + width_l / 3, row["y_39"] + width_l / 2),
    #         (row["x_39"] + width_l / 2, row["y_39"]),
    #         (row["x_39"] + width_l / 3, row["y_39"] - width_l / 2),
    #         (row["x_39"], row["y_39"] - width_l),
    #         (row["x_40"], row["y_40"] - width_l),
    #         (row["x_41"], row["y_41"] - width_l),
    #         (row["x_36"], row["y_36"] - width_l),
    #         (row["x_36"] - width_l / 3, row["y_36"] - width_l / 2),
    #         (row["x_36"] - width_l / 2, row["y_36"]),
    #     ],
    #     img_height,
    # )

    # orb_oc_l_inner = face_polygon_svg(
    #     [
    #         (row["x_36"] - width_l / 6, row["y_36"] + width_l / 5),
    #         (row["x_36"], row["y_36"] + width_l / 2),
    #         (row["x_37"], row["y_37"] + width_l / 2),
    #         (row["x_38"], row["y_38"] + width_l / 2),
    #         (row["x_39"], row["y_39"] + width_l / 2),
    #         (row["x_39"] + width_l / 6, row["y_39"] + width_l / 5),
    #         (row["x_39"] + width_l / 5, row["y_39"]),
    #         (row["x_39"] + width_l / 6, row["y_39"] - width_l / 5),
    #         (row["x_39"], row["y_39"] - width_l),
    #         (row["x_40"], row["y_40"] - width_l),
    #         (row["x_41"], row["y_41"] - width_l),
    #         (row["x_36"], row["y_36"] - width_l),
    #         (row["x_36"] - width_l / 6, row["y_36"] - width_l / 5),
    #         (row["x_36"] - width_l / 5, row["y_36"]),
    #     ],
    #     img_height,
    # )

    # width_l2 = (row["y_38"] - row["y_2"]) / 1.5

    # orb_oc_l_outer = face_polygon_svg(
    #     [
    #         (row["x_39"] + width_l / 2, row["y_39"] + width_l / 2),
    #         (row["x_39"], row["y_39"] - width_l),
    #         (row["x_40"], row["y_40"] - width_l2),
    #         (row["x_41"], row["y_41"] - width_l2),
    #         (row["x_36"], row["y_36"] - width_l2),
    #         (row["x_36"] - width_l2 / 3, row["y_36"] - width_l2 / 2),
    #         (row["x_36"] - width_l / 2, row["y_36"]),
    #     ],
    #     img_height,
    # )

    # width_r = (row["y_23"] - row["y_43"]) / 2

    # orb_oc_r = face_polygon_svg(
    #     [
    #         (row["x_42"] - width_r / 3, row["y_42"] + width_r / 2),
    #         (row["x_42"], row["y_42"] + width_r),
    #         (row["x_43"], row["y_43"] + width_r),
    #         (row["x_44"], row["y_44"] + width_r),
    #         (row["x_45"], row["y_45"] + width_r),
    #         (row["x_45"] + width_r / 3, row["y_45"] + width_r / 2),
    #         (row["x_45"] + width_r / 2, row["y_45"]),
    #         (row["x_45"] + width_r / 3, row["y_45"] - width_r / 2),
    #         (row["x_45"], row["y_45"] - width_r),
    #         (row["x_46"], row["y_46"] - width_r),
    #         (row["x_47"], row["y_47"] - width_r),
    #         (row["x_42"], row["y_42"] - width_r),
    #         (row["x_42"] - width_l / 3, row["y_42"] - width_r / 2),
    #         (row["x_42"] - width_r / 2, row["y_42"]),
    #     ],
    #     img_height,
    # )

    # orb_oc_r_inner = face_polygon_svg(
    #     [
    #         (row["x_42"] - width_r / 6, row["y_42"] + width_r / 5),
    #         (row["x_42"], row["y_42"] + width_r / 2),
    #         (row["x_43"], row["y_43"] + width_r / 2),
    #         (row["x_44"], row["y_44"] + width_r / 2),
    #         (row["x_45"], row["y_45"] + width_r / 2),
    #         (row["x_45"] + width_r / 6, row["y_45"] + width_r / 5),
    #         (row["x_45"] + width_r / 5, row["y_45"]),
    #         (row["x_45"] + width_r / 6, row["y_45"] - width_r / 5),
    #         (row["x_45"], row["y_45"] - width_r / 2),
    #         (row["x_46"], row["y_46"] - width_r / 2),
    #         (row["x_47"], row["y_47"] - width_r / 2),
    #         (row["x_42"], row["y_42"] - width_r / 2),
    #         (row["x_42"] - width_l / 6, row["y_42"] - width_r / 5),
    #         (row["x_42"] - width_r / 5, row["y_42"]),
    #     ],
    #     img_height,
    # )

    # width_r2 = (row["y_44"] - row["y_14"]) / 1.5

    # orb_oc_r_outer = face_polygon_svg(
    #     [
    #         (row["x_42"] - width_r / 2, row["y_42"]),
    #         (row["x_47"], row["y_47"] - width_r2),
    #         (row["x_46"], row["y_46"] - width_r2),
    #         (row["x_45"], row["y_45"] - width_r2),
    #         (row["x_45"] + width_r2 / 3, row["y_45"] - width_r2 / 2),
    #         (row["x_45"] + width_r / 2, row["y_45"]),
    #     ],
    #     img_height,
    # )

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
            color = cmap.as_hex()[
                int(row[aus[muscle_au_dict[muscle]]] * heatmap_resolution)
            ]
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
        line_width (int): (optional) width of line if outputting a plotly figure instance
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
        size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + tdy
    )

    # Y-Axis | drawn in green
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = (
        size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
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
