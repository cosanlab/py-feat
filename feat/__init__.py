# -*- coding: utf-8 -*-

"""Top-level package for FEAT."""

from __future__ import absolute_import

__author__ = """Jin Hyun Cheong, Tiankang Xie, Sophie Byrne, Eshin Jolly, Luke Chang """
__email__ = "jcheong0428@gmail.com, eshin.jolly@gmail.com, luke.j.chang@dartmouth.edu"
__all__ = ["detector", "data", "utils", "plotting", "transforms", "__version__"]

from .data import Fex
from .detector import Detector
from .utils.io import read_fex
from .version import __version__
