"""Regression test: detectors accept ``device='auto'``.

Bug: when loading the emotion / identity / AU sub-models, ``Detector`` (v1) and
``MPDetector`` called ``torch.load`` with ``map_location=device`` — the *raw*
constructor argument — instead of ``self.device`` (which ``set_torch_device``
resolves). With ``device='auto'`` that passed the literal string ``"auto"`` to
``torch.load``, raising ``RuntimeError: don't know how to restore data location
... tagged with auto``. The docs (installation page) explicitly recommend
``device='auto'``, so this hit the documented path.

``Detectorv2`` was never affected (it never passes the raw arg to ``torch.load``)
but is covered here so it stays that way.
"""

import pytest
import torch

from feat.detector import Detector
from feat.detector_v2 import Detectorv2
from feat.MPDetector import MPDetector


@pytest.mark.network
def test_detector_device_auto_constructs():
    det = Detector(device="auto")
    assert isinstance(det.device, torch.device)
    assert det.device.type in {"cuda", "mps", "cpu"}


@pytest.mark.network
def test_mpdetector_device_auto_constructs():
    det = MPDetector(face_model="retinaface", device="auto")
    assert isinstance(det.device, torch.device)
    assert det.device.type in {"cuda", "mps", "cpu"}


@pytest.mark.network
def test_detectorv2_device_auto_constructs():
    det = Detectorv2(device="auto")
    assert isinstance(det.device, torch.device)
    assert det.device.type in {"cuda", "mps", "cpu"}
