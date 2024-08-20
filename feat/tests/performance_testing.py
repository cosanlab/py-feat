# %%
from feat.FastDetector import FastDetector
from feat.utils.image_operations import convert_image_to_tensor
from feat.data import _inverse_face_transform, _inverse_landmark_transform
import torch
import os
from torchvision.io import read_image
from feat.utils.io import get_test_data_path
import numpy as np
import cProfile
import pstats

multi_face = os.path.join(get_test_data_path(), "multi_face.jpg")

detector = FastDetector()

detector.detect_image(multi_face)




# %%

with cProfile.Profile() as pr:
    detector.detect_image(multi_face)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='FastDetector_Profile.prof')