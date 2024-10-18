# %%
from feat.MPDetector import MPDetector
import os
from feat.utils.io import get_test_data_path
import cProfile
import pstats

multi_face = os.path.join(get_test_data_path(), "multi_face.jpg")

detector = MPDetector(device="mps", emotion_model="resmasknet", identity_model="facenet")

# detector.detect(multi_face, data_type='image')


# %%

with cProfile.Profile() as pr:
    detector.detect(multi_face, data_type="image")

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename="MPDetector_Profile.prof")
