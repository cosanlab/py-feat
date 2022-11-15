"""
py-feat helper functions and variables
"""

import torch

""" DEFINE IMPORTANT VARIABLES """
# FEAT columns
FEAT_EMOTION_MAPPER = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}
FEAT_EMOTION_COLUMNS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]
FEAT_FACEBOX_COLUMNS = [
    "FaceRectX",
    "FaceRectY",
    "FaceRectWidth",
    "FaceRectHeight",
    "FaceScore",
]
FEAT_TIME_COLUMNS = ["frame"]
FEAT_FACEPOSE_COLUMNS = ["Pitch", "Roll", "Yaw"]

# OpenFace columns
landmark_length = 68
openface_2d_landmark_columns = [f"x_{i}" for i in range(landmark_length)] + [
    f"y_{i}" for i in range(landmark_length)
]
openface_3d_landmark_columns = (
    [f"X_{i}" for i in range(landmark_length)]
    + [f"Y_{i}" for i in range(landmark_length)]
    + [f"Z_{i}" for i in range(landmark_length)]
)

openface_AU_list = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
openface_AU_intensity = [f"AU" + str(i).zfill(2) + "_r" for i in openface_AU_list]
openface_AU_presence = [f"AU" + str(i).zfill(2) + "_c" for i in openface_AU_list + [28]]
openface_AU_presence.sort()
openface_AU_columns = openface_AU_intensity + openface_AU_presence
openface_time_columns = ["frame", "timestamp"]
openface_gaze_columns = [
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
]
openface_facepose_columns = [
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
]
OPENFACE_ORIG_COLUMNS = (
    openface_time_columns
    + ["confidence", "success"]
    + openface_gaze_columns
    + openface_facepose_columns
    + openface_2d_landmark_columns
    + openface_3d_landmark_columns
    + [
        "p_scale",
        "p_rx",
        "p_ry",
        "p_rz",
        "p_tx",
        "p_ty",
        "p_0",
        "p_1",
        "p_2",
        "p_3",
        "p_4",
        "p_5",
        "p_6",
        "p_7",
        "p_8",
        "p_9",
        "p_10",
        "p_11",
        "p_12",
        "p_13",
        "p_14",
        "p_15",
        "p_16",
        "p_17",
        "p_18",
        "p_19",
        "p_20",
        "p_21",
        "p_22",
        "p_23",
        "p_24",
        "p_25",
        "p_26",
        "p_27",
        "p_28",
        "p_29",
        "p_30",
        "p_31",
        "p_32",
        "p_33",
    ]
    + openface_AU_columns
)


def set_torch_device(device="cpu"):
    """Helper function to set device for pytorch model"""

    if not isinstance(device, torch.device):
        if device not in ["cpu", "cuda", "mps", "auto"]:
            raise ValueError("Device must be ['cpu', 'cuda', 'mps', 'auto']")

        if device == "auto":
            # FIXME: This currently doesn't work on mac's where mps is available because
            # it results in a mix of cpu and mps operations which cause failures. E.g.
            # when we call torch.cat() inside of image_operations.decode from
            # FaceBoxes_tests.py(128)

            # In this case priors are on `cpu`, loc is on `mps`, variances is a list (so
            # cpu I assume). loc also contains all nans

            # This causes retinaface to fail to detect a face properly and tests to fail

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = device
        return torch.device(device)

    else:
        return device
