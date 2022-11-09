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
# FEAT_FACEBOX_COLUMNS = ['FaceRectX1','FaceRectY1','FaceRectX2','FaceRectY2']
FEAT_TIME_COLUMNS = ["frame"]

# FACET columns
FACET_EMOTION_COLUMNS = [
    "Joy",
    "Anger",
    "Surprise",
    "Fear",
    "Contempt",
    "Disgust",
    "Sadness",
    "Confusion",
    "Frustration",
    "Neutral",
    "Positive",
    "Negative",
]
FACET_FACEBOX_COLUMNS = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
FACET_TIME_COLUMNS = ["Timestamp", "MediaTime", "FrameNo", "FrameTime"]
FACET_FACEPOSE_COLUMNS = ["Pitch", "Roll", "Yaw"]
FACET_DESIGN_COLUMNS = ["StimulusName", "SlideType", "EventSource", "Annotation"]

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


def set_torch_device(device="auto"):
    """Helper function to set device for pytorch model"""

    if not isinstance(device, torch.device):
        if device not in ["cpu", "cuda", "mps", "auto"]:
            raise ValueError("Device must be ['cpu', 'cuda', 'mps', 'auto']")

        if device == "auto":
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