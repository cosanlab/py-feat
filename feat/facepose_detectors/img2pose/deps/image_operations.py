import numpy as np


def expand_bbox_rectangle(
    w, h, bbox_x_factor=2.0, bbox_y_factor=2.0, lms=None, expand_forehead=0.3, roll=0
):
    # get a good bbox for the facial landmarks
    min_pt_x = np.min(lms[:, 0], axis=0)
    max_pt_x = np.max(lms[:, 0], axis=0)

    min_pt_y = np.min(lms[:, 1], axis=0)
    max_pt_y = np.max(lms[:, 1], axis=0)

    # find out the bbox of the crop region
    bbox_size_x = int(np.max(max_pt_x - min_pt_x) * bbox_x_factor)
    center_pt_x = 0.5 * min_pt_x + 0.5 * max_pt_x

    bbox_size_y = int(np.max(max_pt_y - min_pt_y) * bbox_y_factor)
    center_pt_y = 0.5 * min_pt_y + 0.5 * max_pt_y

    bbox_min_x, bbox_max_x = (
        center_pt_x - bbox_size_x * 0.5,
        center_pt_x + bbox_size_x * 0.5,
    )

    bbox_min_y, bbox_max_y = (
        center_pt_y - bbox_size_y * 0.5,
        center_pt_y + bbox_size_y * 0.5,
    )

    if abs(roll) > 2.5:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_max_y += expand_forehead_size

    elif roll > 1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_max_x += expand_forehead_size

    elif roll < -1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_min_x -= expand_forehead_size

    else:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_min_y -= expand_forehead_size

    bbox_min_x = bbox_min_x.astype(np.int32)
    bbox_max_x = bbox_max_x.astype(np.int32)
    bbox_min_y = bbox_min_y.astype(np.int32)
    bbox_max_y = bbox_max_y.astype(np.int32)

    # compute necessary padding
    padding_left = abs(min(bbox_min_x, 0))
    padding_top = abs(min(bbox_min_y, 0))
    padding_right = max(bbox_max_x - w, 0)
    padding_bottom = max(bbox_max_y - h, 0)

    # crop the image properly by computing proper crop bounds
    crop_left = 0 if padding_left > 0 else bbox_min_x
    crop_top = 0 if padding_top > 0 else bbox_min_y
    crop_right = w if padding_right > 0 else bbox_max_x
    crop_bottom = h if padding_bottom > 0 else bbox_max_y

    return np.array([crop_left, crop_top, crop_right, crop_bottom])


def bbox_is_dict(bbox):
    # check if the bbox is a not dict and convert it if needed
    if not isinstance(bbox, dict):
        temp_bbox = {}
        temp_bbox["left"] = bbox[0]
        temp_bbox["top"] = bbox[1]
        temp_bbox["right"] = bbox[2]
        temp_bbox["bottom"] = bbox[3]
        bbox = temp_bbox

    return bbox
