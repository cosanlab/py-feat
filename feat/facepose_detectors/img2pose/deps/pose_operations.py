import torch
from scipy.spatial.transform import Rotation
from .image_operations import expand_bbox_rectangle
import kornia
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="kornia")

# def get_bbox_intrinsics(image_intrinsics, bbox):
#     # crop principle point of view
#     bbox_center_x = bbox["left"] + ((bbox["right"] - bbox["left"]) // 2)
#     bbox_center_y = bbox["top"] + ((bbox["bottom"] - bbox["top"]) // 2)

#     # create a camera intrinsics from the bbox center
#     bbox_intrinsics = image_intrinsics.copy()
#     bbox_intrinsics[0, 2] = bbox_center_x
#     bbox_intrinsics[1, 2] = bbox_center_y

#     return bbox_intrinsics


def get_bbox_intrinsics(image_intrinsics, bbox):
    # crop principle point of view
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # create a camera intrinsics from the bbox center
    bbox_intrinsics = image_intrinsics.clone()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics

    # def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    #     # check if bbox is np or dict
    #     bbox = bbox_is_dict(bbox)

    #     # rotation vector
    #     rvec = pose[:3].copy()

    #     # translation and scale vector
    #     tvec = pose[3:].copy()

    #     # get camera intrinsics using bbox
    #     bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    #     # focal length
    #     focal_length = image_intrinsics[0, 0]

    #     # bbox_size
    #     bbox_width = bbox["right"] - bbox["left"]
    #     bbox_height = bbox["bottom"] - bbox["top"]
    #     bbox_size = bbox_width + bbox_height

    #     # adjust scale
    #     tvec[2] *= focal_length / bbox_size

    #     # project crop points using the crop camera intrinsics
    #     projected_point = bbox_intrinsics.dot(tvec.T)

    #     # reverse the projected points using the full image camera intrinsics
    #     tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))

    #     # same for rotation
    #     rmat = Rotation.from_rotvec(rvec).as_matrix()
    #     # project crop points using the crop camera intrinsics
    #     projected_point = bbox_intrinsics.dot(rmat)
    #     # reverse the projected points using the full image camera intrinsics
    #     rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    #     rvec = Rotation.from_matrix(rmat).as_rotvec()

    #     return np.concatenate([rvec, tvec])

    # def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    #     # check if bbox is a tensor or dict
    #     if isinstance(bbox, dict):
    #         bbox = torch.tensor([bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]], dtype=torch.float32)

    #     # rotation vector
    #     rvec = pose[:3]

    #     # translation and scale vector
    #     tvec = pose[3:]

    #     # get camera intrinsics using bbox
    #     bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    #     # focal length
    #     focal_length = image_intrinsics[0, 0]

    #     # bbox_size
    #     bbox_width = bbox[2] - bbox[0]
    #     bbox_height = bbox[3] - bbox[1]
    #     bbox_size = bbox_width + bbox_height

    #     # adjust scale
    #     tvec[2] *= focal_length / bbox_size

    #     # project crop points using the crop camera intrinsics
    #     projected_point = bbox_intrinsics @ tvec.unsqueeze(-1)

    #     # reverse the projected points using the full image camera intrinsics
    #     tvec = (projected_point.squeeze() @ torch.inverse(image_intrinsics.T))

    #     # same for rotation
    #     rmat = Rotation.from_rotvec(rvec.cpu().numpy()).as_matrix()
    #     rmat = torch.tensor(rmat, device=pose.device)

    #     # project crop points using the crop camera intrinsics
    #     projected_point = bbox_intrinsics @ rmat

    #     # reverse the projected points using the full image camera intrinsics
    #     rmat = torch.inverse(image_intrinsics) @ projected_point
    #     rvec = torch.tensor(Rotation.from_matrix(rmat.cpu().numpy()).as_rotvec(), device=pose.device)

    #     return torch.cat([rvec, tvec])

    def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
        # check if bbox is a tensor or dict
        if isinstance(bbox, dict):
            bbox = torch.tensor(
                [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]],
                dtype=torch.float32,
            )

        # rotation vector
        rvec = pose[:3]

        # translation and scale vector
        tvec = pose[3:]

        # get camera intrinsics using bbox
        bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

        # focal length
        focal_length = image_intrinsics[0, 0]

        # bbox_size
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_size = bbox_width + bbox_height

        # adjust scale
        tvec[2] *= focal_length / bbox_size

        # project crop points using the crop camera intrinsics
        projected_point = bbox_intrinsics @ tvec.unsqueeze(-1)

        # reverse the projected points using the full image camera intrinsics
        tvec = projected_point.squeeze() @ torch.inverse(image_intrinsics.T)

        # same for rotation
        # Detach from computation graph and convert to NumPy array
        rmat = Rotation.from_rotvec(rvec.detach().cpu().numpy()).as_matrix()
        rmat = torch.tensor(rmat, device=pose.device)

        # project crop points using the crop camera intrinsics
        projected_point = bbox_intrinsics @ rmat
        # reverse the projected points using the full image camera intrinsics
        rmat = torch.inverse(image_intrinsics) @ projected_point
        rvec = torch.tensor(
            Rotation.from_matrix(rmat.cpu().numpy()).as_rotvec(), device=pose.device
        )

        return torch.cat([rvec, tvec])


# def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
#     # check if bbox is a tensor or dict
#     if isinstance(bbox, dict):
#         bbox = torch.tensor([bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]], dtype=torch.float32)

#     # Ensure pose and image_intrinsics are float32
#     pose = pose.to(torch.float32)
#     image_intrinsics = image_intrinsics.to(torch.float32)

#     # rotation vector
#     rvec = pose[:3]

#     # translation and scale vector
#     tvec = pose[3:]

#     # get camera intrinsics using bbox
#     bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

#     # focal length
#     focal_length = image_intrinsics[0, 0]

#     # bbox_size
#     bbox_width = bbox[2] - bbox[0]
#     bbox_height = bbox[3] - bbox[1]
#     bbox_size = bbox_width + bbox_height

#     # adjust scale
#     tvec[2] *= focal_length / bbox_size

#     # project crop points using the crop camera intrinsics
#     projected_point = bbox_intrinsics @ tvec.unsqueeze(-1)

#     # reverse the projected points using the full image camera intrinsics
#     tvec = (projected_point.squeeze() @ torch.inverse(image_intrinsics.T))

#     # same for rotation
#     # Detach from computation graph and convert to NumPy array
#     rmat = Rotation.from_rotvec(rvec.detach().cpu().numpy()).as_matrix()
#     rmat = torch.tensor(rmat, device=pose.device, dtype=torch.float32)

#     # project crop points using the crop camera intrinsics
#     projected_point = bbox_intrinsics @ rmat
#     # reverse the projected points using the full image camera intrinsics
#     rmat = torch.inverse(image_intrinsics) @ projected_point
#     rvec = torch.tensor(Rotation.from_matrix(rmat.cpu().numpy()).as_rotvec(), device=pose.device, dtype=torch.float32)

#     return torch.cat([rvec, tvec])

# def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
#     # check if bbox is a tensor or dict
#     if isinstance(bbox, dict):
#         bbox = torch.tensor([bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]], dtype=torch.float32)

#     # Ensure pose and image_intrinsics are float32
#     pose = pose.to(torch.float32)
#     image_intrinsics = image_intrinsics.to(torch.float32)

#     # rotation vector
#     rvec = pose[:3]

#     # translation and scale vector
#     tvec = pose[3:]

#     # get camera intrinsics using bbox
#     bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

#     # focal length
#     focal_length = image_intrinsics[0, 0]

#     # bbox_size
#     bbox_width = bbox[2] - bbox[0]
#     bbox_height = bbox[3] - bbox[1]
#     bbox_size = bbox_width + bbox_height

#     # adjust scale
#     tvec[2] *= focal_length / bbox_size

#     # project crop points using the crop camera intrinsics
#     projected_point = bbox_intrinsics @ tvec.unsqueeze(-1)

#     # reverse the projected points using the full image camera intrinsics
#     tvec = (projected_point.squeeze() @ torch.inverse(image_intrinsics.T))

#     # same for rotation
#     # Detach from computation graph and convert to NumPy array
#     rmat = Rotation.from_rotvec(rvec.detach().cpu().numpy()).as_matrix()
#     rmat = torch.tensor(rmat, device=pose.device, dtype=torch.float32)

#     # project crop points using the crop camera intrinsics
#     projected_point = bbox_intrinsics @ rmat
#     # reverse the projected points using the full image camera intrinsics
#     rmat = torch.inverse(image_intrinsics) @ projected_point
#     rvec = torch.tensor(Rotation.from_matrix(rmat.cpu().numpy()).as_rotvec(), device=pose.device, dtype=torch.float32)

#     return torch.cat([rvec, tvec])


def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    # check if bbox is a tensor or dict
    if isinstance(bbox, dict):
        bbox = torch.tensor(
            [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]],
            dtype=torch.float32,
        )

    # Ensure pose and image_intrinsics are float32
    pose = pose.to(torch.float32)
    image_intrinsics = image_intrinsics.to(torch.float32)

    # rotation vector
    rvec = pose[:3].unsqueeze(0)  # Add batch dimension

    # translation and scale vector
    tvec = pose[3:]

    # get camera intrinsics using bbox
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    # focal length
    focal_length = image_intrinsics[0, 0]

    # bbox_size
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_size = bbox_width + bbox_height

    # adjust scale
    tvec[2] *= focal_length / bbox_size

    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics @ tvec.unsqueeze(-1)

    # reverse the projected points using the full image camera intrinsics
    tvec = projected_point.squeeze() @ torch.inverse(image_intrinsics.T)

    # Convert rotation vector to rotation matrix using Kornia
    rmat = kornia.geometry.conversions.axis_angle_to_rotation_matrix(rvec).squeeze(0)

    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics @ rmat
    # reverse the projected points using the full image camera intrinsics
    rmat = torch.inverse(image_intrinsics) @ projected_point
    rvec = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
        rmat.unsqueeze(0)
    ).squeeze(0)

    return torch.cat([rvec, tvec])


# def plot_3d_landmark(verts, campose, intrinsics):
#     lm_3d_trans = transform_points(verts, campose)

#     # project to image plane
#     lms_3d_trans_proj = intrinsics.dot(lm_3d_trans.T).T
#     lms_projected = (
#         lms_3d_trans_proj[:, :2] / np.tile(lms_3d_trans_proj[:, 2], (2, 1)).T
#     )

#     return lms_projected, lms_3d_trans_proj


def plot_3d_landmark(verts, campose, intrinsics):
    lm_3d_trans = transform_points(verts, campose)

    # project to image plane
    lms_3d_trans_proj = (intrinsics @ lm_3d_trans.T).T
    lms_projected = lms_3d_trans_proj[:, :2] / lms_3d_trans_proj[:, 2].unsqueeze(1)

    return lms_projected, lms_3d_trans_proj


# def transform_points(points, pose):
#     return points.dot(Rotation.from_rotvec(pose[:3]).as_matrix().T) + pose[3:]

# def transform_points(points, pose):
#     rmat = Rotation.from_rotvec(pose[:3].cpu().numpy()).as_matrix()
#     rmat = torch.tensor(rmat, device=pose.device)
#     return points @ rmat.T + pose[3:]

# def transform_points(points, pose):
#     # Ensure pose and points are float32
#     points = points.to(torch.float32)
#     pose = pose.to(torch.float32)

#     # Detach from computation graph and convert to NumPy array
#     rmat = Rotation.from_rotvec(pose[:3].detach().cpu().numpy()).as_matrix()
#     rmat = torch.tensor(rmat, device=pose.device, dtype=torch.float32)

#     return points @ rmat.T + pose[3:]


def transform_points(points, pose):
    # Ensure pose and points are float32
    points = points.to(torch.float32)
    pose = pose.to(torch.float32)

    # Convert rotation vector to rotation matrix using Kornia
    rmat = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
        pose[:3].unsqueeze(0)
    ).squeeze(0)

    return points @ rmat.T + pose[3:]


# def transform_pose_global_project_bbox(
#     boxes,
#     dofs,
#     pose_mean,
#     pose_stddev,
#     image_shape,
#     threed_68_points=None,
#     bbox_x_factor=1.1,
#     bbox_y_factor=1.1,
#     expand_forehead=0.3,
# ):
#     if len(dofs) == 0:
#         return boxes, dofs

#     device = dofs.device

#     boxes = boxes.cpu().numpy()
#     dofs = dofs.cpu().numpy()

#     threed_68_points = threed_68_points.numpy()

#     (h, w) = image_shape
#     global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

#     if threed_68_points is not None:
#         threed_68_points = threed_68_points

#     pose_mean = pose_mean.numpy()
#     pose_stddev = pose_stddev.numpy()

#     dof_mean = pose_mean
#     dof_std = pose_stddev
#     dofs = dofs * dof_std + dof_mean

#     projected_boxes = []
#     global_dofs = []

#     for i in range(dofs.shape[0]):
#         global_dof = pose_bbox_to_full_image(dofs[i], global_intrinsics, boxes[i])
#         global_dofs.append(global_dof)

#         if threed_68_points is not None:
#             # project points and get bbox
#             projected_lms, _ = plot_3d_landmark(
#                 threed_68_points, global_dof, global_intrinsics
#             )
#             projected_bbox = expand_bbox_rectangle(
#                 w,
#                 h,
#                 bbox_x_factor=bbox_x_factor,
#                 bbox_y_factor=bbox_y_factor,
#                 lms=projected_lms,
#                 roll=global_dof[2],
#                 expand_forehead=expand_forehead,
#             )
#         else:
#             projected_bbox = boxes[i]

#         projected_boxes.append(projected_bbox)

#     global_dofs = torch.from_numpy(np.asarray(global_dofs)).float()
#     projected_boxes = torch.from_numpy(np.asarray(projected_boxes)).float()

#     return projected_boxes.to(device), global_dofs.to(device)


def transform_pose_global_project_bbox(
    boxes,
    dofs,
    pose_mean,
    pose_stddev,
    image_shape,
    threed_68_points=None,
    bbox_x_factor=1.1,
    bbox_y_factor=1.1,
    expand_forehead=0.3,
):
    if len(dofs) == 0:
        return boxes, dofs

    device = dofs.device

    (h, w) = image_shape
    global_intrinsics = torch.tensor(
        [[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    )

    if threed_68_points is not None:
        threed_68_points = threed_68_points.to(device)

    pose_mean = pose_mean.to(device)
    pose_stddev = pose_stddev.to(device)

    dof_mean = pose_mean
    dof_std = pose_stddev
    dofs = dofs * dof_std + dof_mean

    projected_boxes = []
    global_dofs = []

    for i in range(dofs.shape[0]):
        global_dof = pose_bbox_to_full_image(dofs[i], global_intrinsics, boxes[i])
        global_dofs.append(global_dof)

        if threed_68_points is not None:
            # project points and get bbox
            projected_lms, _ = plot_3d_landmark(
                threed_68_points, global_dof, global_intrinsics
            )
            projected_bbox = expand_bbox_rectangle(
                w,
                h,
                bbox_x_factor=bbox_x_factor,
                bbox_y_factor=bbox_y_factor,
                lms=projected_lms,
                roll=global_dof[2],
                expand_forehead=expand_forehead,
            )
        else:
            projected_bbox = boxes[i]

        projected_boxes.append(projected_bbox)

    global_dofs = torch.stack(global_dofs)
    projected_boxes = torch.stack(projected_boxes)

    return projected_boxes, global_dofs
