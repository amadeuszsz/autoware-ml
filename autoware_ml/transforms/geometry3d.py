# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared geometric operations for 3D scene augmentations.

This module holds the math behind the rotation, scale, translation, and flip augmentations. The
pure functions take and return numpy arrays, and the sample helpers apply one augmentation
consistently to the points, boxes, and camera matrices of a Sample. The modality specific
transforms in transforms.point_cloud.geometry, transforms.camera_lidar.geometry, and
transforms.camera.geometry all reuse exactly the same computations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float32

from autoware_ml.datamodule.samples.camera import ImageSet
from autoware_ml.datamodule.samples.sample import Sample


def rotation_matrix(axis: str, angle: float) -> Float32[np.ndarray, "3 3"]:
    """Build a 3x3 rotation matrix for a single axis.

    Args:
        axis: Rotation axis, one of x, y, or z.
        angle: Rotation angle in radians.

    Returns:
        The 3x3 rotation matrix.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    if axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    raise NotImplementedError(f"Unsupported rotation axis: {axis}")


def resolve_rotation_center(
    coord: Float32[np.ndarray, "num_points 3"],
    configured_center: Float32[np.ndarray, " 3"] | None,
) -> Float32[np.ndarray, " 3"]:
    """Resolve the rotation center for a point cloud.

    Args:
        coord: Point coordinates the center is derived from.
        configured_center: Configured center, or None to use the bounding box center.

    Returns:
        The rotation center.
    """
    if configured_center is not None:
        return configured_center
    return (coord.min(axis=0) + coord.max(axis=0)) / 2.0


def rot_scale_trans_matrix(
    rotation: Float32[np.ndarray, "3 3"], scale: float, translation: Float32[np.ndarray, "1 3"]
) -> Float32[np.ndarray, "4 4"]:
    """Compose a 4x4 point space augmentation from rotation, scale, and translation.

    Args:
        rotation: The 3x3 rotation matrix.
        scale: Uniform scale factor.
        translation: Translation vector.

    Returns:
        The 4x4 augmentation matrix.
    """
    augmentation = np.eye(4, dtype=np.float32)
    augmentation[:3, :3] = rotation * scale
    augmentation[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return augmentation


def flip_matrix(flip_x: bool, flip_y: bool) -> Float32[np.ndarray, "4 4"]:
    """Compose a 4x4 flip matrix negating x and or y.

    Args:
        flip_x: Whether to negate the x axis.
        flip_y: Whether to negate the y axis.

    Returns:
        The 4x4 flip matrix.
    """
    flip = np.eye(4, dtype=np.float32)
    if flip_x:
        flip[0, 0] = -1.0
    if flip_y:
        flip[1, 1] = -1.0
    return flip


def sample_rot_scale_trans(
    rot_range: Sequence[float],
    scale_ratio_range: Sequence[float],
    translation_std: Float32[np.ndarray, " 3"] | None,
) -> tuple[Float32[np.ndarray, "3 3"], float, float, Float32[np.ndarray, "1 3"]]:
    """Sample a z rotation, scale, and translation for a global scene transform.

    Args:
        rot_range: Min and max rotation angles in radians.
        scale_ratio_range: Min and max scale factors.
        translation_std: Per axis Gaussian translation std, or None for no translation.

    Returns:
        Tuple of the rotation matrix, the rotation angle, the scale, and the translation.
    """
    rotation = float(np.random.uniform(rot_range[0], rot_range[1]))
    matrix = rotation_matrix("z", rotation)
    scale = float(np.random.uniform(scale_ratio_range[0], scale_ratio_range[1]))
    if translation_std is not None:
        translation = np.random.normal(0.0, translation_std, size=(1, 3)).astype(np.float32)
    else:
        translation = np.zeros((1, 3), dtype=np.float32)
    return matrix, rotation, scale, translation


def sample_bev_flips(
    flip_ratio_bev_horizontal: float, flip_ratio_bev_vertical: float
) -> tuple[bool, bool]:
    """Sample BEV flips.

    Args:
        flip_ratio_bev_horizontal: Probability of flipping the lateral y axis.
        flip_ratio_bev_vertical: Probability of flipping the longitudinal x axis.

    Returns:
        Tuple of the longitudinal and the lateral flip decision as (flip_x, flip_y).
    """
    flip_y = bool(np.random.rand() < flip_ratio_bev_horizontal)
    flip_x = bool(np.random.rand() < flip_ratio_bev_vertical)
    return flip_x, flip_y


def transform_xyz(
    xyz: Float32[np.ndarray, "num_points 3"],
    rotation: Float32[np.ndarray, "3 3"],
    scale: float,
    translation: Float32[np.ndarray, "1 3"],
) -> Float32[np.ndarray, "num_points 3"]:
    """Rotate, scale, and translate point coordinates.

    Args:
        xyz: Point coordinates.
        rotation: The 3x3 rotation matrix.
        scale: Uniform scale factor.
        translation: Translation vector.

    Returns:
        Transformed point coordinates.
    """
    return ((xyz @ rotation.T) * scale + translation).astype(np.float32)


def rotate_xyz_about_center(
    xyz: Float32[np.ndarray, "num_points 3"],
    rotation: Float32[np.ndarray, "3 3"],
    center: Float32[np.ndarray, " 3"],
) -> Float32[np.ndarray, "num_points 3"]:
    """Rotate point coordinates about a center.

    Args:
        xyz: Point coordinates.
        rotation: The 3x3 rotation matrix.
        center: Rotation center.

    Returns:
        Rotated point coordinates.
    """
    return ((xyz - center) @ rotation.T + center).astype(np.float32)


def flip_xyz(
    xyz: Float32[np.ndarray, "num_points 3"], axis: int
) -> Float32[np.ndarray, "num_points 3"]:
    """Negate one axis of point coordinates.

    Args:
        xyz: Point coordinates.
        axis: Axis to negate.

    Returns:
        Flipped point coordinates.
    """
    xyz = xyz.copy()
    xyz[:, axis] *= -1.0
    return xyz


def transform_box_params(
    params: Float32[np.ndarray, "num_boxes num_box_params"],
    rotation: Float32[np.ndarray, "3 3"],
    rotation_angle: float,
    scale: float,
    translation: Float32[np.ndarray, "1 3"],
) -> Float32[np.ndarray, "num_boxes num_box_params"]:
    """Update box parameters consistently with a global rotation, scale, and translation.

    Args:
        params: Box parameters following Box3DFieldIndex without the vertical velocity.
        rotation: The 3x3 rotation matrix applied to the points.
        rotation_angle: Rotation angle in radians around z.
        scale: Uniform scale factor.
        translation: Translation vector.

    Returns:
        Transformed box parameters.
    """
    params = params.copy()
    params[:, :3] = (params[:, :3] @ rotation.T) * scale + translation
    params[:, 3:6] *= scale
    params[:, 6] += rotation_angle
    # Velocities live in the same scaled space as the coordinates
    params[:, 7:9] = (params[:, 7:9] @ rotation[:2, :2].T) * scale
    return params


def rotate_box_params_about_center(
    params: Float32[np.ndarray, "num_boxes num_box_params"],
    rotation: Float32[np.ndarray, "3 3"],
    rotation_angle: float,
    center: Float32[np.ndarray, " 3"],
) -> Float32[np.ndarray, "num_boxes num_box_params"]:
    """Rotate box parameters about a center consistently with the point rotation.

    Args:
        params: Box parameters following Box3DFieldIndex without the vertical velocity.
        rotation: The 3x3 rotation matrix applied to the points.
        rotation_angle: Rotation angle in radians around z.
        center: Rotation center.

    Returns:
        Rotated box parameters.
    """
    params = params.copy()
    params[:, :3] = (params[:, :3] - center) @ rotation.T + center
    params[:, 6] += rotation_angle
    params[:, 7:9] = params[:, 7:9] @ rotation[:2, :2].T
    return params


def flip_box_params(
    params: Float32[np.ndarray, "num_boxes num_box_params"], axis: int
) -> Float32[np.ndarray, "num_boxes num_box_params"]:
    """Flip box parameters across one BEV axis.

    Args:
        params: Box parameters following Box3DFieldIndex without the vertical velocity.
        axis: Flip axis, 0 for longitudinal x and 1 for lateral y.

    Returns:
        Flipped box parameters.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 (x / longitudinal) or 1 (y / lateral), got {axis}")
    params = params.copy()
    if axis == 1:  # lateral flip: negate y, mirror yaw and y velocity
        params[:, 1] *= -1.0
        params[:, 6] *= -1.0
        params[:, 8] *= -1.0
    else:  # longitudinal flip: negate x, reflect yaw and x velocity
        params[:, 0] *= -1.0
        params[:, 6] = np.pi - params[:, 6]
        params[:, 7] *= -1.0
    return params


def transform_sample_points(
    sample: Sample,
    rotation: Float32[np.ndarray, "3 3"],
    scale: float,
    translation: Float32[np.ndarray, "1 3"],
) -> Sample:
    """Rotate, scale, and translate the point cloud and the boxes of a sample.

    Args:
        sample: Sample with a loaded point cloud.
        rotation: The 3x3 rotation matrix.
        scale: Uniform scale factor.
        translation: Translation vector.

    Returns:
        Sample with the transformed point cloud and boxes.
    """
    rotation_angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    update = {
        "points": sample.points.with_coord(
            transform_xyz(sample.points.coord, rotation, scale, translation)
        )
    }
    if sample.boxes is not None:
        update["boxes"] = sample.boxes.with_params(
            transform_box_params(sample.boxes.params, rotation, rotation_angle, scale, translation)
        )
    return sample.replace(**update)


def rotate_sample_about_center(
    sample: Sample,
    rotation: Float32[np.ndarray, "3 3"],
    rotation_angle: float,
    center: Float32[np.ndarray, " 3"],
) -> Sample:
    """Rotate the point cloud and the boxes of a sample about a center.

    Args:
        sample: Sample with a loaded point cloud.
        rotation: The 3x3 rotation matrix around z.
        rotation_angle: Rotation angle in radians around z.
        center: Rotation center.

    Returns:
        Sample with the rotated point cloud and boxes.
    """
    update = {
        "points": sample.points.with_coord(
            rotate_xyz_about_center(sample.points.coord, rotation, center)
        )
    }
    if sample.boxes is not None:
        update["boxes"] = sample.boxes.with_params(
            rotate_box_params_about_center(sample.boxes.params, rotation, rotation_angle, center)
        )
    return sample.replace(**update)


def flip_sample(sample: Sample, axis: int) -> Sample:
    """Flip the point cloud and the boxes of a sample across one BEV axis.

    Args:
        sample: Sample with a loaded point cloud.
        axis: Flip axis, 0 for longitudinal x and 1 for lateral y.

    Returns:
        Sample with the flipped point cloud and boxes.
    """
    update = {"points": sample.points.with_coord(flip_xyz(sample.points.coord, axis))}
    if sample.boxes is not None:
        update["boxes"] = sample.boxes.with_params(flip_box_params(sample.boxes.params, axis))
    return sample.replace(**update)


def update_image_set_matrices(images: ImageSet, aug_inv: Float32[np.ndarray, "4 4"]) -> ImageSet:
    """Keep the camera projection consistent after a lidar space transform.

    Applies the inverse of the 4x4 point space augmentation to lidar2cam and recomputes
    lidar2img from the camera intrinsics.

    Args:
        images: Image set of the sample.
        aug_inv: Inverse of the 4x4 point space augmentation.

    Returns:
        Image set with updated projection matrices.
    """
    lidar2cam = images.lidar2cam.astype(np.float32) @ aug_inv
    lidar2img = images.camera_intrinsics.astype(np.float32) @ lidar2cam
    return images.model_copy(update={"lidar2cam": lidar2cam, "lidar2img": lidar2img})
