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

"""Camera and lidar fusion transforms.

This module contains the calibration status augmentations, the lidar to image fusion, the
fusion preview writer, and the multiview image space augmentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import transforms3d
from jaxtyping import Bool, Float32, UInt8
from matplotlib.colors import Colormap
from scipy.stats import truncnorm

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.loading import camera_frame_by_channel
from autoware_ml.transforms.camera.utils import copy_calibration_data
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus


class CalibrationMisalignment(BaseTransform):
    """Calibration misalignment augmentation for camera and lidar calibration.

    Each rotation (roll, pitch, yaw) and translation (x, y, z) component has separate negative
    and positive ranges. During augmentation one of the two ranges is randomly selected for
    each component, and each component can be individually activated or deactivated.

    All parameters are specified as positive magnitudes. The _neg suffix indicates the value
    will be negated when applied, which keeps min < max intuitive in the config. When the
    transform runs it perturbs the lidar to camera transformation of the calibration data and
    sets the ground truth status to miscalibrated. When it is skipped the status is set to
    calibrated.
    """

    _required_fields = ["calibration"]

    def __init__(
        self,
        *,
        p: float,
        activate_roll: bool = False,
        activate_pitch: bool = False,
        activate_yaw: bool = False,
        activate_x: bool = False,
        activate_y: bool = False,
        activate_z: bool = False,
        min_roll_neg: float = 0.0,
        max_roll_neg: float = 0.0,
        min_roll_pos: float = 0.0,
        max_roll_pos: float = 0.0,
        min_pitch_neg: float = 0.0,
        max_pitch_neg: float = 0.0,
        min_pitch_pos: float = 0.0,
        max_pitch_pos: float = 0.0,
        min_yaw_neg: float = 0.0,
        max_yaw_neg: float = 0.0,
        min_yaw_pos: float = 0.0,
        max_yaw_pos: float = 0.0,
        min_x_neg: float = 0.0,
        max_x_neg: float = 0.0,
        min_x_pos: float = 0.0,
        max_x_pos: float = 0.0,
        min_y_neg: float = 0.0,
        max_y_neg: float = 0.0,
        min_y_pos: float = 0.0,
        max_y_pos: float = 0.0,
        min_z_neg: float = 0.0,
        max_z_neg: float = 0.0,
        min_z_pos: float = 0.0,
        max_z_pos: float = 0.0,
    ) -> None:
        """Initialize the CalibrationMisalignment transform.

        Args:
            p: Probability of applying the augmentation.
            activate_roll: Whether to apply roll miscalibration.
            activate_pitch: Whether to apply pitch miscalibration.
            activate_yaw: Whether to apply yaw miscalibration.
            activate_x: Whether to apply x translation miscalibration.
            activate_y: Whether to apply y translation miscalibration.
            activate_z: Whether to apply z translation miscalibration.
            min_roll_neg: Min magnitude for negative roll in degrees (applied as negative).
            max_roll_neg: Max magnitude for negative roll in degrees (applied as negative).
            min_roll_pos: Min magnitude for positive roll in degrees.
            max_roll_pos: Max magnitude for positive roll in degrees.
            min_pitch_neg: Min magnitude for negative pitch in degrees (applied as negative).
            max_pitch_neg: Max magnitude for negative pitch in degrees (applied as negative).
            min_pitch_pos: Min magnitude for positive pitch in degrees.
            max_pitch_pos: Max magnitude for positive pitch in degrees.
            min_yaw_neg: Min magnitude for negative yaw in degrees (applied as negative).
            max_yaw_neg: Max magnitude for negative yaw in degrees (applied as negative).
            min_yaw_pos: Min magnitude for positive yaw in degrees.
            max_yaw_pos: Max magnitude for positive yaw in degrees.
            min_x_neg: Min magnitude for negative x translation in meters (applied as negative).
            max_x_neg: Max magnitude for negative x translation in meters (applied as negative).
            min_x_pos: Min magnitude for positive x translation in meters.
            max_x_pos: Max magnitude for positive x translation in meters.
            min_y_neg: Min magnitude for negative y translation in meters (applied as negative).
            max_y_neg: Max magnitude for negative y translation in meters (applied as negative).
            min_y_pos: Min magnitude for positive y translation in meters.
            max_y_pos: Max magnitude for positive y translation in meters.
            min_z_neg: Min magnitude for negative z translation in meters (applied as negative).
            max_z_neg: Max magnitude for negative z translation in meters (applied as negative).
            min_z_pos: Min magnitude for positive z translation in meters.
            max_z_pos: Max magnitude for positive z translation in meters.
        """
        self.activate_roll = activate_roll
        self.activate_pitch = activate_pitch
        self.activate_yaw = activate_yaw
        self.activate_x = activate_x
        self.activate_y = activate_y
        self.activate_z = activate_z

        ranges = {
            "roll_neg": (min_roll_neg, max_roll_neg),
            "roll_pos": (min_roll_pos, max_roll_pos),
            "pitch_neg": (min_pitch_neg, max_pitch_neg),
            "pitch_pos": (min_pitch_pos, max_pitch_pos),
            "yaw_neg": (min_yaw_neg, max_yaw_neg),
            "yaw_pos": (min_yaw_pos, max_yaw_pos),
            "x_neg": (min_x_neg, max_x_neg),
            "x_pos": (min_x_pos, max_x_pos),
            "y_neg": (min_y_neg, max_y_neg),
            "y_pos": (min_y_pos, max_y_pos),
            "z_neg": (min_z_neg, max_z_neg),
            "z_pos": (min_z_pos, max_z_pos),
        }
        for name, (min_value, max_value) in ranges.items():
            self._validate_non_negative(f"min_{name}", min_value)
            self._validate_non_negative(f"max_{name}", max_value)
            self._validate_range(name, min_value, max_value)

        self.min_roll_neg = min_roll_neg
        self.max_roll_neg = max_roll_neg
        self.min_roll_pos = min_roll_pos
        self.max_roll_pos = max_roll_pos
        self.min_pitch_neg = min_pitch_neg
        self.max_pitch_neg = max_pitch_neg
        self.min_pitch_pos = min_pitch_pos
        self.max_pitch_pos = max_pitch_pos
        self.min_yaw_neg = min_yaw_neg
        self.max_yaw_neg = max_yaw_neg
        self.min_yaw_pos = min_yaw_pos
        self.max_yaw_pos = max_yaw_pos
        self.min_x_neg = min_x_neg
        self.max_x_neg = max_x_neg
        self.min_x_pos = min_x_pos
        self.max_x_pos = max_x_pos
        self.min_y_neg = min_y_neg
        self.max_y_neg = max_y_neg
        self.min_y_pos = min_y_pos
        self.max_y_pos = max_y_pos
        self.min_z_neg = min_z_neg
        self.max_z_neg = max_z_neg
        self.min_z_pos = min_z_pos
        self.max_z_pos = max_z_pos
        self.p = p

    def _validate_non_negative(self, name: str, value: float) -> None:
        """Validate that a parameter is non-negative.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            ValueError: If the value is negative.
        """
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (specify as magnitude), got {value}")

    def _validate_range(self, name: str, min_val: float, max_val: float) -> None:
        """Validate that a configured range is well ordered.

        Args:
            name: Range name.
            min_val: Lower bound.
            max_val: Upper bound.

        Raises:
            ValueError: If the lower bound exceeds the upper bound.
        """
        if min_val > max_val:
            raise ValueError(f"min_{name} ({min_val}) must be <= max_{name} ({max_val})")

    def on_skip(self, sample: Sample) -> Sample:
        """Mark the sample as calibrated when the augmentation is skipped.

        Args:
            sample: Sample with a seeded calibration state.

        Returns:
            Sample with the calibrated ground truth status.
        """
        calibration = sample.calibration.model_copy(update={"status": CalibrationStatus.CALIBRATED})
        return sample.replace(calibration=calibration)

    def transform(self, sample: Sample) -> Sample:
        """Apply the calibration misalignment augmentation.

        Args:
            sample: Sample with a seeded calibration state.

        Returns:
            Sample with the perturbed calibration data and the miscalibrated status.
        """
        data = copy_calibration_data(sample.calibration.data)
        noisy_transform, noise = self.alter_calibration(data.lidar_to_camera_transformation)
        data.lidar_to_camera_transformation = noisy_transform
        data.noise = noise
        calibration = sample.calibration.model_copy(
            update={"data": data, "status": CalibrationStatus.MISCALIBRATED}
        )
        return sample.replace(calibration=calibration)

    def bounded_gaussian(
        self, center: float, min_value: float, max_value: float, scale: float
    ) -> float:
        """Generate a value from a truncated normal distribution.

        Args:
            center: Distribution center before truncation.
            min_value: Lower truncation bound.
            max_value: Upper truncation bound.
            scale: Distribution scale parameter.

        Returns:
            Sampled scalar value.

        Raises:
            ValueError: If the bounds are invalid or the scale is non-positive.
        """
        if min_value >= max_value:
            raise ValueError(f"min_value ({min_value}) must be less than max_value ({max_value})")
        if scale <= 0:
            raise ValueError(f"scale ({scale}) must be positive")

        a = (min_value - center) / scale
        b = (max_value - center) / scale
        return truncnorm.rvs(a, b, loc=center, scale=scale)

    def _sample_component(
        self, min_neg: float, max_neg: float, min_pos: float, max_pos: float
    ) -> float:
        """Sample a component value from either the negative or the positive range.

        Randomly selects between the negative and the positive range, then samples from a
        truncated gaussian within that range. All input values are positive magnitudes, and
        values from the negative range are negated after sampling.

        Args:
            min_neg: Minimum magnitude of the negative range (will be negated).
            max_neg: Maximum magnitude of the negative range (will be negated).
            min_pos: Minimum magnitude of the positive range.
            max_pos: Maximum magnitude of the positive range.

        Returns:
            Sampled value, negative when drawn from the negative range.
        """
        use_negative = np.random.rand() > 0.5

        if use_negative:
            min_val, max_val = min_neg, max_neg
            if min_val >= max_val:
                return -min_val
            value = self.bounded_gaussian(
                center=min_val,
                min_value=min_val,
                max_value=max_val,
                scale=(max_val - min_val) / 1.5,
            )
            return -value
        min_val, max_val = min_pos, max_pos
        if min_val >= max_val:
            return min_val
        return self.bounded_gaussian(
            center=min_val,
            min_value=min_val,
            max_value=max_val,
            scale=(max_val - min_val) / 1.5,
        )

    def alter_calibration(
        self, transform: Float32[np.ndarray, "4 4"]
    ) -> tuple[Float32[np.ndarray, "4 4"], Float32[np.ndarray, "4 4"]]:
        """Apply random noise to a 4x4 transformation matrix.

        The noise is applied in the camera frame so that, for example, an x translation shifts
        projected points along the camera x axis (horizontal in the image). Mathematically
        T_noisy = T_noise @ T_l2c, producing the pipeline lidar -> camera -> miscalibration.
        Each activated component randomly selects between its negative and positive range.

        Args:
            transform: The lidar to camera transformation matrix.

        Returns:
            Tuple of the noisy transformation and the applied noise transform.

        Raises:
            ValueError: If the transformation matrix is not 4x4.
        """
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4 matrix, got shape {transform.shape}")

        roll = (
            self._sample_component(
                self.min_roll_neg, self.max_roll_neg, self.min_roll_pos, self.max_roll_pos
            )
            if self.activate_roll
            else 0.0
        )
        pitch = (
            self._sample_component(
                self.min_pitch_neg, self.max_pitch_neg, self.min_pitch_pos, self.max_pitch_pos
            )
            if self.activate_pitch
            else 0.0
        )
        yaw = (
            self._sample_component(
                self.min_yaw_neg, self.max_yaw_neg, self.min_yaw_pos, self.max_yaw_pos
            )
            if self.activate_yaw
            else 0.0
        )

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        tx = (
            self._sample_component(self.min_x_neg, self.max_x_neg, self.min_x_pos, self.max_x_pos)
            if self.activate_x
            else 0.0
        )
        ty = (
            self._sample_component(self.min_y_neg, self.max_y_neg, self.min_y_pos, self.max_y_pos)
            if self.activate_y
            else 0.0
        )
        tz = (
            self._sample_component(self.min_z_neg, self.max_z_neg, self.min_z_pos, self.max_z_pos)
            if self.activate_z
            else 0.0
        )

        rotation_matrix = transforms3d.euler.euler2mat(roll_rad, pitch_rad, yaw_rad, axes="sxyz")

        noise_transform = np.eye(4)
        noise_transform[0:3, 0:3] = rotation_matrix
        noise_transform[0:3, 3] = [tx, ty, tz]

        return noise_transform @ transform, noise_transform


class LidarCameraFusion(BaseTransform):
    """Fuse the lidar points with the calibration image into depth and intensity channels.

    Projects the lidar points onto the image plane of the calibration camera and appends a
    depth and an intensity channel to the color channels, producing a five channel float image
    normalized to [0, 1]. The point cloud of the sample stays untouched.
    """

    _required_fields = ["calibration", "points"]

    def __init__(
        self,
        *,
        max_depth: float = 128.0,
        dilation_size: int = 1,
        ego_box: Sequence[float] | None = None,
        occlusion_adjust_margin: float = 0.01,
    ) -> None:
        """Initialize the LidarCameraFusion transform.

        Args:
            max_depth: Maximum depth of the projected lidar points in meters.
            dilation_size: Size of the dilation kernel used when rendering the points.
            ego_box: Ego chassis box as [x_min, y_min, z_min, x_max, y_max, z_max] used to drop
                points occluded by the vehicle, or None to skip the occlusion filter.
            occlusion_adjust_margin: Distance in meters kept between the camera and an adjusted
                box wall when the camera sits inside the ego box.
        """
        self.max_depth = max_depth
        self.dilation_size = dilation_size
        self.ego_box = ego_box
        self.occlusion_adjust_margin = occlusion_adjust_margin

    def transform(self, sample: Sample) -> Sample:
        """Create the fused image from the calibration camera and the lidar points.

        Args:
            sample: Sample with a loaded calibration image and a loaded point cloud.

        Returns:
            Sample with the fused image set on the calibration state.
        """
        calibration = sample.calibration
        if calibration.image is None:
            raise ValueError("LidarCameraFusion requires a loaded calibration image.")
        xyz = sample.points.coord
        intensities = sample.points.feature(PointFeatureName.INTENSITY)

        fused_image = self._create_fused_image(
            calibration.image,
            xyz,
            intensities,
            calibration.data,
            calibration.affine_transform,
        )
        calibration = calibration.model_copy(update={"fused_image": fused_image})
        return sample.replace(calibration=calibration)

    def _create_fused_image(
        self,
        image: Float32[np.ndarray, "height width channels"],
        xyz: Float32[np.ndarray, "num_points 3"],
        intensities: Float32[np.ndarray, " num_points"],
        calibration_data: CalibrationData,
        affine_transform: Float32[np.ndarray, "3 3"] | None,
    ) -> Float32[np.ndarray, "height width fused_channels"]:
        """Create a fused image with color, depth, and intensity channels.

        Args:
            image: Calibration image with values in [0, 255].
            xyz: Point coordinates in the lidar frame.
            intensities: Point intensities aligned with the coordinates.
            calibration_data: Camera and lidar calibration data.
            affine_transform: Optional image space affine transform.

        Returns:
            Fused five channel image normalized to [0, 1].
        """
        if self.ego_box is not None:
            keep = self._occlusion_keep_mask(
                xyz, calibration_data, affine_transform, image.shape[:2]
            )
            xyz = xyz[keep]
            intensities = intensities[keep]

        point_cloud_ccs = self._transform_points_to_camera(xyz, calibration_data)

        valid_mask = point_cloud_ccs[:, 2] > 0.0
        point_cloud_ccs = point_cloud_ccs[valid_mask]
        intensities = intensities[valid_mask]

        point_cloud_ics = self._project_points_to_image(point_cloud_ccs, calibration_data)

        if affine_transform is not None:
            point_cloud_ics = self._apply_affine_to_points(point_cloud_ics, affine_transform)

        return self._create_lidar_images(image, point_cloud_ics, point_cloud_ccs, intensities)

    def _occlusion_keep_mask(
        self,
        xyz: Float32[np.ndarray, "num_points 3"],
        calibration_data: CalibrationData,
        affine_transform: Float32[np.ndarray, "3 3"] | None,
        image_shape: tuple[int, int],
    ) -> Bool[np.ndarray, " num_points"]:
        """Build the mask of points not occluded by the ego vehicle chassis.

        Ray casting from the camera center drops points whose ray crosses the ego box. When the
        calibration carries noise the true camera position is recovered first. When an affine
        transform is present the mask additionally drops points that leave the transformed
        image bounds. The ego box walls closest to a camera inside the box are pulled behind
        the camera so the camera never sits inside the tested volume.

        Args:
            xyz: Point coordinates in the lidar frame.
            calibration_data: Camera and lidar calibration data, possibly carrying noise.
            affine_transform: Optional image space affine transform.
            image_shape: Height and width of the image.

        Returns:
            Bool[np.ndarray, " num_points"]: Mask of the points to keep.
        """
        lidar2cam = calibration_data.lidar_to_camera_transformation
        if calibration_data.noise is not None:
            noise_inv = np.linalg.inv(calibration_data.noise)
            lidar2cam_true = noise_inv @ lidar2cam
            rotation = lidar2cam_true[:3, :3]
            translation = lidar2cam_true[:3, 3]
        else:
            rotation = lidar2cam[:3, :3]
            translation = lidar2cam[:3, 3]

        camera_center_lidar = -rotation.T @ translation

        box_min = np.array(self.ego_box[:3])
        box_max = np.array(self.ego_box[3:])

        if np.all(camera_center_lidar >= box_min) and np.all(camera_center_lidar <= box_max):
            d_min = camera_center_lidar - box_min
            d_max = box_max - camera_center_lidar
            for i in range(2):
                if d_min[i] < d_max[i]:
                    box_min[i] = camera_center_lidar[i] + self.occlusion_adjust_margin
                else:
                    box_max[i] = camera_center_lidar[i] - self.occlusion_adjust_margin

        ray_origins = camera_center_lidar
        ray_directions = xyz - ray_origins

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (box_min - ray_origins) / ray_directions
            t2 = (box_max - ray_origins) / ray_directions

        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        t_enter = np.max(t_min, axis=1)
        t_exit = np.min(t_max, axis=1)

        hits_box = (t_enter <= t_exit) & (t_exit >= 0)
        occluded = hits_box & (t_enter < 0.999)

        if affine_transform is not None:
            point_cloud_ccs = self._transform_points_to_camera(xyz, calibration_data)
            valid_mask_3d = point_cloud_ccs[:, 2] > 0.0
            if not np.any(valid_mask_3d):
                return ~occluded

            point_cloud_ccs_valid = point_cloud_ccs[valid_mask_3d]
            occluded_valid = occluded[valid_mask_3d]

            point_cloud_ics = self._project_points_to_image(point_cloud_ccs_valid, calibration_data)
            point_cloud_ics_transformed = self._apply_affine_to_points(
                point_cloud_ics, affine_transform
            )

            height, width = image_shape
            in_bounds = (
                (point_cloud_ics_transformed[:, 0] >= 0)
                & (point_cloud_ics_transformed[:, 0] < width)
                & (point_cloud_ics_transformed[:, 1] >= 0)
                & (point_cloud_ics_transformed[:, 1] < height)
            )
            occluded_valid = occluded_valid | ~in_bounds

            full_occluded = np.zeros(xyz.shape[0], dtype=bool)
            full_occluded[valid_mask_3d] = occluded_valid
            full_occluded[~valid_mask_3d] = True
            return ~full_occluded

        return ~occluded

    def _transform_points_to_camera(
        self,
        points: Float32[np.ndarray, "num_points 3"],
        calibration_data: CalibrationData,
    ) -> Float32[np.ndarray, "num_points 3"]:
        """Transform lidar points to the camera coordinate system.

        Args:
            points: Point coordinates in the lidar frame.
            calibration_data: Camera and lidar calibration data.

        Returns:
            Point coordinates in the camera frame.
        """
        num_points = points.shape[0]
        points_hom = np.concatenate([points, np.ones((num_points, 1), dtype=points.dtype)], axis=1)

        lidar2cam = calibration_data.lidar_to_camera_transformation
        points_cam = (lidar2cam @ points_hom.T).T

        return points_cam[:, :3]

    def _project_points_to_image(
        self,
        point_cloud_ccs: Float32[np.ndarray, "num_points 3"],
        calibration_data: CalibrationData,
    ) -> Float32[np.ndarray, "num_points 2"]:
        """Project 3D points to 2D image coordinates.

        Args:
            point_cloud_ccs: Point coordinates in the camera frame.
            calibration_data: Camera and lidar calibration data.

        Returns:
            Projected image coordinates.
        """
        camera_matrix = calibration_data.new_camera_matrix
        distortion_coefficients = calibration_data.distortion_coefficients

        point_cloud_ics, _ = cv2.projectPoints(
            point_cloud_ccs,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            distortion_coefficients,
        )
        if point_cloud_ics is None:
            return np.zeros((0, 2), dtype=np.float32)

        return point_cloud_ics.reshape(-1, 2)

    def _apply_affine_to_points(
        self,
        points_2d: Float32[np.ndarray, "num_points 2"],
        affine_matrix: Float32[np.ndarray, "3 3"],
    ) -> Float32[np.ndarray, "num_points 2"]:
        """Apply an affine transformation to 2D points.

        Args:
            points_2d: The 2D points in image coordinates.
            affine_matrix: The affine transformation matrix.

        Returns:
            Transformed 2D points.
        """
        num_points = points_2d.shape[0]
        homogeneous = np.hstack([points_2d, np.ones((num_points, 1))])
        transformed = (affine_matrix @ homogeneous.T).T[:, :2]
        return transformed.astype(np.float32)

    def _create_lidar_images(
        self,
        image: Float32[np.ndarray, "height width channels"],
        point_cloud_ics: Float32[np.ndarray, "num_points 2"],
        point_cloud_ccs: Float32[np.ndarray, "num_points 3"],
        intensities: Float32[np.ndarray, " num_points"],
    ) -> Float32[np.ndarray, "height width fused_channels"]:
        """Create the fused image with depth and intensity channels.

        Args:
            image: Calibration image with values in [0, 255].
            point_cloud_ics: Projected image coordinates of the points.
            point_cloud_ccs: Point coordinates in the camera frame.
            intensities: Point intensities aligned with the coordinates.

        Returns:
            Fused five channel image normalized to [0, 1].
        """
        h, w = image.shape[:2]
        depth_image = np.zeros((h, w), dtype=np.float32)
        intensity_image = np.zeros((h, w), dtype=np.float32)

        valid_mask = (
            (point_cloud_ics[:, 0] >= 0)
            & (point_cloud_ics[:, 0] <= w - 1)
            & (point_cloud_ics[:, 1] >= 0)
            & (point_cloud_ics[:, 1] <= h - 1)
            & (point_cloud_ccs[:, 2] > 0.0)
            & (point_cloud_ccs[:, 2] < self.max_depth)
        )

        valid_ics = point_cloud_ics[valid_mask]
        valid_ccs = point_cloud_ccs[valid_mask]
        valid_intensities = intensities[valid_mask]

        if valid_ics.size > 0:
            y_offsets, x_offsets = np.mgrid[
                -self.dilation_size : self.dilation_size + 1,
                -self.dilation_size : self.dilation_size + 1,
            ]
            y_offsets = y_offsets.flatten()
            x_offsets = x_offsets.flatten()

            center_rows = valid_ics[:, 1].astype(np.int32)
            center_cols = valid_ics[:, 0].astype(np.int32)

            patch_rows = center_rows[:, np.newaxis] + y_offsets[np.newaxis, :]
            patch_cols = center_cols[:, np.newaxis] + x_offsets[np.newaxis, :]

            in_bounds_mask = (
                (patch_rows >= 0) & (patch_rows < h) & (patch_cols >= 0) & (patch_cols < w)
            )

            center_depths = 255 * valid_ccs[:, 2] / self.max_depth

            broadcasted_depths = np.broadcast_to(center_depths[:, np.newaxis], patch_rows.shape)
            broadcasted_intensities = np.broadcast_to(
                valid_intensities[:, np.newaxis], patch_rows.shape
            )

            final_rows = patch_rows[in_bounds_mask]
            final_cols = patch_cols[in_bounds_mask]
            final_depths = broadcasted_depths[in_bounds_mask]
            final_intensities = broadcasted_intensities[in_bounds_mask]

            sort_indices = np.argsort(final_depths)[::-1]
            sorted_rows = final_rows[sort_indices]
            sorted_cols = final_cols[sort_indices]
            sorted_depths = final_depths[sort_indices]
            sorted_intensities = final_intensities[sort_indices]

            depth_image[sorted_rows, sorted_cols] = sorted_depths
            intensity_image[sorted_rows, sorted_cols] = sorted_intensities

        depth_image = np.expand_dims(depth_image, axis=2)
        intensity_image = np.expand_dims(intensity_image, axis=2)

        fused = np.concatenate([image, depth_image, intensity_image], axis=2)
        return fused.astype(np.float32) / 255.0


class Affine(BaseTransform):
    """Affine transformation augmentation for the calibration image.

    Applies a controlled affine distortion to the image and stores the affine matrix on the
    calibration state so the fusion can apply it to the projected lidar points. A zoom keeps
    the transformed image covering the entire viewport without black borders. When the
    transform is skipped an identity affine matrix is stored.
    """

    _required_fields = ["calibration"]

    def __init__(self, *, p: float = 0.5, max_distortion: float = 0.1) -> None:
        """Initialize the Affine transform.

        Args:
            p: Probability of applying the augmentation.
            max_distortion: Maximum corner displacement as a fraction of the image size.
        """
        self.p = p
        self.max_distortion = max_distortion

    def on_skip(self, sample: Sample) -> Sample:
        """Store the identity affine matrix when the augmentation is skipped.

        Args:
            sample: Sample with a seeded calibration state.

        Returns:
            Sample with the identity affine transform.
        """
        calibration = sample.calibration.model_copy(
            update={"affine_transform": np.eye(3, dtype=np.float32)}
        )
        return sample.replace(calibration=calibration)

    def transform(self, sample: Sample) -> Sample:
        """Apply a random affine transformation to the calibration image.

        Args:
            sample: Sample with a loaded calibration image.

        Returns:
            Sample with the transformed image and the affine matrix.
        """
        calibration = sample.calibration
        if calibration.image is None:
            raise ValueError("Affine requires a loaded calibration image.")
        image = calibration.image.astype(np.uint8)

        h, w = image.shape[:2]

        max_offset_x = self.max_distortion * w
        max_offset_y = self.max_distortion * h

        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst_pts = src_pts + np.random.uniform(
            low=-np.array([[max_offset_x, max_offset_y]] * 3),
            high=np.array([[max_offset_x, max_offset_y]] * 3),
        ).astype(np.float32)

        affine_matrix_2x3 = cv2.getAffineTransform(src_pts, dst_pts)

        # Map the destination corners back to source space to size the required zoom
        inv_affine = cv2.invertAffineTransform(affine_matrix_2x3)

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners_hom = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
        src_corners = (inv_affine @ corners_hom.T).T

        cx, cy = w / 2.0, h / 2.0
        max_x = np.max(np.abs(src_corners[:, 0] - cx))
        max_y = np.max(np.abs(src_corners[:, 1] - cy))
        scale = max(1.0, max_x / cx, max_y / cy)

        zoom_mat = np.array(
            [[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)], [0, 0, 1]],
            dtype=np.float64,
        )

        affine_matrix_3x3 = np.eye(3, dtype=np.float64)
        affine_matrix_3x3[:2, :3] = affine_matrix_2x3
        affine_matrix_3x3 = affine_matrix_3x3 @ zoom_mat
        affine_matrix_2x3 = affine_matrix_3x3[:2]

        image = cv2.warpAffine(image, affine_matrix_2x3, (w, h), borderMode=cv2.BORDER_CONSTANT)

        calibration = calibration.model_copy(
            update={
                "image": image.astype(np.float32),
                "affine_transform": affine_matrix_3x3.astype(np.float32),
            }
        )
        return sample.replace(calibration=calibration)


class SaveFusionPreview(BaseTransform):
    """Save preview images of the fused camera and lidar data for visualization.

    Creates two overlay images per sample, the color image with the depth points and the color
    image with the intensity points, each colorized with a configurable colormap. The transform
    passes the sample through unchanged.
    """

    _required_fields = ["calibration"]

    def __init__(
        self,
        *,
        p: float = 1.0,
        out_dir: str = "",
        max_depth: float = 128.0,
        alpha: float = 0.5,
        depth_colormap: str = "turbo",
        intensity_colormap: str = "jet",
    ) -> None:
        """Initialize the SaveFusionPreview transform.

        Args:
            p: Probability of saving the preview images of a sample.
            out_dir: Output directory of the preview images.
            max_depth: Maximum depth used during the fusion, needed to recover depth values.
            alpha: Blending factor of the overlay, 0.0 keeps the color image and 1.0 keeps
                only the overlay points.
            depth_colormap: Matplotlib colormap name of the depth visualization.
            intensity_colormap: Matplotlib colormap name of the intensity visualization.
        """
        self.p = p
        self.out_dir = Path(out_dir)
        self.max_depth = max_depth
        self.alpha = alpha
        self.depth_cmap = matplotlib.colormaps[depth_colormap]
        self.intensity_cmap = matplotlib.colormaps[intensity_colormap]

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def transform(self, sample: Sample) -> Sample:
        """Save the preview images of the sample.

        Args:
            sample: Sample with a fused calibration image.

        Returns:
            The unchanged sample.
        """
        calibration = sample.calibration
        if calibration.fused_image is None:
            raise ValueError("SaveFusionPreview requires a fused calibration image.")
        camera_frame = camera_frame_by_channel(sample, calibration.camera_name)
        base_name = Path(camera_frame.camera_image_path).stem
        self._save_preview(calibration.fused_image, base_name, calibration.status)
        return sample

    def _save_preview(
        self,
        fused_image: Float32[np.ndarray, "height width fused_channels"],
        base_name: str,
        calibration_status: CalibrationStatus | None,
    ) -> None:
        """Save the depth and the intensity preview image of one sample.

        Args:
            fused_image: Fused five channel image normalized to [0, 1].
            base_name: Base filename of the preview images.
            calibration_status: Ground truth calibration status, or None when undecided.
        """
        color, depth, intensity = self._recover_channels(fused_image)

        if calibration_status is None:
            status_suffix = ""
        else:
            status_suffix = (
                "_calibrated"
                if calibration_status == CalibrationStatus.CALIBRATED
                else "_miscalibrated"
            )

        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        depth_overlay = self._create_overlay(rgb, depth, self.depth_cmap, self.alpha)
        intensity_overlay = self._create_overlay(rgb, intensity, self.intensity_cmap, self.alpha)

        depth_path = self.out_dir / f"{base_name}{status_suffix}_depth.png"
        intensity_path = self.out_dir / f"{base_name}{status_suffix}_intensity.png"

        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(intensity_path), cv2.cvtColor(intensity_overlay, cv2.COLOR_RGB2BGR))

    def _recover_channels(
        self, fused_image: Float32[np.ndarray, "height width fused_channels"]
    ) -> tuple[
        UInt8[np.ndarray, "height width 3"],
        Float32[np.ndarray, "height width"],
        Float32[np.ndarray, "height width"],
    ]:
        """Recover the color, depth, and intensity values from the fused image.

        Args:
            fused_image: Fused five channel image normalized to [0, 1].

        Returns:
            Tuple of the color image, the depth values, and the intensity values.
        """
        color = (fused_image[:, :, :3] * 255).astype(np.uint8)
        depth = fused_image[:, :, 3] * self.max_depth
        intensity = fused_image[:, :, 4] * 255
        return color, depth, intensity

    def _create_overlay(
        self,
        rgb: UInt8[np.ndarray, "height width 3"],
        values: Float32[np.ndarray, "height width"],
        cmap: Colormap,
        alpha: float,
    ) -> UInt8[np.ndarray, "height width 3"]:
        """Create an alpha blended overlay of the image with colorized point values.

        Args:
            rgb: Color image in height, width, channels layout.
            values: Value array colorized onto the image.
            cmap: Matplotlib colormap object.
            alpha: Blending factor of the overlay points.

        Returns:
            Blended image in height, width, channels layout.
        """
        mask = values > 0

        max_val = values.max() if values.max() > 0 else 1.0
        normalized = values / max_val

        colored = cmap(normalized)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        result = rgb.copy()
        result[mask] = (
            (1 - alpha) * rgb[mask].astype(np.float32) + alpha * colored[mask].astype(np.float32)
        ).astype(np.uint8)

        return result


class ImageAug3D(BaseTransform):
    """Resize, crop, flip, and rotate the multiview images and track the projection updates.

    Every view samples its own image space transform. The camera intrinsics absorb the
    transform, the lidar to image matrices are recomputed, and the per view augmentation
    matrices are exposed on the image set. The intrinsics before the first image space
    transform are preserved as the original intrinsics.
    """

    _required_fields = ["images"]

    def __init__(
        self,
        *,
        final_dim: Sequence[int],
        resize_lim: Sequence[float],
        bot_pct_lim: Sequence[float],
        rand_flip: bool = False,
        rot_lim: Sequence[float] | None = None,
        training: bool = True,
    ) -> None:
        """Initialize the ImageAug3D transform.

        Args:
            final_dim: Final image size as [height, width].
            resize_lim: Minimum and maximum resize factors.
            bot_pct_lim: Minimum and maximum bottom crop ratios.
            rand_flip: Whether horizontal flipping is enabled.
            rot_lim: Optional in plane rotation range in degrees.
            training: Whether to sample stochastic augmentation parameters.
        """
        self.final_dim = tuple(final_dim)
        self.resize_lim = tuple(resize_lim)
        self.bot_pct_lim = tuple(bot_pct_lim)
        if len(self.resize_lim) != 2 or len(self.bot_pct_lim) != 2:
            raise ValueError(
                f"resize_lim and bot_pct_lim must contain [min, max], got {resize_lim} and "
                f"{bot_pct_lim}."
            )
        self.rand_flip = rand_flip
        self.rot_lim = tuple(rot_lim) if rot_lim is not None else (0.0, 0.0)
        self.training = training

    def transform(self, sample: Sample) -> Sample:
        """Augment the multiview images and update the projection matrices.

        Args:
            sample: Sample with a loaded image set.

        Returns:
            Sample with the augmented image set.
        """
        images = sample.images
        augmented = []
        aug_matrices = []
        intrinsics = images.camera_intrinsics.copy()
        for view_index in range(len(images)):
            image_hwc = np.transpose(images.images[view_index], (1, 2, 0))
            aug_matrix, augmented_image = self._augment_image(image_hwc)
            augmented.append(np.transpose(augmented_image, (2, 0, 1)))
            aug_matrices.append(aug_matrix)
            intrinsics[view_index] = aug_matrix @ intrinsics[view_index]

        ori_camera_intrinsics = images.ori_camera_intrinsics
        if ori_camera_intrinsics is None:
            ori_camera_intrinsics = images.camera_intrinsics

        updated = images.model_copy(
            update={
                "images": np.stack(augmented, axis=0),
                "camera_intrinsics": intrinsics,
                "lidar2img": intrinsics @ images.lidar2cam,
                "ori_camera_intrinsics": ori_camera_intrinsics,
                "img_aug_matrix": np.stack(aug_matrices, axis=0).astype(np.float32),
            }
        )
        return sample.replace(images=updated)

    def _augment_image(
        self, image: Float32[np.ndarray, "height width channels"]
    ) -> tuple[Float32[np.ndarray, "4 4"], Float32[np.ndarray, "out_height out_width channels"]]:
        """Augment one view and build its 4x4 image space transform.

        Args:
            image: Image in height, width, channels layout.

        Returns:
            Tuple of the 4x4 augmentation matrix and the augmented image.
        """
        source_height, source_width = image.shape[:2]
        final_height, final_width = self.final_dim

        if self.training:
            resize = np.random.uniform(*self.resize_lim)
            crop_bottom = np.random.uniform(*self.bot_pct_lim)
            crop_height = int((1 - crop_bottom) * final_height)
            crop_width = final_width
            flip = bool(self.rand_flip and np.random.randint(2))
            rotate = float(np.random.uniform(*self.rot_lim))
        else:
            resize = float(np.mean(self.resize_lim))
            crop_bottom = float(np.mean(self.bot_pct_lim))
            crop_height = int((1 - crop_bottom) * final_height)
            crop_width = final_width
            flip = False
            rotate = 0.0

        resized_width = int(source_width * resize)
        resized_height = int(source_height * resize)
        resized = cv2.resize(image, (resized_width, resized_height))

        crop_y = max(0, resized_height - crop_height)
        crop_x = max(0, (resized_width - crop_width) // 2)
        cropped = resized[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        cropped = cv2.resize(cropped, (final_width, final_height))

        transform = np.eye(4, dtype=np.float32)
        transform[0, 0] = resize * final_width / crop_width
        transform[1, 1] = resize * final_height / crop_height
        transform[0, 2] = -crop_x * final_width / crop_width
        transform[1, 2] = -crop_y * final_height / crop_height

        if flip:
            cropped = np.ascontiguousarray(np.fliplr(cropped))
            flip_mat = np.array(
                [
                    [-1.0, 0.0, final_width - 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            transform = flip_mat @ transform

        if abs(rotate) > 1e-6:
            center = (final_width / 2.0, final_height / 2.0)
            affine = cv2.getRotationMatrix2D(center, rotate, 1.0).astype(np.float32)
            cropped = cv2.warpAffine(cropped, affine, (final_width, final_height))
            rot_mat = np.eye(4, dtype=np.float32)
            rot_mat[:2, :3] = affine
            transform = rot_mat @ transform

        return transform, cropped.astype(image.dtype)


__all__ = [
    "Affine",
    "CalibrationMisalignment",
    "ImageAug3D",
    "LidarCameraFusion",
    "SaveFusionPreview",
]
