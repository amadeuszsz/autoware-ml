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

"""Point cloud geometric augmentations for lidar only pipelines.

The transforms operate on the point cloud and the boxes of a sample and never touch camera
matrices. The camera aware variants live in transforms.camera_lidar.geometry and
transforms.camera.geometry and share the exact same math via transforms.geometry3d.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms import geometry3d as g3d
from autoware_ml.transforms.base import BaseTransform


class RandomRotateTargetAngle(BaseTransform):
    """Rotate the point cloud and the boxes by one sampled discrete target angle."""

    _required_fields = ["points"]

    def __init__(
        self,
        *,
        p: float = 0.5,
        angle: Sequence[float],
        axis: str = "z",
        center: Sequence[float] | None = None,
    ) -> None:
        """Initialize the RandomRotateTargetAngle transform.

        Args:
            p: Probability of applying the transform.
            angle: Candidate rotation angles in multiples of pi radians.
            axis: Rotation axis. Only z is supported for box aware use.
            center: Optional rotation center.
        """
        self.p = p
        self.angle = list(angle)
        self.axis = axis
        self.center = np.asarray(center, dtype=np.float32) if center is not None else None

    def transform(self, sample: Sample) -> Sample:
        """Rotate the point coordinates and the boxes by one selected target angle.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the rotated point cloud and boxes.
        """
        if sample.boxes is not None and self.axis != "z":
            raise ValueError(
                "RandomRotateTargetAngle with boxes requires axis='z'; a rotation "
                f"around '{self.axis}' cannot be expressed as a yaw update."
            )
        angle = float(np.random.choice(self.angle)) * np.pi
        rotation = g3d.rotation_matrix(self.axis, angle)
        center = g3d.resolve_rotation_center(sample.points.coord, self.center)
        return g3d.rotate_sample_about_center(sample, rotation, angle, center)


class RandomFlip3D(BaseTransform):
    """Randomly flip the point cloud and the boxes across the BEV axes."""

    _required_fields = ["points"]

    def __init__(
        self,
        *,
        flip_ratio_bev_horizontal: float = 0.5,
        flip_ratio_bev_vertical: float = 0.5,
    ) -> None:
        """Initialize the RandomFlip3D transform.

        Args:
            flip_ratio_bev_horizontal: Probability of flipping the lateral y axis.
            flip_ratio_bev_vertical: Probability of flipping the longitudinal x axis.
        """
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def transform(self, sample: Sample) -> Sample:
        """Apply BEV flips to the point cloud and the boxes.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the flipped point cloud and boxes.
        """
        flip_x, flip_y = g3d.sample_bev_flips(
            self.flip_ratio_bev_horizontal, self.flip_ratio_bev_vertical
        )
        if flip_y:
            sample = g3d.flip_sample(sample, axis=1)
        if flip_x:
            sample = g3d.flip_sample(sample, axis=0)
        return sample


class GlobalRotScaleTrans(BaseTransform):
    """Apply a global rotation, scaling, and optional translation to the point cloud."""

    _required_fields = ["points"]

    def __init__(
        self,
        *,
        rot_range: Sequence[float],
        scale_ratio_range: Sequence[float],
        translation_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the GlobalRotScaleTrans transform.

        Args:
            rot_range: Min and max rotation angles in radians around z.
            scale_ratio_range: Min and max scale factors.
            translation_std: Optional per axis Gaussian translation std as [x, y, z].
        """
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            np.asarray(translation_std, dtype=np.float32) if translation_std is not None else None
        )

    def transform(self, sample: Sample) -> Sample:
        """Rotate, scale, and translate the point cloud and the boxes.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the transformed point cloud and boxes.
        """
        rotation, _, scale, translation = g3d.sample_rot_scale_trans(
            self.rot_range, self.scale_ratio_range, self.translation_std
        )
        return g3d.transform_sample_points(sample, rotation, scale, translation)
