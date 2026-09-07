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

"""Camera and lidar fusion geometric augmentations.

Identical scene math to the lidar only variants in transforms.point_cloud.geometry, shared via
transforms.geometry3d, plus the camera matrix update so the projection stays consistent after
the transform. The transforms require both a loaded point cloud and a loaded image set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms import geometry3d as g3d
from autoware_ml.transforms.base import BaseTransform


class RandomFlip3D(BaseTransform):
    """Random BEV flip of the point cloud, the boxes, and the camera matrices."""

    _required_fields = ["points", "images"]

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
        """Apply BEV flips to the point cloud, the boxes, and the camera matrices.

        Args:
            sample: Sample with a loaded point cloud and a loaded image set.

        Returns:
            Sample with the flipped scene and consistent projection matrices.
        """
        flip_x, flip_y = g3d.sample_bev_flips(
            self.flip_ratio_bev_horizontal, self.flip_ratio_bev_vertical
        )
        if flip_y:
            sample = g3d.flip_sample(sample, axis=1)
        if flip_x:
            sample = g3d.flip_sample(sample, axis=0)
        flip = g3d.flip_matrix(flip_x, flip_y)
        images = g3d.update_image_set_matrices(sample.images, np.linalg.inv(flip))
        return sample.replace(images=images)


class GlobalRotScaleTrans(BaseTransform):
    """Global rotation, scaling, and translation of the scene and the camera matrices."""

    _required_fields = ["points", "images"]

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
        """Rotate, scale, and translate the scene and update the camera matrices.

        Args:
            sample: Sample with a loaded point cloud and a loaded image set.

        Returns:
            Sample with the transformed scene and consistent projection matrices.
        """
        rotation, _, scale, translation = g3d.sample_rot_scale_trans(
            self.rot_range, self.scale_ratio_range, self.translation_std
        )
        sample = g3d.transform_sample_points(sample, rotation, scale, translation)
        augmentation = g3d.rot_scale_trans_matrix(rotation, scale, translation)
        images = g3d.update_image_set_matrices(sample.images, np.linalg.inv(augmentation))
        return sample.replace(images=images)
