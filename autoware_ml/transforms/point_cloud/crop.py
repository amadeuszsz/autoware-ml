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

"""Point cloud cropping transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class PointsRangeFilter(BaseTransform):
    """Drop points outside a configured spatial range.

    The lower bound is inclusive and the upper bound is exclusive, so voxel indices computed
    from the surviving points always stay inside the grid.
    """

    _required_fields = ["points"]

    def __init__(self, *, point_cloud_range: Sequence[float]) -> None:
        """Initialize the PointsRangeFilter transform.

        Args:
            point_cloud_range: Bounds [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must have 6 elements, got {len(point_cloud_range)}."
            )
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def transform(self, sample: Sample) -> Sample:
        """Filter the points to the configured spatial range.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the filtered points and aligned segmentation labels.
        """
        coord = sample.points.coord
        lower = self.point_cloud_range[:3]
        upper = self.point_cloud_range[3:]
        mask = ((coord >= lower) & (coord < upper)).all(axis=1)
        return sample.filter_points(mask)


class CropBoxInner(BaseTransform):
    """Remove points inside one axis aligned 3D box."""

    _required_fields = ["points"]

    def __init__(self, *, crop_box: Sequence[float]) -> None:
        """Initialize the CropBoxInner transform.

        Args:
            crop_box: Box bounds [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        if len(crop_box) != 6:
            raise ValueError(f"crop_box must have 6 elements, got {len(crop_box)}.")
        self.crop_box = np.asarray(crop_box, dtype=np.float32)

    def transform(self, sample: Sample) -> Sample:
        """Keep only the points outside the configured box.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the filtered points and aligned segmentation labels.
        """
        coord = sample.points.coord
        x_min, y_min, z_min, x_max, y_max, z_max = self.crop_box
        mask = (
            (coord[:, 0] < x_min)
            | (coord[:, 0] > x_max)
            | (coord[:, 1] < y_min)
            | (coord[:, 1] > y_max)
            | (coord[:, 2] < z_min)
            | (coord[:, 2] > z_max)
        )
        return sample.filter_points(mask)
