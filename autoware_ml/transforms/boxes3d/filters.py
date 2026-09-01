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

"""3D bounding-box filter transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Int64

from autoware_ml.datamodule.samples.boxes3d import Boxes3D
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.geometry.utils import points_in_boxes_3d
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.time_lag import current_frame_mask


def _count_current_frame_points(
    points: PointCloud, boxes: Boxes3D
) -> Int64[np.ndarray, " num_boxes"]:
    """Count the current frame points inside every box.

    A ground truth box is annotated on the current frame, so its point support is a property
    of that frame: points appended from earlier sweeps must not decide whether the box is
    kept, or the surviving set of boxes would depend on how many sweeps the model consumes.

    Args:
        points: Point cloud of the sample.
        boxes: Boxes whose interior points are counted.

    Returns:
        Number of current frame points inside every box.
    """
    mask = current_frame_mask(points)
    coord = points.coord if mask is None else points.coord[mask]
    inside = points_in_boxes_3d(
        torch.from_numpy(np.ascontiguousarray(coord, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(boxes.params, dtype=np.float32)),
    )
    return inside.sum(dim=1).numpy().astype(np.int64)


class ObjectRangeFilter(BaseTransform):
    """Remove boxes whose center lies outside the configured point cloud range."""

    _required_fields = ["boxes"]

    def __init__(self, *, point_cloud_range: Sequence[float]) -> None:
        """Initialize the ObjectRangeFilter transform.

        Args:
            point_cloud_range: Range bounds [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        if len(point_cloud_range) != 6:
            raise ValueError(
                f"point_cloud_range must contain 6 bounds, got {list(point_cloud_range)}."
            )
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def transform(self, sample: Sample) -> Sample:
        """Filter boxes whose centers fall outside the configured range.

        Args:
            sample: Sample with loaded detection boxes.

        Returns:
            Sample with the out-of-range boxes removed.
        """
        centers = sample.boxes.params[:, :3]
        lower = self.point_cloud_range[:3]
        upper = self.point_cloud_range[3:]
        mask = ((centers >= lower) & (centers <= upper)).all(axis=1)
        return sample.model_copy(update={"boxes": sample.boxes.filter(mask)})


class ObjectRangeMinPointsFilter(BaseTransform):
    """Remove boxes below a point count threshold within a BEV radial interval.

    The point counts are recomputed from the current frame points of the sample, so the
    surviving boxes do not depend on the number of loaded sweeps.
    """

    _required_fields = ["points", "boxes"]

    def __init__(self, *, range_radius: Sequence[float], min_num_points: int) -> None:
        """Initialize the ObjectRangeMinPointsFilter transform.

        Args:
            range_radius: Radial interval [min_radius, max_radius] in meters.
            min_num_points: Minimum points required for boxes inside the interval.
        """
        if len(range_radius) != 2:
            raise ValueError(f"range_radius must contain [min, max], got {range_radius}")
        min_radius, max_radius = (float(value) for value in range_radius)
        if min_radius < 0.0 or min_radius >= max_radius:
            raise ValueError(f"Expected 0 <= min radius < max radius, got {range_radius}")
        if min_num_points <= 0:
            raise ValueError(f"min_num_points must be positive, got {min_num_points}")
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_num_points = min_num_points

    def transform(self, sample: Sample) -> Sample:
        """Filter boxes in the configured radial band by current frame point count.

        Args:
            sample: Sample with a loaded point cloud and detection boxes.

        Returns:
            Sample with the low-support in-range boxes removed.
        """
        radii = np.linalg.norm(sample.boxes.params[:, :2], axis=1)
        in_range = (radii >= self.min_radius) & (radii < self.max_radius)
        counts = _count_current_frame_points(sample.points, sample.boxes)
        mask = ~in_range | (counts >= self.min_num_points)
        return sample.model_copy(update={"boxes": sample.boxes.filter(mask)})
