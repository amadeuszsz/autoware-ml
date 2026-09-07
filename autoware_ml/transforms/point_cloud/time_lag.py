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


"""Current frame selection of a densified point cloud.

Point loaders place the current frame first and record its size as num_current_points, which
every row preserving transform keeps up to date. Any transform whose decision must concern the
current frame alone selects its points with the mask built here, whatever features the cloud
carries.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool

from autoware_ml.datamodule.samples.point_cloud import PointCloud


def current_frame_mask(point_cloud: PointCloud) -> Bool[np.ndarray, " num_points"]:
    """Return the mask selecting the points captured in the current frame.

    Args:
        point_cloud: Point cloud of the sample.

    Returns:
        Boolean mask of the leading current frame block.

    Raises:
        ValueError: If the point cloud no longer tracks its current frame block.
    """
    num_current_points = point_cloud.num_current_points
    if num_current_points is None:
        raise ValueError(
            "The point cloud does not track its current frame block anymore, a transform "
            "reordered its rows before the current frame was selected."
        )
    mask = np.zeros(len(point_cloud), dtype=bool)
    mask[:num_current_points] = True
    return mask
