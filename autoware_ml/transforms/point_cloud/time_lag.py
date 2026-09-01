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


"""Per point time lag of a densified point cloud.

Point loaders place the current frame first and stamp every point with the seconds elapsed since
it was captured, 0 for the current frame and positive for points appended from earlier sweeps.
Any transform whose decision must concern the current frame alone selects its points with the
mask built here. The point cloud is self describing, the timestamp_difference feature carries
the lag.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.types.geometry import PointFeatureName


def current_frame_mask(point_cloud: PointCloud) -> Bool[np.ndarray, " num_points"] | None:
    """Return the mask selecting the points captured in the current frame.

    Args:
        point_cloud: Point cloud of the sample.

    Returns:
        Boolean mask of the current frame points, or None when the point cloud carries no
        timestamp_difference feature, every point then belongs to the current frame and no
        selection is needed.
    """
    if not point_cloud.has_feature(PointFeatureName.TIMESTAMP_DIFFERENCE):
        return None
    return point_cloud.feature(PointFeatureName.TIMESTAMP_DIFFERENCE) == 0
