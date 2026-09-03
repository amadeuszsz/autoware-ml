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

"""Feature construction transforms shared across tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.loading import coerce_feature_names
from autoware_ml.types.geometry import PointFeatureName


class BuildPointFeatures(BaseTransform):
    """Reduce the point cloud to the configured feature columns in the configured order.

    The transform packs the selected feature columns of the point cloud so the model consumes
    exactly the declared layout. Every selected feature must exist, the row order stays
    untouched, and the current frame block is preserved.
    """

    _required_fields = ["points"]

    def __init__(self, *, feature_names: Sequence[str | PointFeatureName]) -> None:
        """Initialize the BuildPointFeatures transform.

        Args:
            feature_names: Feature columns of the output point cloud in the requested order,
                starting with x, y, and z.
        """
        self.feature_names = coerce_feature_names(feature_names)

    def transform(self, sample: Sample) -> Sample:
        """Pack the configured feature columns into a new point cloud.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the packed point cloud.
        """
        points = sample.points
        point_cloud = PointCloud(
            features=np.ascontiguousarray(points.pack(self.feature_names), dtype=np.float32),
            feature_names=self.feature_names,
            num_current_points=points.num_current_points,
        )
        return sample.model_copy(update={"points": point_cloud})
