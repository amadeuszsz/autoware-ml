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

"""Formatting transforms shared across point cloud pipelines."""

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.geometry import PointFeatureName


class PreparePointCloudInput(BaseTransform):
    """Normalize the intensity feature and validate the point cloud layout.

    The transform divides the intensity column by 255 so the network consumes intensities in
    [0, 1], and it validates the declared time lag expectation so a pipeline never mixes
    densified and single frame clouds silently.
    """

    _required_fields = ["points"]

    def __init__(self, *, require_time_lag: bool) -> None:
        """Initialize the PreparePointCloudInput transform.

        Args:
            require_time_lag: Whether the point cloud must carry the timestamp_difference
                feature. Set to False for single frame pipelines, where the feature must be
                absent.
        """
        self.require_time_lag = require_time_lag

    def transform(self, sample: Sample) -> Sample:
        """Normalize the intensity feature of the point cloud.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the normalized point cloud.
        """
        points = sample.points
        has_time_lag = points.has_feature(PointFeatureName.TIMESTAMP_DIFFERENCE)
        if self.require_time_lag and not has_time_lag:
            raise ValueError(
                "PreparePointCloudInput requires the timestamp_difference feature but the "
                f"point cloud carries {points.feature_names}."
            )
        if not self.require_time_lag and has_time_lag:
            raise ValueError(
                "PreparePointCloudInput was configured without a time lag but the point cloud "
                f"carries {points.feature_names}."
            )
        intensity = points.feature(PointFeatureName.INTENSITY)
        return sample.replace(
            points=points.with_feature(PointFeatureName.INTENSITY, intensity / 255.0)
        )
