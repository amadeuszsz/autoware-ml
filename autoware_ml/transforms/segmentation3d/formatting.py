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

"""Formatting transforms for point cloud segmentation pipelines."""

from __future__ import annotations

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.time_lag import current_frame_mask


class PreparePointSegInput(BaseTransform):
    """Pad the current frame segmentation labels to the full densified point cloud.

    Labels exist only for the current frame, which the point loader places as the leading
    block of the cloud. Points appended from earlier sweeps receive ignore_index so they
    contribute to the geometry but never to the loss or the metrics. A training or evaluation
    pipeline requires loaded labels, a prediction pipeline declares that it runs without them
    and every current frame point then receives ignore_index. The transform reads the leading
    block size from the point cloud and verifies it against the timestamp_difference feature
    when the cloud carries one.
    """

    _required_fields = ["points"]

    def __init__(self, *, ignore_index: int, require_labels: bool) -> None:
        """Initialize the PreparePointSegInput transform.

        Args:
            ignore_index: Label assigned to points without supervision.
            require_labels: Whether the sample must carry loaded segmentation labels. False only
                for prediction pipelines, which run without ground truth.
        """
        self.ignore_index = int(ignore_index)
        self.require_labels = require_labels

    def transform(self, sample: Sample) -> Sample:
        """Build the full length segmentation labels of the point cloud.

        Args:
            sample: Sample with a loaded point cloud and optional current frame labels.

        Returns:
            Sample with segmentation labels covering every point.
        """
        points = sample.points
        num_current = points.num_current_points
        if num_current is None:
            raise ValueError(
                "PreparePointSegInput requires num_current_points, the point cloud does not "
                "track its leading current frame block anymore."
            )
        num_points = len(points)
        if sample.segment is None and self.require_labels:
            raise ValueError(
                "PreparePointSegInput requires loaded segmentation labels, run "
                "LoadSeg3DAnnotations before it or declare a prediction pipeline with "
                "require_labels false."
            )
        if sample.segment is None:
            mask = np.full(num_current, self.ignore_index, dtype=np.int64)
        else:
            mask = sample.segment.labels
        if mask.shape[0] != num_current:
            raise ValueError(
                "PreparePointSegInput requires one semantic label per current-frame point: "
                f"got {mask.shape[0]} labels for {num_current} points."
            )
        current_mask = current_frame_mask(points)
        if current_mask is None:
            if num_current != num_points:
                raise ValueError(
                    "PreparePointSegInput found no timestamp_difference feature, so every "
                    f"point must belong to the current frame: got {num_current} current "
                    f"points for {num_points} points."
                )
        else:
            if np.any(~current_mask[:num_current]) or np.any(current_mask[num_current:]):
                raise ValueError(
                    "PreparePointSegInput requires the current frame (time lag 0) to be "
                    f"exactly the leading block of {num_current} points."
                )
        segment = np.full(num_points, self.ignore_index, dtype=np.int64)
        segment[:num_current] = mask
        return sample.model_copy(update={"segment": SegmentationLabels(labels=segment)})
