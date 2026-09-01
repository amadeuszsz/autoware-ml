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

"""Point cloud sampling and subsampling transforms."""

from __future__ import annotations

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class PointShuffle(BaseTransform):
    """Randomly permute the points of the sample.

    The segmentation labels are reordered with the same permutation, so both stay aligned.
    Shuffling breaks the leading current frame block, so the point cloud stops tracking
    num_current_points.
    """

    _required_fields = ["points"]

    def __init__(self, *, p: float | None = None) -> None:
        """Initialize the PointShuffle transform.

        Args:
            p: Probability of applying the transform. None means always apply.
        """
        self.p = p

    def transform(self, sample: Sample) -> Sample:
        """Shuffle the points with one shared permutation.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the shuffled points and aligned segmentation labels.
        """
        permutation = np.random.permutation(len(sample.points))
        return sample.reorder_points(permutation)


class RandomDropout(BaseTransform):
    """Randomly remove a fraction of the points."""

    _required_fields = ["points"]

    def __init__(self, *, p: float = 0.5, dropout_ratio: float = 0.2) -> None:
        """Initialize the RandomDropout transform.

        Args:
            p: Probability of applying the transform.
            dropout_ratio: Fraction of points removed when dropout is applied.
        """
        self.p = p
        self.dropout_ratio = dropout_ratio

    def transform(self, sample: Sample) -> Sample:
        """Randomly drop a subset of the points.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the surviving points and aligned segmentation labels.
        """
        point_count = len(sample.points)
        keep_count = max(1, int(point_count * (1 - self.dropout_ratio)))
        keep_indices = np.sort(np.random.choice(point_count, keep_count, replace=False))
        mask = np.zeros(point_count, dtype=bool)
        mask[keep_indices] = True
        return sample.filter_points(mask)
