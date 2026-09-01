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

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Int64
from pydantic import BaseModel, ConfigDict, model_validator


class SegmentationLabels(BaseModel):
    """
    Semantic segmentation labels of one sample, aligned row by row with the point cloud of the
    sample. Points without supervision carry the ignore label of the task configuration.

    Attributes:
      labels: Semantic label of every point.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    labels: Int64[np.ndarray, " num_points"]

    @model_validator(mode="after")
    def validate_labels(self) -> SegmentationLabels:
        """
        Validate the label array.

        Returns:
          SegmentationLabels: The validated labels.
        """

        if self.labels.ndim != 1:
            raise ValueError(
                f"Segmentation labels must be 1D, got shape {self.labels.shape}."
            )
        if self.labels.dtype != np.int64:
            raise ValueError(
                f"Segmentation labels must be int64, got {self.labels.dtype}."
            )
        return self

    def __len__(self) -> int:
        """
        Get the number of labeled points.

        Returns:
          int: Number of labeled points.
        """

        return self.labels.shape[0]

    def filter(self, mask: Bool[np.ndarray, " num_points"]) -> SegmentationLabels:
        """
        Create labels keeping only the masked rows.

        Args:
          mask: Boolean mask of the rows to keep.

        Returns:
          SegmentationLabels: Filtered labels.
        """

        if mask.dtype != np.bool_ or mask.shape != (len(self),):
            raise ValueError(
                f"Filter mask must be a boolean array of shape ({len(self)},), "
                f"got {mask.dtype} with shape {mask.shape}."
            )
        return self.model_copy(update={"labels": self.labels[mask]})

    def reorder(self, indices: Int64[np.ndarray, " num_points"]) -> SegmentationLabels:
        """
        Create labels with reordered rows.

        Args:
          indices: Permutation of the row indices.

        Returns:
          SegmentationLabels: Reordered labels.
        """

        if indices.shape != (len(self),):
            raise ValueError(
                f"Reorder indices must have shape ({len(self)},), got {indices.shape}."
            )
        return self.model_copy(update={"labels": self.labels[indices]})
