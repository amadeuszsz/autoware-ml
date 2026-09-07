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
from jaxtyping import Bool, Float32, Int64
from pydantic import BaseModel, ConfigDict, model_validator

# Number of box parameters following Box3DFieldIndex without the vertical velocity:
# x, y, z, length, width, height, yaw, velocity_x, velocity_y
NUM_BOX_PARAMS = 9


class Boxes3D(BaseModel):
    """
    Detection ground truth boxes of one sample in the lidar frame. The box center sits at the
    gravity center and the parameters follow Box3DFieldIndex without the vertical velocity.

    Attributes:
      params: Box parameters with one row per box.
      labels: Label index of every box. Ignored boxes carry a negative label.
      names: Label name of every box.
      num_lidar_points: Number of lidar points inside every box.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    params: Float32[np.ndarray, "num_boxes num_box_params"]
    labels: Int64[np.ndarray, " num_boxes"]
    names: tuple[str, ...]
    num_lidar_points: Int64[np.ndarray, " num_boxes"]

    @model_validator(mode="after")
    def validate_boxes(self) -> Boxes3D:
        """
        Validate the box arrays against each other.

        Returns:
          Boxes3D: The validated boxes.
        """

        if self.params.ndim != 2 or self.params.shape[1] != NUM_BOX_PARAMS:
            raise ValueError(
                f"Box parameters must have shape (num_boxes, {NUM_BOX_PARAMS}), "
                f"got {self.params.shape}."
            )
        if self.params.dtype != np.float32:
            raise ValueError(f"Box parameters must be float32, got {self.params.dtype}.")
        if self.labels.dtype != np.int64 or self.labels.shape != (len(self),):
            raise ValueError(
                f"Box labels must be int64 with shape ({len(self)},), "
                f"got {self.labels.dtype} with shape {self.labels.shape}."
            )
        if len(self.names) != len(self):
            raise ValueError(f"Boxes declare {len(self.names)} names but hold {len(self)} boxes.")
        if self.num_lidar_points.dtype != np.int64 or self.num_lidar_points.shape != (len(self),):
            raise ValueError(
                f"Box lidar point counts must be int64 with shape ({len(self)},), "
                f"got {self.num_lidar_points.dtype} with shape {self.num_lidar_points.shape}."
            )
        return self

    def __len__(self) -> int:
        """
        Get the number of boxes.

        Returns:
          int: Number of boxes.
        """

        return self.params.shape[0]

    def filter(self, mask: Bool[np.ndarray, " num_boxes"]) -> Boxes3D:
        """
        Create boxes keeping only the masked rows.

        Args:
          mask: Boolean mask of the boxes to keep.

        Returns:
          Boxes3D: Filtered boxes.
        """

        if mask.dtype != np.bool_ or mask.shape != (len(self),):
            raise ValueError(
                f"Filter mask must be a boolean array of shape ({len(self)},), "
                f"got {mask.dtype} with shape {mask.shape}."
            )
        return self.model_copy(
            update={
                "params": self.params[mask],
                "labels": self.labels[mask],
                "names": tuple(name for name, keep in zip(self.names, mask) if keep),
                "num_lidar_points": self.num_lidar_points[mask],
            }
        )

    def with_params(self, params: Float32[np.ndarray, "num_boxes num_box_params"]) -> Boxes3D:
        """
        Create boxes with replaced parameters and untouched labels, names, and point counts.

        Args:
          params: New box parameters.

        Returns:
          Boxes3D: Boxes with the new parameters.
        """

        if params.shape != self.params.shape:
            raise ValueError(
                f"Replacement parameters must have shape {self.params.shape}, got {params.shape}."
            )
        return self.model_copy(update={"params": params.astype(np.float32)})
