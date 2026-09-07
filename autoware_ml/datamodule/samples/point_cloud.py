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

from autoware_ml.types.geometry import PointFeatureName


class PointCloud(BaseModel):
    """
    Point cloud of one sample. Points are stored as a single packed feature matrix whose
    columns are described by feature_names, with the coordinates always in the first three
    columns. Sweep points follow the current frame points, so the current frame is the leading
    block of rows whenever num_current_points is known.

    Attributes:
      features: Packed point features with one row per point.
      feature_names: Name of every feature column.
      num_current_points: Number of current frame points forming the leading row block. None
        when a transform reordered the rows and the leading block is not tracked anymore.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    features: Float32[np.ndarray, "num_points num_features"]
    feature_names: tuple[PointFeatureName, ...]
    num_current_points: int | None

    @model_validator(mode="after")
    def validate_point_cloud(self) -> PointCloud:
        """
        Validate the feature matrix against the declared feature names.

        Returns:
          PointCloud: The validated point cloud.
        """

        if self.features.ndim != 2:
            raise ValueError(f"Point features must be 2D, got shape {self.features.shape}.")
        if self.features.dtype != np.float32:
            raise ValueError(f"Point features must be float32, got {self.features.dtype}.")
        if len(self.feature_names) != self.features.shape[1]:
            raise ValueError(
                f"Point cloud declares {len(self.feature_names)} feature names but the feature "
                f"matrix has {self.features.shape[1]} columns."
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(f"Point feature names must be unique, got {self.feature_names}.")
        expected_leading = (PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z)
        if self.feature_names[:3] != expected_leading:
            raise ValueError(
                f"The first three point features must be {expected_leading}, "
                f"got {self.feature_names[:3]}."
            )
        if self.num_current_points is not None and not (0 <= self.num_current_points <= len(self)):
            raise ValueError(
                f"num_current_points {self.num_current_points} is outside the point count "
                f"{len(self)}."
            )
        return self

    def __len__(self) -> int:
        """
        Get the number of points.

        Returns:
          int: Number of points.
        """

        return self.features.shape[0]

    @property
    def coord(self) -> Float32[np.ndarray, "num_points 3"]:
        """
        Get the point coordinates.

        Returns:
          Float32[np.ndarray, "num_points 3"]: Point coordinates.
        """

        return self.features[:, :3]

    def has_feature(self, feature_name: PointFeatureName) -> bool:
        """
        Check whether a feature column exists.

        Args:
          feature_name: Name of the feature column.

        Returns:
          bool: True when the feature column exists.
        """

        return feature_name in self.feature_names

    def feature(self, feature_name: PointFeatureName) -> Float32[np.ndarray, " num_points"]:
        """
        Get one feature column.

        Args:
          feature_name: Name of the feature column.

        Returns:
          Float32[np.ndarray, " num_points"]: The feature column.
        """

        if feature_name not in self.feature_names:
            raise KeyError(
                f"Point cloud has no feature {feature_name}, available: {self.feature_names}."
            )
        return self.features[:, self.feature_names.index(feature_name)]

    def pack(
        self, feature_names: tuple[PointFeatureName, ...]
    ) -> Float32[np.ndarray, "num_points num_selected_features"]:
        """
        Pack the selected feature columns into a new matrix.

        Args:
          feature_names: Names of the feature columns to pack, in the requested order.

        Returns:
          Float32[np.ndarray, "num_points num_selected_features"]: Packed feature matrix.
        """

        column_indices = []
        for feature_name in feature_names:
            if feature_name not in self.feature_names:
                raise KeyError(
                    f"Point cloud has no feature {feature_name}, available: {self.feature_names}."
                )
            column_indices.append(self.feature_names.index(feature_name))
        return self.features[:, column_indices]

    def with_coord(self, coord: Float32[np.ndarray, "num_points 3"]) -> PointCloud:
        """
        Create a point cloud with replaced coordinates and untouched remaining features.

        Args:
          coord: New point coordinates.

        Returns:
          PointCloud: Point cloud with the new coordinates.
        """

        if coord.shape != (len(self), 3):
            raise ValueError(
                f"Replacement coordinates must have shape ({len(self)}, 3), got {coord.shape}."
            )
        features = self.features.copy()
        features[:, :3] = coord.astype(np.float32)
        return self.model_copy(update={"features": features})

    def with_feature(
        self, feature_name: PointFeatureName, values: Float32[np.ndarray, " num_points"]
    ) -> PointCloud:
        """
        Create a point cloud with one replaced feature column.

        Args:
          feature_name: Name of the feature column to replace.
          values: New feature values.

        Returns:
          PointCloud: Point cloud with the replaced feature column.
        """

        if feature_name not in self.feature_names:
            raise KeyError(
                f"Point cloud has no feature {feature_name}, available: {self.feature_names}."
            )
        if values.shape != (len(self),):
            raise ValueError(
                f"Replacement feature must have shape ({len(self)},), got {values.shape}."
            )
        features = self.features.copy()
        features[:, self.feature_names.index(feature_name)] = values.astype(np.float32)
        return self.model_copy(update={"features": features})

    def filter(self, mask: Bool[np.ndarray, " num_points"]) -> PointCloud:
        """
        Create a point cloud keeping only the masked rows. The row order is preserved, so the
        leading current frame block stays a leading block and num_current_points is recounted.

        Args:
          mask: Boolean mask of the rows to keep.

        Returns:
          PointCloud: Filtered point cloud.
        """

        if mask.dtype != np.bool_ or mask.shape != (len(self),):
            raise ValueError(
                f"Filter mask must be a boolean array of shape ({len(self)},), "
                f"got {mask.dtype} with shape {mask.shape}."
            )
        num_current_points = self.num_current_points
        if num_current_points is not None:
            num_current_points = int(mask[:num_current_points].sum())
        return self.model_copy(
            update={
                "features": self.features[mask],
                "num_current_points": num_current_points,
            }
        )

    def reorder(self, indices: Int64[np.ndarray, " num_points"]) -> PointCloud:
        """
        Create a point cloud with reordered rows. Reordering breaks the leading current frame
        block, so num_current_points becomes None.

        Args:
          indices: Permutation of the row indices.

        Returns:
          PointCloud: Reordered point cloud.
        """

        if indices.shape != (len(self),):
            raise ValueError(
                f"Reorder indices must have shape ({len(self)},), got {indices.shape}."
            )
        return self.model_copy(
            update={
                "features": self.features[indices],
                "num_current_points": None,
            }
        )
