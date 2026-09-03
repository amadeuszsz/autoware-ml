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

"""Point cloud loading helpers shared by the loading transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float32

from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.point_cloud.paths import resolve_frame_path
from autoware_ml.types.geometry import PointFeatureName

# Raw feature column names of a stored point cloud file, in storage order.
RAW_FEATURE_NAMES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
    PointFeatureName.RING,
)


def coerce_feature_names(
    feature_names: Sequence[str | PointFeatureName],
) -> tuple[PointFeatureName, ...]:
    """
    Coerce configured feature names to PointFeatureName members.

    Args:
      feature_names: Configured feature names.

    Returns:
      tuple[PointFeatureName, ...]: Coerced feature names.
    """

    return tuple(PointFeatureName(feature_name) for feature_name in feature_names)


def load_frame_points(
    data_root: str, lidar_frame: LidarFrameDataModel
) -> Float32[np.ndarray, "num_points num_features"]:
    """
    Load the raw point matrix of one lidar frame.

    Args:
      data_root: Root directory of the dataset files.
      lidar_frame: Lidar frame data model of the frame.

    Returns:
      Float32[np.ndarray, "num_points num_features"]: Raw point matrix with the stored feature
        layout.
    """

    path = resolve_frame_path(data_root, lidar_frame.lidar_pointcloud_path)
    num_features = lidar_frame.lidar_pointcloud_num_features
    return np.fromfile(path, dtype=np.float32).reshape(-1, num_features)


def select_raw_features(
    points: Float32[np.ndarray, "num_points num_features"],
    feature_names: Sequence[PointFeatureName],
) -> Float32[np.ndarray, "num_points num_selected_features"]:
    """
    Select raw feature columns by name from a raw point matrix.

    Args:
      points: Raw point matrix with the stored feature layout.
      feature_names: Names of the raw feature columns to select, in the requested order.

    Returns:
      Float32[np.ndarray, "num_points num_selected_features"]: Selected feature columns.
    """

    column_indices = []
    for feature_name in feature_names:
        if feature_name not in RAW_FEATURE_NAMES:
            raise ValueError(
                f"{feature_name} is not a raw point feature, available: {RAW_FEATURE_NAMES}."
            )
        column_index = RAW_FEATURE_NAMES.index(feature_name)
        if column_index >= points.shape[1]:
            raise ValueError(
                f"The stored point cloud carries {points.shape[1]} features and has no "
                f"column for {feature_name}."
            )
        column_indices.append(column_index)
    return points[:, column_indices]


def keyframe_lidar_frame(sample: Sample) -> LidarFrameDataModel:
    """
    Get the keyframe lidar frame of a sample record. The first lidar frame of a record is the
    frame of the sample and the following frames are its preceding sweeps. The keyframe flag
    marks a frame the dataset annotates, so in a corpus annotated at every frame the sweeps
    carry it as well and the position decides.

    Args:
      sample: Sample holding the dataset record.

    Returns:
      LidarFrameDataModel: The keyframe lidar frame.
    """

    if not len(sample.record.lidar_frames):
        raise ValueError(f"The record of sample {sample.meta.sample_id} has no lidar frames.")
    keyframe = sample.record.lidar_frames[0]
    if not keyframe.lidar_keyframe:
        raise ValueError(
            f"The first lidar frame of sample {sample.meta.sample_id} is not the keyframe."
        )
    return keyframe


__all__ = [
    "RAW_FEATURE_NAMES",
    "coerce_feature_names",
    "keyframe_lidar_frame",
    "load_frame_points",
    "resolve_frame_path",
    "select_raw_features",
]
