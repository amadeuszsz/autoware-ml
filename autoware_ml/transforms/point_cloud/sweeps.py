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

"""Point cloud sweep loading transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float32

from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.loading import (
    coerce_feature_names,
    keyframe_lidar_frame,
    load_frame_points,
    select_raw_features,
)
from autoware_ml.types.geometry import PointFeatureName

SWEEP_SELECTIONS = frozenset({"nearest", "random"})


class LoadPointsFromMultiSweeps(BaseTransform):
    """Load the current frame and append historical sweep points from the sample record.

    This is the single point loading transform. With sweeps_num 1 it loads the current frame
    alone and the sweep arguments must stay unset, so a single frame pipeline declares no
    meaningless sweep knobs.

    When use_features contains timestamp_difference, every point carries the seconds elapsed
    since it was captured, 0 for the current frame and key timestamp minus sweep timestamp for
    sweep points. The current frame is always the leading block of the output and its size is
    exposed as num_current_points so label pipelines can pad the unlabeled sweep points.

    Which stored sweeps are appended is declared, not inferred. time_lag_range bounds the age
    of an eligible sweep in seconds, and sweep_selection picks among the eligible ones:
    "nearest" takes the most recent, which is what evaluation and deployment see, and "random"
    samples them, which varies the temporal baseline during training so the network has to read
    the time lag instead of assuming a fixed frame interval. Sweeps outside the window count as
    unavailable, exactly like the missing sweep of a scene's first frame.
    """

    def __init__(
        self,
        *,
        sweeps_num: int,
        use_features: Sequence[str | PointFeatureName],
        sweep_selection: str | None = None,
        time_lag_range: Sequence[float] | None = None,
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        close_radius: float = 1.0,
    ) -> None:
        """Initialize the LoadPointsFromMultiSweeps transform.

        Args:
            sweeps_num: Number of sweeps included in the output including the current frame,
                at least 1. With 1 only the current frame is loaded.
            use_features: Feature columns of the loaded point cloud, starting with x, y, and z.
                When timestamp_difference is included, the transform computes it per point.
            sweep_selection: How to pick the appended sweeps among the eligible entries:
                "nearest" takes the most recent ones, "random" samples them uniformly without
                replacement. Required when sweeps_num is above 1, forbidden otherwise.
            time_lag_range: Inclusive [min, max] age in seconds an eligible sweep may have,
                with 0 < min < max. Entries outside it are treated as unavailable. Required
                when sweeps_num is above 1, forbidden otherwise.
            pad_empty_sweeps: Whether to repeat the current frame when no sweeps exist. The
                copies carry the minimum time lag, so current frame selections never count
                them.
            remove_close: Whether to drop sweep points close to the origin.
            close_radius: Half width in meters of the removed region when remove_close is
                enabled.
        """
        if sweeps_num < 1:
            raise ValueError(f"sweeps_num must be at least 1, got {sweeps_num}.")
        self.sweeps_num = sweeps_num
        self.use_features = coerce_feature_names(use_features)
        self.raw_features = tuple(
            feature_name
            for feature_name in self.use_features
            if feature_name != PointFeatureName.TIMESTAMP_DIFFERENCE
        )
        if sweeps_num == 1:
            sweep_arguments = (sweep_selection, time_lag_range, pad_empty_sweeps, remove_close)
            if any(argument for argument in sweep_arguments):
                raise ValueError(
                    "A single frame load appends no sweeps, so sweep_selection, "
                    "time_lag_range, pad_empty_sweeps, and remove_close must stay unset."
                )
            self.sweep_selection = None
            self.min_time_lag = None
            self.max_time_lag = None
        else:
            if sweep_selection not in SWEEP_SELECTIONS:
                raise ValueError(
                    f"sweep_selection must be one of {sorted(SWEEP_SELECTIONS)}, "
                    f"got {sweep_selection!r}."
                )
            if time_lag_range is None or len(time_lag_range) != 2:
                raise ValueError(f"time_lag_range must contain [min, max], got {time_lag_range}.")
            min_time_lag, max_time_lag = (float(value) for value in time_lag_range)
            if min_time_lag <= 0.0 or min_time_lag >= max_time_lag:
                raise ValueError(
                    f"Expected 0 < min time lag < max time lag, got {time_lag_range}. The "
                    "current frame owns lag 0, a zero minimum would let a zero lag sweep "
                    "masquerade as it."
                )
            self.sweep_selection = sweep_selection
            self.min_time_lag = min_time_lag
            self.max_time_lag = max_time_lag
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.close_radius = close_radius

    def transform(self, sample: Sample) -> Sample:
        """Load the current frame and append the selected sweep points.

        Args:
            sample: Sample holding the dataset record.

        Returns:
            Sample with the loaded multi sweep point cloud.
        """
        keyframe = keyframe_lidar_frame(sample)
        current_features = self._build_frame_features(
            sample=sample, lidar_frame=keyframe, time_lag=0.0
        )
        num_current_points = current_features.shape[0]

        sweep_frames = sample.record.lidar_frames[1:]
        selected_sweeps = self._select_sweeps(
            sweep_frames=sweep_frames,
            needed=max(0, self.sweeps_num - 1),
            key_timestamp=keyframe.lidar_timestamp_seconds,
        )

        feature_blocks = [current_features]
        if not selected_sweeps and self.pad_empty_sweeps and self.sweeps_num > 1:
            padding = np.tile(current_features, (self.sweeps_num - 1, 1))
            if PointFeatureName.TIMESTAMP_DIFFERENCE in self.use_features:
                # Padded copies stand in for sweeps, so they carry the youngest admissible
                # lag and never masquerade as current frame points.
                time_column = self.use_features.index(PointFeatureName.TIMESTAMP_DIFFERENCE)
                padding[:, time_column] = self.min_time_lag
            feature_blocks.append(padding)

        for time_lag, sweep_frame in selected_sweeps:
            sweep_features = self._build_frame_features(
                sample=sample, lidar_frame=sweep_frame, time_lag=time_lag
            )
            if self.remove_close:
                sweep_features = self._remove_close_points(sweep_features)
            sweep_features = self._transform_sweep_to_keyframe(sweep_features, sweep_frame)
            feature_blocks.append(sweep_features)

        point_cloud = PointCloud(
            features=np.ascontiguousarray(np.concatenate(feature_blocks, axis=0), dtype=np.float32),
            feature_names=self.use_features,
            num_current_points=num_current_points,
        )
        return sample.model_copy(update={"points": point_cloud})

    def _build_frame_features(
        self, sample: Sample, lidar_frame: LidarFrameDataModel, time_lag: float
    ) -> Float32[np.ndarray, "num_points num_features"]:
        """Load one frame and assemble its feature columns in the configured order.

        Args:
            sample: Sample holding the dataset record.
            lidar_frame: Lidar frame to load.
            time_lag: Time lag stamped on the timestamp_difference column when configured.

        Returns:
            Float32[np.ndarray, "num_points num_features"]: Feature matrix of the frame.
        """
        raw_points = load_frame_points(sample.data_root, lidar_frame)
        raw_columns = select_raw_features(raw_points, self.raw_features)
        if PointFeatureName.TIMESTAMP_DIFFERENCE not in self.use_features:
            return raw_columns

        features = np.empty((raw_points.shape[0], len(self.use_features)), dtype=np.float32)
        raw_cursor = 0
        for column, feature_name in enumerate(self.use_features):
            if feature_name == PointFeatureName.TIMESTAMP_DIFFERENCE:
                features[:, column] = time_lag
            else:
                features[:, column] = raw_columns[:, raw_cursor]
                raw_cursor += 1
        return features

    def _select_sweeps(
        self,
        sweep_frames: Sequence[LidarFrameDataModel],
        needed: int,
        key_timestamp: float,
    ) -> list[tuple[float, LidarFrameDataModel]]:
        """Return the sweeps to append, newest first, paired with their age in seconds.

        Frames outside time_lag_range are unavailable, so a scene whose previous frames were
        dropped yields fewer sweeps rather than a stale one.

        Args:
            sweep_frames: Stored sweep frames of the record.
            needed: Number of sweeps to append.
            key_timestamp: Capture time of the current frame every sweep age is measured
                against.

        Returns:
            list[tuple[float, LidarFrameDataModel]]: Selected sweeps with their time lag.
        """
        if needed == 0:
            return []
        eligible = []
        for sweep_frame in sweep_frames:
            time_lag = key_timestamp - sweep_frame.lidar_timestamp_seconds
            if self.min_time_lag <= time_lag <= self.max_time_lag:
                eligible.append((time_lag, sweep_frame))
        eligible.sort(key=lambda entry: entry[0])
        if self.sweep_selection == "random" and len(eligible) > needed:
            indices = np.random.choice(len(eligible), needed, replace=False)
            return sorted((eligible[index] for index in indices), key=lambda entry: entry[0])
        return eligible[:needed]

    @staticmethod
    def _transform_sweep_to_keyframe(
        features: Float32[np.ndarray, "num_points num_features"],
        sweep_frame: LidarFrameDataModel,
    ) -> Float32[np.ndarray, "num_points num_features"]:
        """Transform sweep point coordinates into the keyframe lidar frame.

        Args:
            features: Feature matrix of the sweep with the coordinates in the first three
                columns.
            sweep_frame: Lidar frame the points were loaded from.

        Returns:
            Float32[np.ndarray, "num_points num_features"]: Feature matrix with transformed
                coordinates.
        """
        sweep_to_keyframe = np.linalg.inv(sweep_frame.lidar_sensor_to_lidar_sweep_matrix).astype(
            np.float32
        )
        features = features.copy()
        features[:, :3] = features[:, :3] @ sweep_to_keyframe[:3, :3].T + sweep_to_keyframe[:3, 3]
        return features

    def _remove_close_points(
        self, features: Float32[np.ndarray, "num_points num_features"]
    ) -> Float32[np.ndarray, "num_kept_points num_features"]:
        """Remove points close to the origin in the xy plane.

        The removed region is the axis aligned box |x|, |y| < close_radius.

        Args:
            features: Feature matrix with the coordinates in the first three columns.

        Returns:
            Float32[np.ndarray, "num_kept_points num_features"]: Feature matrix without the
                close points.
        """
        close = (np.abs(features[:, 0]) < self.close_radius) & (
            np.abs(features[:, 1]) < self.close_radius
        )
        return features[~close]
