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
from enum import StrEnum
from typing import Annotated

import numpy as np
from jaxtyping import Float32
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

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

# A stored sweep paired with its signed time lag, the current frame timestamp minus its own.
LaggedFrame = tuple[float, LidarFrameDataModel]


class SweepSelection(StrEnum):
    """
    How the appended sweeps are picked among the eligible stored frames.

    Attributes:
      NEAREST: The frames closest in time to the current frame, which is what evaluation and
        deployment see.
      RANDOM: A uniform sample without replacement, which varies the temporal baseline during
        training so the network reads the time lag instead of assuming a fixed frame interval.
    """

    NEAREST = "nearest"
    RANDOM = "random"


class SweepWindow(BaseModel):
    """
    Sweeps appended from one side of the current frame.

    time_lag_range bounds the distance in seconds between an eligible stored frame and the
    current frame. Frames outside it are unavailable, exactly like the frames a scene does not
    have before its first or after its last sample, and an unavailable frame contributes no
    points.

    Attributes:
      num: Number of sweeps appended when enough frames are eligible.
      time_lag_range: Inclusive [min, max] distance in seconds, with 0 < min < max. The
        current frame owns lag 0, so a zero minimum would let a sweep masquerade as it.
      selection: How the appended sweeps are picked among the eligible frames.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    num: int = Field(ge=1)
    time_lag_range: Annotated[tuple[float, float], BeforeValidator(tuple)]
    selection: Annotated[SweepSelection, BeforeValidator(SweepSelection)]

    @model_validator(mode="after")
    def validate_time_lag_range(self) -> SweepWindow:
        """
        Validate the ordering of the time lag bounds.

        Returns:
          SweepWindow: The validated window.
        """

        if self.min_time_lag <= 0.0 or self.min_time_lag >= self.max_time_lag:
            raise ValueError(
                f"Expected 0 < min time lag < max time lag, got {list(self.time_lag_range)}."
            )
        return self

    @property
    def min_time_lag(self) -> float:
        """Smallest distance in seconds an eligible frame may have."""
        return self.time_lag_range[0]

    @property
    def max_time_lag(self) -> float:
        """Largest distance in seconds an eligible frame may have."""
        return self.time_lag_range[1]


def _distance(lagged_frame: LaggedFrame) -> float:
    """Distance in seconds between a stored sweep and the current frame."""
    return abs(lagged_frame[0])


class LoadPointsFromMultiSweeps(BaseTransform):
    """Load the current frame and append stored sweep points from the sample record.

    This is the single point loading transform. Without a past or a future window it loads the
    current frame alone and the sweep arguments must stay unset, so a single frame pipeline
    declares no meaningless sweep knobs.

    The stored frames of a record are split by their capture time into the frames before and
    the frames after the current frame, and each window declares what is appended from its
    side. When use_features contains timestamp_difference, every point carries the current
    frame timestamp minus its own capture timestamp: 0 for the current frame, positive for past
    sweeps and negative for future sweeps. The current frame is always the leading block of the
    output and its size is exposed as num_current_points so label pipelines can pad the
    unlabeled sweep points. The past sweeps follow it nearest first, then the future sweeps
    nearest first.

    A sweep that is unavailable, because the scene ends, the dataset flagged the frame invalid
    or its distance falls outside the window, contributes no points. Nothing is duplicated or
    synthesised in its place, so every split sees the same rule.
    """

    def __init__(
        self,
        *,
        use_features: Sequence[str | PointFeatureName],
        past: SweepWindow | None = None,
        future: SweepWindow | None = None,
        remove_close: bool = False,
        close_radius: float = 1.0,
    ) -> None:
        """Initialize the LoadPointsFromMultiSweeps transform.

        Args:
            use_features: Feature columns of the loaded point cloud, starting with x, y, and z.
                When timestamp_difference is included, the transform computes it per point.
            past: Sweeps appended from the frames captured before the current frame, none
                when omitted.
            future: Sweeps appended from the frames captured after the current frame, none
                when omitted.
            remove_close: Whether to drop sweep points close to the origin. Requires a window.
            close_radius: Half width in meters of the removed region when remove_close is
                enabled.
        """
        self.use_features = coerce_feature_names(use_features)
        self.raw_features = tuple(
            feature_name
            for feature_name in self.use_features
            if feature_name != PointFeatureName.TIMESTAMP_DIFFERENCE
        )
        self._check_window("past", past)
        self._check_window("future", future)
        if past is None and future is None and remove_close:
            raise ValueError(
                "A single frame load appends no sweeps, so remove_close must stay unset."
            )
        self.past = past
        self.future = future
        self.remove_close = remove_close
        self.close_radius = close_radius

    @staticmethod
    def _check_window(name: str, window: SweepWindow | None) -> None:
        """Reject a window that is not a SweepWindow, such as a mapping Hydra did not build.

        Args:
            name: Argument name for the error message.
            window: Configured window.
        """
        if window is not None and not isinstance(window, SweepWindow):
            raise TypeError(f"{name} must be a SweepWindow or None, got {type(window).__name__}.")

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
        past_frames, future_frames = self._split_stored_frames(sample, keyframe)
        selected_sweeps = [
            *self._select_sweeps(self.past, past_frames),
            *self._select_sweeps(self.future, future_frames),
        ]

        feature_blocks = [current_features]
        for time_lag, sweep_frame in selected_sweeps:
            sweep_features = self._build_frame_features(
                sample=sample, lidar_frame=sweep_frame, time_lag=time_lag
            )
            if self.remove_close:
                sweep_features = self._remove_close_points(sweep_features)
            feature_blocks.append(self._transform_sweep_to_keyframe(sweep_features, sweep_frame))

        point_cloud = PointCloud(
            features=np.ascontiguousarray(np.concatenate(feature_blocks, axis=0), dtype=np.float32),
            feature_names=self.use_features,
            num_current_points=current_features.shape[0],
        )
        return sample.replace(points=point_cloud)

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

    @staticmethod
    def _split_stored_frames(
        sample: Sample, keyframe: LidarFrameDataModel
    ) -> tuple[list[LaggedFrame], list[LaggedFrame]]:
        """Pair every stored sweep with its signed time lag and split them by side.

        The frames after the first are the stored sweeps of the sample, whatever their keyframe
        flag says about the annotation of the dataset. A stored sweep sharing the timestamp of
        the current frame is rejected, every consumer identifies the current frame by lag 0.

        Args:
            sample: Sample holding the dataset record.
            keyframe: Lidar frame of the sample.

        Returns:
            tuple[list[LaggedFrame], list[LaggedFrame]]: Frames captured before the current
                frame and frames captured after it, in stored order.
        """
        key_timestamp = keyframe.lidar_timestamp_seconds
        past_frames: list[LaggedFrame] = []
        future_frames: list[LaggedFrame] = []
        for sweep_frame in sample.record.lidar_frames[1:]:
            time_lag = key_timestamp - sweep_frame.lidar_timestamp_seconds
            if time_lag == 0.0:
                raise ValueError(
                    f"Sample {sample.meta.sample_id} stores sweep {sweep_frame.lidar_frame_id} "
                    "at the timestamp of its current frame."
                )
            side = past_frames if time_lag > 0.0 else future_frames
            side.append((time_lag, sweep_frame))
        return past_frames, future_frames

    @staticmethod
    def _select_sweeps(
        window: SweepWindow | None, lagged_frames: Sequence[LaggedFrame]
    ) -> list[LaggedFrame]:
        """Return the sweeps a window appends from one side, nearest first.

        Frames whose distance falls outside the window are unavailable, so a scene whose
        neighbouring frames were dropped yields fewer sweeps rather than a stale one.

        Args:
            window: Window declared for this side, None when nothing is appended from it.
            lagged_frames: Stored sweeps of this side with their signed time lag.

        Returns:
            list[LaggedFrame]: Selected sweeps with their signed time lag.
        """
        if window is None:
            return []
        eligible = [
            lagged_frame
            for lagged_frame in lagged_frames
            if window.min_time_lag <= _distance(lagged_frame) <= window.max_time_lag
        ]
        eligible.sort(key=_distance)
        if window.selection is SweepSelection.RANDOM and len(eligible) > window.num:
            indices = np.random.choice(len(eligible), window.num, replace=False)
            return sorted((eligible[index] for index in indices), key=_distance)
        return eligible[: window.num]

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
