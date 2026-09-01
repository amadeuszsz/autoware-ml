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

"""Tests for the 3D box filter transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from autoware_ml.datamodule.samples.boxes3d import Boxes3D
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.boxes3d.filters import ObjectRangeFilter, ObjectRangeMinPointsFilter
from autoware_ml.types.geometry import PointFeatureName


def _boxes(
    params_rows: Sequence[Sequence[float]],
    names: Sequence[str],
    num_lidar_points: Sequence[int],
) -> Boxes3D:
    return Boxes3D(
        params=np.asarray(params_rows, dtype=np.float32),
        labels=np.arange(len(names), dtype=np.int64),
        names=tuple(names),
        num_lidar_points=np.asarray(num_lidar_points, dtype=np.int64),
    )


def _point_cloud(
    coords: Sequence[Sequence[float]], time_lags: Sequence[float] | None = None
) -> PointCloud:
    coord = np.asarray(coords, dtype=np.float32)
    intensity = np.zeros((coord.shape[0], 1), dtype=np.float32)
    feature_names = [
        PointFeatureName.X,
        PointFeatureName.Y,
        PointFeatureName.Z,
        PointFeatureName.INTENSITY,
    ]
    columns = [coord, intensity]
    if time_lags is not None:
        columns.append(np.asarray(time_lags, dtype=np.float32).reshape(-1, 1))
        feature_names.append(PointFeatureName.TIMESTAMP_DIFFERENCE)
    return PointCloud(
        features=np.concatenate(columns, axis=1),
        feature_names=tuple(feature_names),
        num_current_points=None,
    )


def test_object_range_filter_keeps_boxes_inside_the_inclusive_range() -> None:
    sample = make_sample(
        boxes=_boxes(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
            names=["car", "truck", "pedestrian"],
            num_lidar_points=[7, 5, 3],
        )
    )

    output = ObjectRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 5.0, 5.0, 5.0])(sample)

    assert output.boxes.names == ("car", "truck")
    assert output.boxes.labels.tolist() == [0, 1]
    assert output.boxes.num_lidar_points.tolist() == [7, 5]
    assert output.boxes.params.shape == (2, 9)


def test_object_range_filter_requires_boxes() -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="'boxes' must be set"):
        ObjectRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 5.0, 5.0, 5.0])(sample)


def test_object_range_min_points_filter_counts_points_inside_rotated_boxes() -> None:
    # The box heads along +y (yaw pi / 2), so it spans 1 m along y and 0.5 m along x.
    # The point at x = 0.6 is outside; a yaw-blind count would place it inside.
    boxes = _boxes(
        [[0.0, 0.0, 0.0, 2.0, 1.0, 2.0, np.pi / 2, 0.0, 0.0]],
        names=["car"],
        num_lidar_points=[3],
    )
    points = _point_cloud([[0.0, 0.4, 0.0], [0.0, -0.4, 0.0], [0.6, 0.0, 0.0]])
    sample = make_sample(points=points, boxes=boxes)

    kept = ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=2)(sample)
    dropped = ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=3)(sample)

    assert kept.boxes.names == ("car",)
    assert dropped.boxes.names == ()


def test_object_range_min_points_filter_uses_distance_specific_threshold() -> None:
    boxes = _boxes(
        [
            [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [70.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        ],
        names=["car", "car"],
        num_lidar_points=[4, 3],
    )
    points = _point_cloud(
        [
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [10.3, 0.0, 0.0],
            [70.0, 0.0, 0.0],
            [70.1, 0.0, 0.0],
            [70.2, 0.0, 0.0],
        ]
    )
    sample = make_sample(points=points, boxes=boxes)

    near_filtered = ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=5)(sample)
    output = ObjectRangeMinPointsFilter(range_radius=[60.0, 130.0], min_num_points=3)(near_filtered)

    assert output.boxes.params[:, 0].tolist() == [70.0]


# A ground truth box is annotated on the current frame, so sweep points must never decide
# whether it survives: otherwise the surviving GT set would depend on how many sweeps the
# model consumes.
def test_object_range_min_points_filter_ignores_sweep_points() -> None:
    boxes = _boxes(
        [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]],
        names=["car"],
        num_lidar_points=[4],
    )
    points = _point_cloud(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        time_lags=[0.0, 0.0, 0.1, 0.1],
    )
    sample = make_sample(points=points, boxes=boxes)

    kept = ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=2)(sample)
    dropped = ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=3)(sample)

    assert kept.boxes.names == ("car",)
    assert dropped.boxes.names == ()


def test_object_range_min_points_filter_counts_every_point_without_time_lag() -> None:
    boxes = _boxes(
        [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]],
        names=["car"],
        num_lidar_points=[4],
    )
    points = _point_cloud([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]])
    sample = make_sample(points=points, boxes=boxes)

    output = ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=3)(sample)

    assert output.boxes.names == ("car",)


def test_object_range_min_points_filter_requires_points_and_boxes() -> None:
    boxes = _boxes(
        [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]],
        names=["car"],
        num_lidar_points=[4],
    )
    sample = make_sample(boxes=boxes)

    with pytest.raises(ValueError, match="'points' must be set"):
        ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=1)(sample)


def test_object_range_min_points_filter_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="min radius"):
        ObjectRangeMinPointsFilter(range_radius=[60.0, 60.0], min_num_points=1)

    with pytest.raises(ValueError, match="min_num_points"):
        ObjectRangeMinPointsFilter(range_radius=[0.0, 60.0], min_num_points=0)
