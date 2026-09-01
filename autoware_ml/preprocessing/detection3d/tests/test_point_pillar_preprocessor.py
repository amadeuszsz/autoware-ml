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

"""Tests for the point pillar voxelization preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.preprocessing.detection3d.point_pillar import (
    PillarInputs,
    PointPillarPreprocessor,
)
from autoware_ml.testing.factories import make_record, make_sample
from autoware_ml.types.geometry import PointFeatureName

FEATURE_NAMES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
)


def _batch(points_per_sample) -> Batch:
    samples = []
    for index, points in enumerate(points_per_sample):
        features = np.asarray(points, dtype=np.float32).reshape(-1, 4)
        cloud = PointCloud(
            features=features, feature_names=FEATURE_NAMES, num_current_points=len(features)
        )
        record = make_record(sample_id=f"sample-{index}")
        samples.append(make_sample(record=record, points=cloud))
    return Batch.collate(samples)


def _preprocessor(**overrides) -> PointPillarPreprocessor:
    settings = {
        "voxel_size": [1.0, 1.0, 4.0],
        "point_cloud_range": [0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
        "max_num_points": 2,
        "max_voxels": 8,
        "voxelization_z_order_first": True,
    }
    settings.update(overrides)
    return PointPillarPreprocessor(**settings)


def test_forward_builds_padded_pillars() -> None:
    batch = _batch(
        [
            [
                [0.1, 0.1, 0.0, 1.0],
                [0.2, 0.2, 0.0, 2.0],
                [1.1, 1.1, 0.0, 3.0],
            ]
        ]
    )

    outputs = _preprocessor()(batch, is_training=True)

    assert isinstance(outputs, PillarInputs)
    assert outputs.voxels.shape == (2, 2, 4)
    assert outputs.num_points.tolist() == [2, 1]
    assert outputs.voxel_coords.shape == (2, 4)
    assert outputs.voxel_coords[:, 0].tolist() == [0, 0]


def test_point_voxel_indices_follow_concatenated_batch_order() -> None:
    batch = _batch(
        [
            [[0.1, 0.1, 0.0, 1.0], [9.0, 9.0, 0.0, 2.0]],
            [[0.2, 0.2, 0.0, 3.0], [1.1, 1.1, 0.0, 4.0]],
        ]
    )

    outputs = _preprocessor()(batch, is_training=True)

    indices = outputs.point_voxel_indices
    assert indices.shape == (4,)
    # The out of range point stays unassigned.
    assert int(indices[1]) == -1
    assert sorted(indices[indices >= 0].tolist()) == [0, 1, 2]
    assert int(outputs.num_dropped_voxels) == 0
    # The voxel row a point maps to carries that point's batch index.
    assert int(outputs.voxel_coords[indices[0], 0]) == 0
    assert int(outputs.voxel_coords[indices[2], 0]) == 1
    assert int(outputs.voxel_coords[indices[3], 0]) == 1


def test_batch_column_increments_per_sample() -> None:
    point = [[0.5, 0.5, 0.0, 1.0]]
    batch = _batch([point, point, point])

    outputs = _preprocessor()(batch, is_training=True)

    assert outputs.voxel_coords[:, 0].tolist() == [0, 1, 2]


def test_empty_sample_inside_a_batch_contributes_no_voxels() -> None:
    point = [[0.5, 0.5, 0.0, 1.0]]
    batch = _batch([point, np.zeros((0, 4), dtype=np.float32), point])

    outputs = _preprocessor()(batch, is_training=True)

    assert outputs.voxels.shape[0] == 2
    assert set(outputs.voxel_coords[:, 0].tolist()) == {0, 2}


def test_points_all_out_of_range_produce_empty_pillar_tensors() -> None:
    batch = _batch([[[9.0, 9.0, 0.0, 1.0]]])

    outputs = _preprocessor()(batch, is_training=True)

    assert outputs.voxels.shape == (0, 2, 4)
    assert outputs.num_points.shape == (0,)
    assert outputs.voxel_coords.shape == (0, 4)
    assert outputs.point_voxel_indices.tolist() == [-1]
    assert int(outputs.num_dropped_voxels) == 0


def test_eval_mode_uses_the_eval_voxel_budget() -> None:
    preprocessor = _preprocessor(max_voxels=1, eval_max_voxels=8)
    batch = _batch(
        [
            [
                [0.1, 0.1, 0.0, 1.0],
                [1.1, 1.1, 0.0, 2.0],
                [2.1, 2.1, 0.0, 3.0],
            ]
        ]
    )

    train_outputs = preprocessor(batch, is_training=True)
    eval_outputs = preprocessor(batch, is_training=False)

    assert train_outputs.voxels.shape[0] == 1
    assert int(train_outputs.num_dropped_voxels) == 2
    assert eval_outputs.voxels.shape[0] == 3


def test_eval_mode_without_eval_max_voxels_raises() -> None:
    batch = _batch([[[0.5, 0.5, 0.0, 1.0]]])

    with pytest.raises(ValueError, match="eval_max_voxels"):
        _preprocessor()(batch, is_training=False)


def test_train_mode_does_not_require_eval_max_voxels() -> None:
    batch = _batch([[[0.5, 0.5, 0.0, 1.0]]])

    outputs = _preprocessor()(batch, is_training=True)

    assert outputs.voxels.shape[0] == 1


def test_a_batch_without_point_clouds_is_rejected() -> None:
    batch = Batch.collate([make_sample()])

    with pytest.raises(ValueError, match="requires a point cloud batch"):
        _preprocessor()(batch, is_training=True)
