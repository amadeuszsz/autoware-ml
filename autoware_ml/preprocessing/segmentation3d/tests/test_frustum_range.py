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

"""Tests for the frustum range view preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.preprocessing.segmentation3d.frustum_range import (
    FrustumInputs,
    FrustumRangePreprocessor,
)
from autoware_ml.testing.factories import make_record, make_sample
from autoware_ml.types.geometry import PointFeatureName

FEATURE_NAMES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
)


def _batch(points_per_sample, labels_per_sample=None) -> Batch:
    samples = []
    for index, points in enumerate(points_per_sample):
        features = np.asarray(points, dtype=np.float32).reshape(-1, 4)
        cloud = PointCloud(
            features=features, feature_names=FEATURE_NAMES, num_current_points=len(features)
        )
        record = make_record(sample_id=f"sample-{index}")
        base = make_sample(record=record, points=cloud)
        if labels_per_sample is None:
            samples.append(base)
            continue
        segment = SegmentationLabels(labels=np.asarray(labels_per_sample[index], dtype=np.int64))
        samples.append(
            Sample(
                record=base.record,
                data_root=base.data_root,
                meta=base.meta,
                points=cloud,
                segment=segment,
            )
        )
    return Batch.collate(samples)


def _preprocessor(**overrides) -> FrustumRangePreprocessor:
    settings = {
        "height": 2,
        "width": 4,
        "fov_up": 10.0,
        "fov_down": -10.0,
        "ignore_index": 255,
        "num_classes": 4,
    }
    settings.update(overrides)
    return FrustumRangePreprocessor(**settings)


def test_forward_builds_sparse_frustum_targets() -> None:
    batch = _batch(
        [[[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2], [1.0, 1.0, 0.0, 0.3]]],
        [[3, 3, 1]],
    )

    outputs = _preprocessor()(batch, is_training=False)

    assert isinstance(outputs, FrustumInputs)
    assert outputs.points.shape == (3, 4)
    assert outputs.coors.shape == (3, 3)
    # The first two points share one range view cell.
    assert outputs.voxel_coors.shape == (2, 3)
    assert outputs.inverse_map.shape == (3,)
    assert torch.equal(outputs.pts_semantic_mask, torch.tensor([3, 3, 1]))
    assert outputs.semantic_seg.shape == (1, 2, 4)
    assert outputs.semantic_seg[0, 1, 2].item() == 3
    assert outputs.semantic_seg[0, 1, 1].item() == 1
    assert outputs.semantic_seg[0, 0, 0].item() == 255


def test_forward_concatenates_a_batch_of_two_samples() -> None:
    batch = _batch(
        [
            [[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2]],
            [[1.0, 1.0, 0.0, 0.3]],
        ],
        [[0, 1], [2]],
    )

    outputs = _preprocessor(num_classes=3)(batch, is_training=False)

    assert outputs.points.shape == (3, 4)
    assert outputs.pts_semantic_mask.shape == (3,)
    assert outputs.semantic_seg.shape == (2, 2, 4)
    assert outputs.sample_count == 2


def test_forward_without_labels_produces_no_targets() -> None:
    batch = _batch([[[1.0, 0.0, 0.0, 0.1]]])

    outputs = _preprocessor()(batch, is_training=False)

    assert outputs.pts_semantic_mask is None
    assert outputs.semantic_seg is None
    assert outputs.points.shape == (1, 4)
    assert outputs.voxel_coors.shape == (1, 3)


def test_forward_masks_negative_ignore_labels_before_the_majority_vote() -> None:
    batch = _batch(
        [[[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2]]],
        [[-1, 2]],
    )

    outputs = _preprocessor(ignore_index=-1, num_classes=3)(batch, is_training=False)

    assert outputs.semantic_seg.shape == (1, 2, 4)
    assert (outputs.semantic_seg == 2).any()
    assert (outputs.semantic_seg == -1).any()


def test_forward_keeps_ignore_only_cells_at_the_ignore_index() -> None:
    batch = _batch(
        [[[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2], [1.0, 1.0, 0.0, 0.3]]],
        [[255, 255, 1]],
    )

    outputs = _preprocessor(num_classes=3)(batch, is_training=False)

    assert outputs.semantic_seg[0, 1, 2].item() == 255
    assert outputs.semantic_seg[0, 1, 1].item() == 1


def test_a_batch_without_point_clouds_is_rejected() -> None:
    batch = Batch.collate([make_sample()])

    with pytest.raises(ValueError, match="requires a point cloud batch"):
        _preprocessor()(batch, is_training=False)
