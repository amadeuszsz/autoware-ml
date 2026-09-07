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

"""Tests for the sample model validators and the point cloud row operations."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.testing.factories import make_point_cloud, make_sample
from autoware_ml.types.geometry import PointFeatureName


def _segmented_sample(num_points: int, num_current_points: int | None = None) -> Sample:
    points = make_point_cloud(num_points=num_points, num_current_points=num_current_points)
    base = make_sample(points=points)
    segment = SegmentationLabels(labels=np.arange(num_points, dtype=np.int64))
    return Sample(
        record=base.record,
        data_root=base.data_root,
        meta=base.meta,
        points=points,
        segment=segment,
    )


def test_sample_rejects_segmentation_labels_without_points() -> None:
    base = make_sample()

    with pytest.raises(ValidationError, match="require a loaded point cloud"):
        Sample(
            record=base.record,
            data_root=base.data_root,
            meta=base.meta,
            segment=SegmentationLabels(labels=np.zeros(3, dtype=np.int64)),
        )


def test_sample_rejects_misaligned_segmentation_labels() -> None:
    base = make_sample(points=make_point_cloud(num_points=4))

    with pytest.raises(ValidationError, match="Segmentation labels cover"):
        Sample(
            record=base.record,
            data_root=base.data_root,
            meta=base.meta,
            points=base.points,
            segment=SegmentationLabels(labels=np.zeros(3, dtype=np.int64)),
        )


def test_replace_validates_the_derived_sample() -> None:
    sample = _segmented_sample(4)

    replaced = sample.replace(points=make_point_cloud(num_points=4, seed=1))

    assert replaced.segment is sample.segment
    assert not np.array_equal(replaced.points.features, sample.points.features)
    with pytest.raises(ValidationError, match="Segmentation labels cover"):
        sample.replace(points=make_point_cloud(num_points=3))


def test_model_copy_is_rejected_in_favour_of_replace() -> None:
    sample = _segmented_sample(4)

    with pytest.raises(TypeError, match="use Sample.replace"):
        sample.model_copy(update={"points": make_point_cloud(num_points=3)})


def test_point_operations_require_a_loaded_point_cloud() -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="filter points"):
        sample.filter_points(np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="reorder points"):
        sample.reorder_points(np.arange(3, dtype=np.int64))


def test_filter_points_filters_the_segmentation_labels_together() -> None:
    sample = _segmented_sample(num_points=6, num_current_points=4)
    mask = np.array([True, False, True, True, False, True])

    filtered = sample.filter_points(mask)

    assert len(filtered.points) == 4
    assert np.array_equal(filtered.segment.labels, np.array([0, 2, 3, 5]))
    # Three of the four current frame rows survive the mask.
    assert filtered.points.num_current_points == 3


def test_reorder_points_reorders_the_segmentation_labels_together() -> None:
    sample = _segmented_sample(num_points=4)
    indices = np.array([3, 1, 0, 2], dtype=np.int64)

    reordered = sample.reorder_points(indices)

    assert np.array_equal(reordered.segment.labels, np.array([3, 1, 0, 2]))
    assert np.array_equal(reordered.points.features, sample.points.features[indices])
    assert reordered.points.num_current_points is None


def test_point_cloud_pack_selects_columns_in_the_requested_order() -> None:
    cloud = make_point_cloud(num_points=5)

    packed = cloud.pack((PointFeatureName.INTENSITY, PointFeatureName.X))

    assert packed.shape == (5, 2)
    assert np.array_equal(packed[:, 0], cloud.feature(PointFeatureName.INTENSITY))
    assert np.array_equal(packed[:, 1], cloud.feature(PointFeatureName.X))

    with pytest.raises(KeyError, match="has no feature"):
        cloud.pack((PointFeatureName.RING,))


def test_point_cloud_filter_recounts_the_leading_current_block() -> None:
    cloud = make_point_cloud(num_points=6, num_current_points=4)
    mask = np.array([True, False, False, True, True, True])

    filtered = cloud.filter(mask)

    assert len(filtered) == 4
    assert filtered.num_current_points == 2


def test_point_cloud_reorder_forgets_the_current_block() -> None:
    cloud = make_point_cloud(num_points=4, num_current_points=2)

    reordered = cloud.reorder(np.array([2, 0, 3, 1], dtype=np.int64))

    assert reordered.num_current_points is None


def test_point_cloud_rejects_a_current_count_beyond_the_point_count() -> None:
    with pytest.raises(ValidationError, match="outside the point count"):
        make_point_cloud(num_points=3, num_current_points=4)


def test_point_cloud_requires_the_coordinate_columns_first() -> None:
    features = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValidationError, match="first three point features"):
        PointCloud(
            features=features,
            feature_names=(
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.INTENSITY,
                PointFeatureName.Z,
            ),
            num_current_points=2,
        )


def test_point_cloud_rejects_a_feature_name_count_mismatch() -> None:
    features = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValidationError, match="feature names"):
        PointCloud(
            features=features,
            feature_names=(PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z),
            num_current_points=2,
        )
