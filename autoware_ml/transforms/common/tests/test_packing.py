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

"""Tests for the point feature packing transforms."""

import numpy as np
import pytest

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.common.packing import BuildPointFeatures
from autoware_ml.types.geometry import PointFeatureName


def _make_sample() -> Sample:
    features = np.arange(30, dtype=np.float32).reshape(5, 6)
    point_cloud = PointCloud(
        features=features,
        feature_names=(
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.INTENSITY,
            PointFeatureName.RING,
            PointFeatureName.TIMESTAMP_DIFFERENCE,
        ),
        num_current_points=3,
    )
    return make_sample(points=point_cloud)


def test_build_point_features_packs_the_requested_columns_in_order() -> None:
    sample = _make_sample()
    transform = BuildPointFeatures(
        feature_names=["x", "y", "z", "intensity", "timestamp_difference"]
    )

    output = transform(sample)

    assert output.points.feature_names == (
        PointFeatureName.X,
        PointFeatureName.Y,
        PointFeatureName.Z,
        PointFeatureName.INTENSITY,
        PointFeatureName.TIMESTAMP_DIFFERENCE,
    )
    expected = sample.points.features[:, [0, 1, 2, 3, 5]]
    assert np.array_equal(output.points.features, expected)
    assert output.points.num_current_points == 3


def test_build_point_features_missing_feature_raises() -> None:
    sample = _make_sample()
    sample = sample.replace(
        points=PointCloud(
            features=sample.points.features[:, :4],
            feature_names=(
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ),
            num_current_points=3,
        )
    )
    transform = BuildPointFeatures(feature_names=["x", "y", "z", "timestamp_difference"])

    with pytest.raises(KeyError, match="timestamp_difference"):
        transform(sample)


def test_build_point_features_requires_points() -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="points"):
        BuildPointFeatures(feature_names=["x", "y", "z"])(sample)
