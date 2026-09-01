"""Tests for the current frame selection helper."""

from __future__ import annotations

import numpy as np

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.transforms.point_cloud.time_lag import current_frame_mask
from autoware_ml.types.geometry import PointFeatureName


def test_current_frame_mask_is_none_without_the_time_lag_feature() -> None:
    points = PointCloud(
        features=np.zeros((3, 3), dtype=np.float32),
        feature_names=(PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z),
        num_current_points=3,
    )

    assert current_frame_mask(points) is None


def test_current_frame_mask_selects_the_zero_lag_points() -> None:
    features = np.zeros((3, 4), dtype=np.float32)
    features[2, 3] = 0.1
    points = PointCloud(
        features=features,
        feature_names=(
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.TIMESTAMP_DIFFERENCE,
        ),
        num_current_points=2,
    )

    assert current_frame_mask(points).tolist() == [True, True, False]
