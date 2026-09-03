"""Tests for the current frame selection helper."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.transforms.point_cloud.time_lag import current_frame_mask
from autoware_ml.types.geometry import PointFeatureName


def _cloud(num_points: int, num_current: int | None, with_time_lag: bool) -> PointCloud:
    feature_names = [PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z]
    if with_time_lag:
        feature_names.append(PointFeatureName.TIMESTAMP_DIFFERENCE)
    features = np.zeros((num_points, len(feature_names)), dtype=np.float32)
    return PointCloud(
        features=features, feature_names=tuple(feature_names), num_current_points=num_current
    )


def test_current_frame_mask_selects_the_leading_block() -> None:
    assert current_frame_mask(_cloud(3, 2, with_time_lag=True)).tolist() == [True, True, False]


def test_current_frame_mask_does_not_need_the_time_lag_feature() -> None:
    # A multi sweep cloud loaded without a lag column still knows its current frame block
    assert current_frame_mask(_cloud(3, 1, with_time_lag=False)).tolist() == [True, False, False]


def test_current_frame_mask_rejects_an_untracked_block() -> None:
    with pytest.raises(ValueError, match="does not track its current frame block"):
        current_frame_mask(_cloud(3, None, with_time_lag=True))
