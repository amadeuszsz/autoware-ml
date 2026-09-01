"""Tests for the point cloud input preparation transform."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.testing.factories import make_point_cloud, make_sample
from autoware_ml.transforms.point_cloud.formatting import PreparePointCloudInput
from autoware_ml.types.geometry import PointFeatureName


def test_prepare_point_cloud_input_normalizes_the_intensity() -> None:
    points = make_point_cloud(num_points=10, with_time_lag=True)
    sample = make_sample(points=points)

    output = PreparePointCloudInput(require_time_lag=True)(sample)

    expected = points.feature(PointFeatureName.INTENSITY) / 255.0
    assert np.allclose(output.points.feature(PointFeatureName.INTENSITY), expected)
    assert np.allclose(output.points.coord, points.coord)


def test_prepare_point_cloud_input_requires_the_declared_time_lag() -> None:
    sample = make_sample(points=make_point_cloud(with_time_lag=False))

    with pytest.raises(ValueError, match="requires the timestamp_difference"):
        PreparePointCloudInput(require_time_lag=True)(sample)


def test_prepare_point_cloud_input_rejects_an_undeclared_time_lag() -> None:
    sample = make_sample(points=make_point_cloud(with_time_lag=True))

    with pytest.raises(ValueError, match="configured without a time lag"):
        PreparePointCloudInput(require_time_lag=False)(sample)
