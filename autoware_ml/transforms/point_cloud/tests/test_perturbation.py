"""Tests for the point cloud perturbation transforms."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.point_cloud.perturbation import RandomJitter, RandomStrengthJitter
from autoware_ml.types.geometry import PointFeatureName

FEATURE_NAMES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
)


def _sample_with_intensity(intensity: np.ndarray):
    features = np.zeros((intensity.shape[0], 4), dtype=np.float32)
    features[:, 3] = intensity
    points = PointCloud(
        features=features, feature_names=FEATURE_NAMES, num_current_points=intensity.shape[0]
    )
    return make_sample(points=points)


def test_random_jitter_perturbs_only_the_coordinates_within_the_clip() -> None:
    sample = _sample_with_intensity(np.linspace(0.0, 1.0, 8, dtype=np.float32))

    np.random.seed(0)
    output = RandomJitter(sigma=0.1, clip=0.05)(sample)

    delta = output.points.coord - sample.points.coord
    assert np.all(np.abs(delta) <= 0.05 + 1e-7)
    assert np.any(delta != 0.0)
    assert np.array_equal(
        output.points.feature(PointFeatureName.INTENSITY),
        sample.points.feature(PointFeatureName.INTENSITY),
    )


def test_random_strength_jitter_stays_normalized_and_monotonic() -> None:
    sample = _sample_with_intensity(np.linspace(0.0, 1.0, 5, dtype=np.float32))

    np.random.seed(0)
    output = RandomStrengthJitter(
        gamma_range=[0.8, 1.25], scale_range=[0.9, 1.1], shift_range=[-0.02, 0.02]
    )(sample)

    intensity = output.points.feature(PointFeatureName.INTENSITY)
    assert intensity.dtype == np.float32
    assert intensity.min() >= 0.0
    assert intensity.max() <= 1.0
    assert np.all(np.diff(intensity) >= 0.0)
    assert np.array_equal(output.points.coord, sample.points.coord)


def test_random_strength_jitter_requires_the_intensity_feature() -> None:
    points = PointCloud(
        features=np.zeros((2, 3), dtype=np.float32),
        feature_names=(PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z),
        num_current_points=2,
    )
    transform = RandomStrengthJitter(
        gamma_range=[0.8, 1.25], scale_range=[0.9, 1.1], shift_range=[-0.02, 0.02]
    )

    with pytest.raises(KeyError, match="intensity"):
        transform(make_sample(points=points))


def test_random_strength_jitter_rejects_malformed_ranges() -> None:
    with pytest.raises(ValueError, match="gamma_range"):
        RandomStrengthJitter(gamma_range=[0.0, 1.0], scale_range=[1.0, 1.0], shift_range=[0.0, 0.0])
    with pytest.raises(ValueError, match="scale_range"):
        RandomStrengthJitter(gamma_range=[1.0, 1.0], scale_range=[1.1, 0.9], shift_range=[0.0, 0.0])
