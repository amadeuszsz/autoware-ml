"""Tests for the segmentation formatting transforms."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.segmentation3d.formatting import PreparePointSegInput
from autoware_ml.types.geometry import PointFeatureName


def _lag_cloud(time_lags: list[float], num_current: int | None) -> PointCloud:
    features = np.zeros((len(time_lags), 5), dtype=np.float32)
    features[:, 0] = np.arange(len(time_lags))
    features[:, 1] = 2.0
    features[:, 2] = 3.0
    features[:, 3] = 0.5
    features[:, 4] = time_lags
    return PointCloud(
        features=features,
        feature_names=(
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.INTENSITY,
            PointFeatureName.TIMESTAMP_DIFFERENCE,
        ),
        num_current_points=num_current,
    )


def _plain_cloud(num_points: int, num_current: int) -> PointCloud:
    features = np.zeros((num_points, 4), dtype=np.float32)
    features[:, 0] = np.arange(num_points)
    return PointCloud(
        features=features,
        feature_names=(
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.INTENSITY,
        ),
        num_current_points=num_current,
    )


def _with_labels(points: PointCloud, labels: list[int]) -> Sample:
    sample = make_sample(points=points)
    return sample.model_copy(
        update={"segment": SegmentationLabels(labels=np.array(labels, dtype=np.int64))}
    )


def test_pads_sweep_points_with_ignore_index() -> None:
    sample = _with_labels(_lag_cloud([0.0, 0.0, 0.1], num_current=2), [7, 8])

    output = PreparePointSegInput(ignore_index=-1)(sample)

    assert np.array_equal(output.segment.labels, np.array([7, 8, -1], dtype=np.int64))
    assert len(output.segment) == len(output.points)
    assert output.points.num_current_points == 2


def test_defaults_to_ignore_without_labels() -> None:
    sample = make_sample(points=_lag_cloud([0.0, 0.1], num_current=1))

    output = PreparePointSegInput(ignore_index=-1)(sample)

    assert np.array_equal(output.segment.labels, np.array([-1, -1], dtype=np.int64))


def test_rejects_label_point_count_mismatch() -> None:
    sample = _with_labels(_lag_cloud([0.0, 0.0], num_current=2), [7, 8, 9])

    with pytest.raises(ValueError, match="one semantic label per current-frame point"):
        PreparePointSegInput(ignore_index=-1)(sample)


def test_rejects_current_frame_outside_leading_block() -> None:
    sample = _with_labels(_lag_cloud([0.0, 0.1, 0.0], num_current=2), [7, 8])

    with pytest.raises(ValueError, match="leading block"):
        PreparePointSegInput(ignore_index=-1)(sample)


def test_supports_a_cloud_without_time_lag() -> None:
    sample = _with_labels(_plain_cloud(2, num_current=2), [7, 8])

    output = PreparePointSegInput(ignore_index=-1)(sample)

    assert np.array_equal(output.segment.labels, np.array([7, 8], dtype=np.int64))


def test_rejects_sweep_points_without_a_time_lag_feature() -> None:
    sample = _with_labels(_plain_cloud(2, num_current=1), [7])

    with pytest.raises(ValueError, match="every point must belong to the current frame"):
        PreparePointSegInput(ignore_index=-1)(sample)


def test_rejects_an_untracked_current_frame_block() -> None:
    sample = make_sample(points=_lag_cloud([0.0, 0.1], num_current=None))

    with pytest.raises(ValueError, match="num_current_points"):
        PreparePointSegInput(ignore_index=-1)(sample)
