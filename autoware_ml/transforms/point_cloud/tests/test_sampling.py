"""Tests for the point cloud sampling transforms."""

from __future__ import annotations

import numpy as np

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.point_cloud.sampling import PointShuffle, RandomDropout
from autoware_ml.types.geometry import PointFeatureName

XYZ = (PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z)


def _indexed_sample(num_points: int, num_current_points: int | None = None):
    """A sample whose x coordinate and segmentation label both equal the point index."""
    features = np.zeros((num_points, 3), dtype=np.float32)
    features[:, 0] = np.arange(num_points, dtype=np.float32)
    points = PointCloud(
        features=features,
        feature_names=XYZ,
        num_current_points=(num_current_points if num_current_points is not None else num_points),
    )
    sample = make_sample(points=points)
    return sample.model_copy(
        update={"segment": SegmentationLabels(labels=np.arange(num_points, dtype=np.int64))}
    )


def test_point_shuffle_keeps_points_and_segment_aligned() -> None:
    sample = _indexed_sample(16)

    np.random.seed(0)
    output = PointShuffle()(sample)

    assert sorted(output.segment.labels.tolist()) == list(range(16))
    assert np.array_equal(output.points.coord[:, 0].astype(np.int64), output.segment.labels)
    assert not np.array_equal(output.segment.labels, np.arange(16))


def test_point_shuffle_invalidates_the_current_frame_block() -> None:
    sample = _indexed_sample(8, num_current_points=4)

    output = PointShuffle()(sample)

    assert output.points.num_current_points is None


def test_random_dropout_keeps_points_and_segment_aligned() -> None:
    sample = _indexed_sample(4)

    np.random.seed(0)
    output = RandomDropout(dropout_ratio=0.5, p=1.0)(sample)

    assert len(output.points) == 2
    assert len(output.segment) == 2
    assert np.array_equal(output.points.coord[:, 0].astype(np.int64), output.segment.labels)


def test_random_dropout_recounts_the_current_frame_block() -> None:
    sample = _indexed_sample(8, num_current_points=4)

    np.random.seed(0)
    output = RandomDropout(dropout_ratio=0.5, p=1.0)(sample)

    kept_current = int(np.sum(output.points.coord[:, 0] < 4))
    assert output.points.num_current_points == kept_current


def test_random_dropout_respects_application_probability() -> None:
    sample = _indexed_sample(4)

    output = RandomDropout(dropout_ratio=0.5, p=0.0)(sample)

    assert len(output.points) == 4
    assert len(output.segment) == 4
