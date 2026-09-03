"""Tests for the segmentation range view transforms."""

from __future__ import annotations

import numpy as np

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.segmentation3d.range_view import RangeInterpolation
from autoware_ml.types.geometry import PointFeatureName


def _range_sample(features: list[list[float]], labels: list[int] | None) -> Sample:
    points = PointCloud(
        features=np.array(features, dtype=np.float32),
        feature_names=(
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.INTENSITY,
        ),
        num_current_points=len(features),
    )
    sample = make_sample(points=points)
    if labels is None:
        return sample
    return sample.replace(segment=SegmentationLabels(labels=np.array(labels, dtype=np.int64)))


def test_adds_midpoint_and_boundary_label() -> None:
    sample = _range_sample([[1.0, 1.0, 0.0, 0.5], [-1.0, -1.0, 0.0, 1.5]], [3, 7])

    output = RangeInterpolation(height=1, width=4, fov_up=10.0, fov_down=-10.0, ignore_index=255)(
        sample
    )

    assert len(output.points) == 3
    assert len(output.segment) == 3
    assert output.points.num_current_points == 2
    assert output.segment.labels[-1] == 255
    assert np.allclose(output.points.features[-1], np.array([0.0, 0.0, 0.0, 1.0]))


def test_no_interpolatable_points_leaves_sample_unchanged() -> None:
    sample = _range_sample([[1.0, 0.0, 0.0, 0.5]], [3])

    output = RangeInterpolation(height=4, width=8, fov_up=10.0, fov_down=-10.0, ignore_index=255)(
        sample
    )

    assert np.array_equal(output.points.features, sample.points.features)
    assert np.array_equal(output.segment.labels, sample.segment.labels)
    assert output.points.num_current_points == 1


def test_extends_points_without_labels() -> None:
    sample = _range_sample([[1.0, 1.0, 0.0, 0.5], [-1.0, -1.0, 0.0, 1.5]], None)

    output = RangeInterpolation(height=1, width=4, fov_up=10.0, fov_down=-10.0, ignore_index=255)(
        sample
    )

    assert len(output.points) == 3
    assert output.segment is None
