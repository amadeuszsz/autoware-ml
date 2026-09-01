"""Tests for the point cloud cropping transforms."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.point_cloud.crop import CropBoxInner, PointsRangeFilter
from autoware_ml.types.geometry import PointFeatureName

FEATURE_NAMES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
)


def _segmented_sample(coord: list[list[float]], num_current_points: int | None = None):
    coord = np.asarray(coord, dtype=np.float32)
    features = np.concatenate(
        [coord, np.arange(coord.shape[0], dtype=np.float32)[:, None]], axis=1
    )
    points = PointCloud(
        features=features,
        feature_names=FEATURE_NAMES,
        num_current_points=(
            num_current_points if num_current_points is not None else coord.shape[0]
        ),
    )
    sample = make_sample(points=points)
    return sample.model_copy(
        update={"segment": SegmentationLabels(labels=np.arange(coord.shape[0], dtype=np.int64))}
    )


def test_points_range_filter_keeps_points_and_segment_aligned() -> None:
    sample = _segmented_sample(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [5.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    output = PointsRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0])(sample)

    expected = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    assert np.allclose(output.points.coord, expected)
    assert output.points.feature(PointFeatureName.INTENSITY).tolist() == [0.0, 1.0, 6.0]
    assert output.segment.labels.tolist() == [0, 1, 6]


def test_points_range_filter_excludes_the_upper_bound() -> None:
    sample = _segmented_sample(
        [
            [0.0, 0.0, 0.0],
            [1.999, 1.999, 1.999],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )

    output = PointsRangeFilter(point_cloud_range=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0])(sample)

    # Points on the upper bound are excluded so voxel indices stay inside the grid.
    grid_coord = np.floor(output.points.coord / 1.0).astype(np.int64)
    assert output.points.coord.shape == (2, 3)
    assert np.all(grid_coord >= 0)
    assert np.all(grid_coord < 2)


def test_points_range_filter_recounts_the_current_frame_block() -> None:
    sample = _segmented_sample(
        [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.0, 1.0, 1.0]], num_current_points=2
    )

    output = PointsRangeFilter(point_cloud_range=[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])(sample)

    assert output.points.num_current_points == 1


def test_points_range_filter_rejects_a_malformed_range() -> None:
    with pytest.raises(ValueError, match="6 elements"):
        PointsRangeFilter(point_cloud_range=[0.0, 0.0, 0.0])


def test_crop_box_inner_removes_the_points_inside_the_box() -> None:
    sample = _segmented_sample(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    output = CropBoxInner(crop_box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])(sample)

    expected = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [-2.0, 0.0, 0.0]], dtype=np.float32
    )
    assert np.allclose(output.points.coord, expected)
    assert output.segment.labels.tolist() == [2, 3, 4, 5]
