"""Tests for the lidar only geometric augmentations."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.samples.boxes3d import Boxes3D
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.point_cloud.geometry import (
    GlobalRotScaleTrans,
    RandomFlip3D,
    RandomRotateTargetAngle,
)
from autoware_ml.types.geometry import PointFeatureName

XYZ = (PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z)


def _sample(coord: list[list[float]], box_params: list[list[float]] | None = None):
    features = np.asarray(coord, dtype=np.float32)
    points = PointCloud(
        features=features, feature_names=XYZ, num_current_points=features.shape[0]
    )
    boxes = None
    if box_params is not None:
        params = np.asarray(box_params, dtype=np.float32)
        boxes = Boxes3D(
            params=params,
            labels=np.zeros(params.shape[0], dtype=np.int64),
            names=tuple(["car"] * params.shape[0]),
            num_lidar_points=np.ones(params.shape[0], dtype=np.int64),
        )
    return make_sample(points=points, boxes=boxes)


def test_random_rotate_target_angle_rotates_boxes_with_points() -> None:
    sample = _sample([[2.0, 0.0, 1.0]], [[2.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.1, 3.0, 0.0]])

    output = RandomRotateTargetAngle(angle=[0.5], center=[0.0, 0.0, 0.0], p=1.0)(sample)

    box = output.boxes.params[0]
    assert np.allclose(output.points.coord, [[0.0, 2.0, 1.0]], atol=1e-6)
    assert np.allclose(box[:3], [0.0, 2.0, 1.0], atol=1e-6)
    assert np.allclose(box[3:6], [4.0, 2.0, 1.5])
    assert np.isclose(box[6], 0.1 + 0.5 * np.pi)
    assert np.allclose(box[7:9], [0.0, 3.0], atol=1e-6)


def test_random_rotate_target_angle_rejects_boxes_off_z_axis() -> None:
    sample = _sample([[0.0, 0.0, 0.0]], [[0.0] * 9])

    with pytest.raises(ValueError, match="axis='z'"):
        RandomRotateTargetAngle(angle=[0.5], axis="x", p=1.0)(sample)


def test_random_rotate_target_angle_respects_probability(monkeypatch) -> None:
    sample = _sample([[1.0, 0.0, 0.0]])
    monkeypatch.setattr(np.random, "rand", lambda: 0.75)

    output = RandomRotateTargetAngle(angle=(0.5,), center=[0.0, 0.0, 0.0], p=0.5)(sample)

    assert np.allclose(output.points.coord, [[1.0, 0.0, 0.0]])


def test_random_flip3d_updates_boxes_with_points() -> None:
    sample = _sample([[1.0, 2.0, 0.0]], [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.25, 1.5, -0.5]])

    output = RandomFlip3D(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=0.0)(sample)

    assert np.allclose(output.points.coord[0, :2], [1.0, -2.0])
    assert np.allclose(output.boxes.params[0, [1, 6, 8]], [-2.0, -0.25, 0.5])


def test_random_flip3d_uses_configured_probability_per_axis(monkeypatch) -> None:
    sample = _sample([[1.0, 2.0, 0.0]])
    calls = iter([0.2, 0.8])
    monkeypatch.setattr(np.random, "rand", lambda: next(calls))

    # The horizontal draw returns 0.2 < 0.5 so the y axis flips, the vertical draw returns
    # 0.8 >= 0.5 so the x axis stays.
    output = RandomFlip3D(flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5)(sample)

    assert np.allclose(output.points.coord, [[1.0, -2.0, 0.0]])


def test_global_rot_scale_trans_updates_boxes() -> None:
    sample = _sample([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0, 1.0, 0.0]])

    np.random.seed(0)
    output = GlobalRotScaleTrans(
        rot_range=[0.1, 0.1],
        scale_ratio_range=[2.0, 2.0],
        translation_std=[0.0, 0.0, 0.0],
    )(sample)

    assert np.allclose(output.boxes.params[0, 3:6], [8.0, 4.0, 2.0])
    assert np.allclose(output.boxes.params[0, 6], 0.1)


def test_global_rot_scale_trans_scales_velocity_with_space() -> None:
    sample = _sample([[1.0, 2.0, 0.5]], [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.3, 1.5, -0.5]])

    output = GlobalRotScaleTrans(rot_range=[0.0, 0.0], scale_ratio_range=[0.5, 0.5])(sample)

    box = output.boxes.params[0]
    assert np.allclose(box[:3], [0.5, 1.0, 0.0])
    assert np.allclose(box[3:6], [2.0, 1.0, 0.5])
    assert np.isclose(box[6], 0.3)
    assert np.allclose(box[7:9], [0.75, -0.25])


def test_geometry_transforms_require_a_loaded_point_cloud() -> None:
    sample = make_sample()

    with pytest.raises(ValueError, match="points"):
        GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.0, 1.0])(sample)
