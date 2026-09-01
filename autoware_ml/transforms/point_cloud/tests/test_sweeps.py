"""Tests for the multi sweep point cloud loader."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.testing.factories import make_lidar_frame, make_record, make_sample
from autoware_ml.transforms.point_cloud.sweeps import LoadPointsFromMultiSweeps
from autoware_ml.types.geometry import PointFeatureName

USE_FEATURES = ["x", "y", "z", "intensity", "timestamp_difference"]


def _write_cloud(tmp_path, name: str, points: np.ndarray) -> None:
    np.asarray(points, dtype=np.float32).tofile(tmp_path / name)


def _keyframe(tmp_path, points: np.ndarray, timestamp: float = 10.0):
    _write_cloud(tmp_path, "key.bin", points)
    return make_lidar_frame(
        pointcloud_path="key.bin", timestamp_seconds=timestamp, num_features=points.shape[1]
    )


def _sweep_frame(tmp_path, name: str, points: np.ndarray, timestamp: float, sensor_to_sweep=None):
    _write_cloud(tmp_path, name, points)
    return make_lidar_frame(
        frame_id=name,
        keyframe=False,
        pointcloud_path=name,
        timestamp_seconds=timestamp,
        num_features=points.shape[1],
        sensor_to_sweep=sensor_to_sweep,
    )


def _sample(tmp_path, lidar_frames):
    record = make_record(lidar_frames=lidar_frames)
    return make_sample(record=record, data_root=str(tmp_path))


def _aged_sample(tmp_path):
    """A keyframe at 10.0 s with stored sweeps aged 0.1, 0.2, and 0.5 s."""
    frames = [_keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32))]
    aged = (("s1.bin", 10.0, 9.9), ("s2.bin", 20.0, 9.8), ("s3.bin", 30.0, 9.5))
    for name, value, timestamp in aged:
        frames.append(
            _sweep_frame(tmp_path, name, np.full((1, 4), value, dtype=np.float32), timestamp)
        )
    return _sample(tmp_path, frames)


def _load_with(selection: str, window, sweeps_num: int = 2) -> LoadPointsFromMultiSweeps:
    return LoadPointsFromMultiSweeps(
        sweeps_num=sweeps_num,
        use_features=USE_FEATURES,
        sweep_selection=selection,
        time_lag_range=window,
    )


def test_multi_sweeps_stamps_time_lags_and_exposes_the_leading_current_block(tmp_path) -> None:
    frames = [
        _keyframe(tmp_path, np.zeros((2, 4), dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", np.full((3, 4), 5.0, dtype=np.float32), 9.9),
    ]

    output = _load_with("nearest", [0.01, 1.0])(_sample(tmp_path, frames))

    time_lag = output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE)
    assert output.points.features.shape == (5, 5)
    assert output.points.num_current_points == 2
    assert np.all(time_lag[:2] == 0.0)
    assert np.allclose(time_lag[2:], 0.1, atol=1e-6)


def test_multi_sweeps_transforms_sweep_points_into_the_keyframe_frame(tmp_path) -> None:
    sensor_to_sweep = np.eye(4)
    sensor_to_sweep[:3, 3] = [1.0, 2.0, 0.0]
    frames = [
        _keyframe(tmp_path, np.full((1, 4), 9.0, dtype=np.float32)),
        _sweep_frame(
            tmp_path,
            "s.bin",
            np.zeros((1, 4), dtype=np.float32),
            9.9,
            sensor_to_sweep=sensor_to_sweep,
        ),
    ]

    output = _load_with("nearest", [0.01, 1.0])(_sample(tmp_path, frames))

    assert np.allclose(output.points.coord[1], [-1.0, -2.0, 0.0], atol=1e-6)


def test_multi_sweeps_remove_close_removes_an_axis_aligned_box(tmp_path) -> None:
    """The removed region is the box |x|, |y| < close_radius, not a radial circle. The point
    (0.9, 0.9) lies outside the r=1.0 circle but inside the box, so only the box semantics
    remove it."""
    sweep_points = np.array(
        [
            [0.9, 0.9, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0],
            [1.05, 0.0, 0.0, 0.0],
            [0.0, -1.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    frames = [
        _keyframe(tmp_path, np.full((1, 4), 5.0, dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", sweep_points, 9.9),
    ]
    transform = LoadPointsFromMultiSweeps(
        sweeps_num=2,
        use_features=USE_FEATURES,
        sweep_selection="nearest",
        time_lag_range=[0.01, 1.0],
        remove_close=True,
        close_radius=1.0,
    )

    output = transform(_sample(tmp_path, frames))

    coord = output.points.coord
    assert coord.shape == (3, 3)
    assert not np.any(np.all(np.isclose(coord[:, :2], [0.9, 0.9]), axis=1))
    assert np.any(np.all(np.isclose(coord[:, :2], [1.05, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(coord[:, :2], [0.0, -1.2]), axis=1))


def test_nearest_selection_takes_the_most_recent_eligible_sweep(tmp_path) -> None:
    output = _load_with("nearest", [0.05, 0.25])(_aged_sample(tmp_path))

    assert output.points.features.shape[0] == 2
    assert np.allclose(output.points.coord[1], 10.0)


def test_time_lag_range_makes_sweeps_outside_the_window_unavailable(tmp_path) -> None:
    # Only the 0.2 s sweep is eligible, 0.1 s is too recent and 0.5 s too old.
    output = _load_with("nearest", [0.15, 0.25])(_aged_sample(tmp_path))

    assert np.allclose(output.points.coord[1], 20.0)


def test_a_frame_whose_sweeps_are_all_stale_runs_without_them(tmp_path) -> None:
    frames = [
        _keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", np.ones((1, 4), dtype=np.float32), 9.0),
    ]

    output = _load_with("nearest", [0.05, 0.25])(_sample(tmp_path, frames))

    assert output.points.features.shape[0] == 1
    assert output.points.num_current_points == 1


def test_random_selection_samples_only_among_the_eligible_sweeps(tmp_path, monkeypatch) -> None:
    calls = {}

    def fake_choice(num_entries, size, replace):
        calls["args"] = (num_entries, size, replace)
        return np.array([1])

    monkeypatch.setattr("autoware_ml.transforms.point_cloud.sweeps.np.random.choice", fake_choice)
    # The window admits the 0.1 s and 0.2 s sweeps but not the 0.5 s one.
    output = _load_with("random", [0.05, 0.25])(_aged_sample(tmp_path))

    assert calls["args"] == (2, 1, False)
    assert np.allclose(output.points.coord[1], 20.0)


def test_random_selection_keeps_the_appended_sweeps_ordered_by_recency(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "autoware_ml.transforms.point_cloud.sweeps.np.random.choice",
        lambda num_entries, size, replace: np.array([2, 0]),
    )

    output = _load_with("random", [0.05, 0.6], sweeps_num=3)(_aged_sample(tmp_path))

    # Sampled the 0.5 s and 0.1 s sweeps, appended newest first.
    assert np.allclose(output.points.coord[1], 10.0)
    assert np.allclose(output.points.coord[2], 30.0)


def test_multi_sweeps_pads_empty_sweeps_with_the_minimum_time_lag(tmp_path) -> None:
    """Scene first padding stands in for sweeps. The copies carry the minimum admissible lag
    so current frame selections never count them."""
    frames = [_keyframe(tmp_path, np.zeros((2, 4), dtype=np.float32))]
    transform = LoadPointsFromMultiSweeps(
        sweeps_num=3,
        use_features=USE_FEATURES,
        sweep_selection="nearest",
        time_lag_range=[0.05, 0.25],
        pad_empty_sweeps=True,
    )

    output = transform(_sample(tmp_path, frames))

    time_lag = output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE)
    assert output.points.features.shape == (6, 5)
    assert output.points.num_current_points == 2
    assert np.all(time_lag[:2] == 0.0)
    assert np.allclose(time_lag[2:], 0.05)


def test_multi_sweeps_rejects_a_zero_minimum_time_lag() -> None:
    """The current frame owns lag 0, so a window admitting zero lag sweeps is a config error,
    such a sweep would be indistinguishable from the current frame downstream."""
    with pytest.raises(ValueError, match="0 < min time lag"):
        _load_with("nearest", [0.0, 1.0])


def test_multi_sweeps_rejects_an_unknown_selection_or_window() -> None:
    with pytest.raises(ValueError, match="sweep_selection"):
        _load_with("newest", [0.05, 0.25])
    with pytest.raises(ValueError, match="min time lag < max time lag"):
        _load_with("nearest", [0.25, 0.05])
    with pytest.raises(ValueError, match=r"\[min, max\]"):
        _load_with("nearest", [0.25])


def test_single_frame_load_packs_the_selected_features_and_ignores_stored_sweeps(
    tmp_path,
) -> None:
    stored = np.array([[1.0, 2.0, 3.0, 40.0, 7.0], [4.0, 5.0, 6.0, 50.0, 8.0]], dtype=np.float32)
    frames = [
        _keyframe(tmp_path, stored),
        _sweep_frame(tmp_path, "s.bin", np.full((3, 5), 5.0, dtype=np.float32), 9.9),
    ]

    output = LoadPointsFromMultiSweeps(sweeps_num=1, use_features=["x", "y", "z", "intensity"])(
        _sample(tmp_path, frames)
    )

    assert output.points.features.shape == (2, 4)
    assert np.allclose(output.points.features, stored[:, :4])
    assert output.points.num_current_points == 2


def test_single_frame_load_stamps_a_zero_time_lag_column(tmp_path) -> None:
    frames = [_keyframe(tmp_path, np.ones((3, 4), dtype=np.float32))]

    output = LoadPointsFromMultiSweeps(sweeps_num=1, use_features=USE_FEATURES)(
        _sample(tmp_path, frames)
    )

    assert np.all(output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE) == 0.0)


def test_single_frame_load_rejects_sweep_arguments() -> None:
    with pytest.raises(ValueError, match="single frame"):
        LoadPointsFromMultiSweeps(
            sweeps_num=1, use_features=USE_FEATURES, sweep_selection="nearest"
        )
    with pytest.raises(ValueError, match="single frame"):
        LoadPointsFromMultiSweeps(
            sweeps_num=1, use_features=USE_FEATURES, time_lag_range=[0.05, 0.25]
        )
    with pytest.raises(ValueError, match="single frame"):
        LoadPointsFromMultiSweeps(sweeps_num=1, use_features=USE_FEATURES, remove_close=True)


def test_multi_sweep_load_requires_the_sweep_arguments() -> None:
    with pytest.raises(ValueError, match="sweep_selection"):
        LoadPointsFromMultiSweeps(sweeps_num=2, use_features=USE_FEATURES)
    with pytest.raises(ValueError, match=r"\[min, max\]"):
        LoadPointsFromMultiSweeps(
            sweeps_num=2, use_features=USE_FEATURES, sweep_selection="nearest"
        )


def test_sweeps_num_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LoadPointsFromMultiSweeps(sweeps_num=0, use_features=USE_FEATURES)
