"""Tests for the multi sweep point cloud loader."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from autoware_ml.testing.factories import make_lidar_frame, make_record, make_sample
from autoware_ml.transforms.point_cloud.sweeps import (
    LoadPointsFromMultiSweeps,
    SweepSelection,
    SweepWindow,
)
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
    """A keyframe at 10.0 s with stored past sweeps aged 0.1, 0.2, and 0.5 s."""
    frames = [_keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32))]
    aged = (("s1.bin", 10.0, 9.9), ("s2.bin", 20.0, 9.8), ("s3.bin", 30.0, 9.5))
    for name, value, timestamp in aged:
        frames.append(
            _sweep_frame(tmp_path, name, np.full((1, 4), value, dtype=np.float32), timestamp)
        )
    return _sample(tmp_path, frames)


def _two_sided_sample(tmp_path):
    """A keyframe at 10.0 s with past sweeps 0.1 and 0.2 s old and future sweeps 0.1 and
    0.2 s ahead, stored past first as the generators write them."""
    frames = [_keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32))]
    stored = (
        ("p1.bin", 10.0, 9.9),
        ("p2.bin", 20.0, 9.8),
        ("f1.bin", -10.0, 10.1),
        ("f2.bin", -20.0, 10.2),
    )
    for name, value, timestamp in stored:
        frames.append(
            _sweep_frame(tmp_path, name, np.full((1, 4), value, dtype=np.float32), timestamp)
        )
    return _sample(tmp_path, frames)


def _window(selection: str, time_lag_range, num: int = 1) -> SweepWindow:
    return SweepWindow(num=num, time_lag_range=time_lag_range, selection=selection)


def _load_past(selection: str, time_lag_range, num: int = 1) -> LoadPointsFromMultiSweeps:
    return LoadPointsFromMultiSweeps(
        use_features=USE_FEATURES, past=_window(selection, time_lag_range, num)
    )


def test_past_sweeps_stamp_positive_time_lags_and_expose_the_leading_current_block(
    tmp_path,
) -> None:
    frames = [
        _keyframe(tmp_path, np.zeros((2, 4), dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", np.full((3, 4), 5.0, dtype=np.float32), 9.9),
    ]

    output = _load_past("nearest", [0.01, 1.0])(_sample(tmp_path, frames))

    time_lag = output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE)
    assert output.points.features.shape == (5, 5)
    assert output.points.num_current_points == 2
    assert np.all(time_lag[:2] == 0.0)
    assert np.allclose(time_lag[2:], 0.1, atol=1e-6)


def test_future_sweeps_stamp_negative_time_lags(tmp_path) -> None:
    frames = [
        _keyframe(tmp_path, np.zeros((2, 4), dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", np.full((3, 4), 5.0, dtype=np.float32), 10.1),
    ]
    transform = LoadPointsFromMultiSweeps(
        use_features=USE_FEATURES, future=_window("nearest", [0.01, 1.0])
    )

    output = transform(_sample(tmp_path, frames))

    time_lag = output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE)
    assert output.points.features.shape == (5, 5)
    assert output.points.num_current_points == 2
    assert np.all(time_lag[:2] == 0.0)
    assert np.allclose(time_lag[2:], -0.1, atol=1e-6)


def test_each_window_selects_on_its_own_side_past_block_before_future_block(tmp_path) -> None:
    transform = LoadPointsFromMultiSweeps(
        use_features=USE_FEATURES,
        past=_window("nearest", [0.05, 0.25], num=2),
        future=_window("nearest", [0.05, 0.25], num=2),
    )

    output = transform(_two_sided_sample(tmp_path))

    assert np.allclose(output.points.coord[1:, 0], [10.0, 20.0, -10.0, -20.0])
    time_lag = output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE)
    assert np.allclose(time_lag, [0.0, 0.1, 0.2, -0.1, -0.2], atol=1e-6)


def test_a_window_ignores_the_stored_frames_of_the_other_side(tmp_path) -> None:
    output = LoadPointsFromMultiSweeps(
        use_features=USE_FEATURES, future=_window("nearest", [0.05, 0.6], num=3)
    )(_two_sided_sample(tmp_path))

    assert np.allclose(output.points.coord[1:, 0], [-10.0, -20.0])


def test_a_side_without_stored_frames_contributes_nothing(tmp_path) -> None:
    """The last sample of a scene stores past frames only, so a future window appends no
    points there and nothing stands in for them."""
    transform = LoadPointsFromMultiSweeps(
        use_features=USE_FEATURES,
        past=_window("nearest", [0.05, 0.25]),
        future=_window("nearest", [0.05, 0.25], num=3),
    )

    output = transform(_aged_sample(tmp_path))

    assert output.points.features.shape[0] == 2
    assert np.allclose(output.points.coord[1], 10.0)


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

    output = _load_past("nearest", [0.01, 1.0])(_sample(tmp_path, frames))

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
        use_features=USE_FEATURES,
        past=_window("nearest", [0.01, 1.0]),
        remove_close=True,
        close_radius=1.0,
    )

    output = transform(_sample(tmp_path, frames))

    coord = output.points.coord
    assert coord.shape == (3, 3)
    assert not np.any(np.all(np.isclose(coord[:, :2], [0.9, 0.9]), axis=1))
    assert np.any(np.all(np.isclose(coord[:, :2], [1.05, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(coord[:, :2], [0.0, -1.2]), axis=1))


def test_a_sweep_flagged_as_keyframe_is_still_appended_as_a_sweep(tmp_path) -> None:
    # Corpora annotated at every frame flag the stored sweeps as keyframes of other samples
    _write_cloud(tmp_path, "s1.bin", np.full((1, 4), 10.0, dtype=np.float32))
    frames = [
        _keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32)),
        make_lidar_frame(
            frame_id="s1.bin",
            keyframe=True,
            pointcloud_path="s1.bin",
            timestamp_seconds=9.9,
            num_features=4,
        ),
    ]

    output = _load_past("nearest", [0.05, 0.25])(_sample(tmp_path, frames))

    assert len(output.points) == 2
    assert output.points.num_current_points == 1
    assert np.isclose(output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE)[1], 0.1)


def test_a_stored_sweep_at_the_current_timestamp_is_rejected(tmp_path) -> None:
    """Every consumer identifies the current frame by lag 0, so a stored sweep sharing the
    timestamp would be scored and labelled as current frame points."""
    frames = [
        _keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", np.ones((1, 4), dtype=np.float32), 10.0),
    ]

    with pytest.raises(ValueError, match="timestamp of its current frame"):
        _load_past("nearest", [0.05, 0.25])(_sample(tmp_path, frames))


def test_nearest_selection_takes_the_closest_eligible_sweep(tmp_path) -> None:
    output = _load_past("nearest", [0.05, 0.25])(_aged_sample(tmp_path))

    assert output.points.features.shape[0] == 2
    assert np.allclose(output.points.coord[1], 10.0)


def test_time_lag_range_makes_sweeps_outside_the_window_unavailable(tmp_path) -> None:
    # Only the 0.2 s sweep is eligible, 0.1 s is too recent and 0.5 s too old.
    output = _load_past("nearest", [0.15, 0.25])(_aged_sample(tmp_path))

    assert np.allclose(output.points.coord[1], 20.0)


def test_a_frame_whose_sweeps_are_all_stale_runs_without_them(tmp_path) -> None:
    frames = [
        _keyframe(tmp_path, np.zeros((1, 4), dtype=np.float32)),
        _sweep_frame(tmp_path, "s.bin", np.ones((1, 4), dtype=np.float32), 9.0),
    ]

    output = _load_past("nearest", [0.05, 0.25])(_sample(tmp_path, frames))

    assert output.points.features.shape[0] == 1
    assert output.points.num_current_points == 1


def test_random_selection_samples_only_among_the_eligible_sweeps(tmp_path, monkeypatch) -> None:
    calls = {}

    def fake_choice(num_entries, size, replace):
        calls["args"] = (num_entries, size, replace)
        return np.array([1])

    monkeypatch.setattr("autoware_ml.transforms.point_cloud.sweeps.np.random.choice", fake_choice)
    # The window admits the 0.1 s and 0.2 s sweeps but not the 0.5 s one.
    output = _load_past("random", [0.05, 0.25])(_aged_sample(tmp_path))

    assert calls["args"] == (2, 1, False)
    assert np.allclose(output.points.coord[1], 20.0)


def test_random_selection_keeps_the_appended_sweeps_ordered_by_distance(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "autoware_ml.transforms.point_cloud.sweeps.np.random.choice",
        lambda num_entries, size, replace: np.array([2, 0]),
    )

    output = _load_past("random", [0.05, 0.6], num=2)(_aged_sample(tmp_path))

    # Sampled the 0.5 s and 0.1 s sweeps, appended nearest first.
    assert np.allclose(output.points.coord[1], 10.0)
    assert np.allclose(output.points.coord[2], 30.0)


def test_a_window_coerces_its_configured_values() -> None:
    window = _window("random", [0.05, 0.25], num=3)

    assert window.selection is SweepSelection.RANDOM
    assert window.time_lag_range == (0.05, 0.25)
    assert (window.min_time_lag, window.max_time_lag) == (0.05, 0.25)


def test_a_window_rejects_a_zero_minimum_time_lag() -> None:
    """The current frame owns lag 0, so a window admitting zero lag sweeps is a config error,
    such a sweep would be indistinguishable from the current frame downstream."""
    with pytest.raises(ValidationError, match="0 < min time lag"):
        _window("nearest", [0.0, 1.0])


def test_a_window_rejects_an_unknown_selection_or_malformed_range() -> None:
    with pytest.raises(ValidationError, match="SweepSelection"):
        _window("newest", [0.05, 0.25])
    with pytest.raises(ValidationError, match="min time lag < max time lag"):
        _window("nearest", [0.25, 0.05])
    with pytest.raises(ValidationError, match="time_lag_range"):
        _window("nearest", [0.25])
    with pytest.raises(ValidationError, match="num"):
        _window("nearest", [0.05, 0.25], num=0)


def test_single_frame_load_packs_the_selected_features_and_ignores_stored_sweeps(
    tmp_path,
) -> None:
    stored = np.array([[1.0, 2.0, 3.0, 40.0, 7.0], [4.0, 5.0, 6.0, 50.0, 8.0]], dtype=np.float32)
    frames = [
        _keyframe(tmp_path, stored),
        _sweep_frame(tmp_path, "s.bin", np.full((3, 5), 5.0, dtype=np.float32), 9.9),
        _sweep_frame(tmp_path, "f.bin", np.full((3, 5), 6.0, dtype=np.float32), 10.1),
    ]

    output = LoadPointsFromMultiSweeps(use_features=["x", "y", "z", "intensity"])(
        _sample(tmp_path, frames)
    )

    assert output.points.features.shape == (2, 4)
    assert np.allclose(output.points.features, stored[:, :4])
    assert output.points.num_current_points == 2


def test_single_frame_load_stamps_a_zero_time_lag_column(tmp_path) -> None:
    frames = [_keyframe(tmp_path, np.ones((3, 4), dtype=np.float32))]

    output = LoadPointsFromMultiSweeps(use_features=USE_FEATURES)(_sample(tmp_path, frames))

    assert np.all(output.points.feature(PointFeatureName.TIMESTAMP_DIFFERENCE) == 0.0)


def test_single_frame_load_rejects_sweep_arguments() -> None:
    with pytest.raises(ValueError, match="single frame"):
        LoadPointsFromMultiSweeps(use_features=USE_FEATURES, remove_close=True)


def test_a_window_must_be_built_before_it_reaches_the_loader() -> None:
    with pytest.raises(TypeError, match="past must be a SweepWindow"):
        LoadPointsFromMultiSweeps(
            use_features=USE_FEATURES,
            past={"num": 1, "time_lag_range": [0.05, 0.25], "selection": "nearest"},
        )
