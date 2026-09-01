"""Tests for the point cloud loading helpers."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.testing.factories import make_lidar_frame, make_record, make_sample
from autoware_ml.transforms.point_cloud.loading import (
    keyframe_lidar_frame,
    resolve_frame_path,
    select_raw_features,
)
from autoware_ml.types.geometry import PointFeatureName


def test_select_raw_features_rejects_features_the_file_does_not_store() -> None:
    stored = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="ring"):
        select_raw_features(stored, [PointFeatureName.X, PointFeatureName.RING])


def test_select_raw_features_rejects_derived_features() -> None:
    stored = np.zeros((1, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="timestamp_difference"):
        select_raw_features(stored, [PointFeatureName.TIMESTAMP_DIFFERENCE])


def test_resolve_frame_path_rejects_absolute_record_paths() -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_frame_path("/data", "/absolute/frame.bin")


def test_keyframe_lidar_frame_requires_a_leading_keyframe() -> None:
    record = make_record(lidar_frames=[make_lidar_frame(keyframe=False)])
    sample = make_sample(record=record)

    with pytest.raises(ValueError, match="keyframe"):
        keyframe_lidar_frame(sample)
