"""Tests for the per-frame evaluation metadata helpers."""

from __future__ import annotations

import pytest

from autoware_ml.datamodule.common.frame_meta import scene_dir_fragment


def test_scene_dir_fragment() -> None:
    path = "db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path) == "db_j6gen2_v2/13cabeac-a81b/0"


def test_scene_dir_fragment_rejects_short_paths() -> None:
    with pytest.raises(ValueError, match="scene directory"):
        scene_dir_fragment("no_scene.bin")
