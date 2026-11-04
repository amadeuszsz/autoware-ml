# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the map elements the online map targets read around every record."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.datamodule.common.motion import record_poses
from autoware_ml.datamodule.common.online_map import OnlineMapTargets
from autoware_ml.geometry.tests.vector_map_fixtures import CLASSES, write_map

SCENE = "db_test/scene/0"


def records(x: float, yaw: float) -> pl.DataFrame:
    """One record of the scene whose lidar sits at map (x, 0), heading yaw."""
    pose = np.eye(4, dtype=np.float32)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[0, 3] = x
    frame = {
        LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: pose,
        LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: np.eye(4, dtype=np.float32),
        LidarFrameDatasetSchema.lidar_pointcloud_relative_path.name: (
            f"{SCENE}/data/LIDAR_CONCAT/0.pcd.bin"
        ),
    }
    matrix = pl.Array(pl.Float32, shape=(4, 4))
    return pl.DataFrame({DatasetTableSchema.LIDAR_FRAMES.name: [[frame]]}).with_columns(
        pl.col(DatasetTableSchema.LIDAR_FRAMES.name).cast(
            pl.List(
                pl.Struct(
                    {
                        LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: matrix,
                        LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: matrix,
                        LidarFrameDatasetSchema.lidar_pointcloud_relative_path.name: pl.String,
                    }
                )
            )
        )
    )


def task(root: Path, dataframe: pl.DataFrame) -> OnlineMapTargets:
    """Map targets over the fixture classes and a 100 m by 50 m region."""
    return OnlineMapTargets(
        str(root),
        dataframe,
        record_poses(dataframe),
        classes={
            name: {"line_types": source.line_types, "regions": source.regions}
            for name, source in CLASSES.items()
        },
        roi=(-50.0, -25.0, 50.0, 25.0),
        num_points=20,
        min_length=1.0,
        augmentation_margin=5.0,
    )


class TestOnlineMapTargets(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_the_elements_of_a_record_lie_in_its_lidar_frame(self) -> None:
        """
        Input: the scene map and a record at map (10, 0) facing map +y.
        Expected: the whole divider along frame x = 0 and the border along x = -5, the crossing as
          an area, all inside the crop square that holds the region.
        """
        write_map(self.tmp_path / SCENE / "map" / "lanelet2_map.osm")

        sample = task(self.tmp_path, records(10.0, np.pi / 2)).get_gt_sample(0)

        elements = sample.elements
        self.assertEqual(sorted(elements.labels.tolist()), [0, 1, 2])
        self.assertEqual(elements.areas[elements.labels == 2].tolist(), [True])
        offsets = np.concatenate([[0], np.cumsum(elements.lengths)])
        index = int(np.flatnonzero(elements.labels == 0)[0])
        divider = elements.points[offsets[index] : offsets[index + 1]]
        np.testing.assert_allclose(divider[:, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(sorted(divider[[0, -1], 1].tolist()), [-10.0, 10.0], atol=1e-5)
        self.assertAlmostEqual(sample.spec.crop_half_size, np.hypot(50.0, 25.0) + 5.0, delta=1e-4)

    def test_maps_can_live_under_their_own_root(self) -> None:
        """
        Input: the scene map under a separate root that the targets name,
          none under the database root.
        Expected: the targets read the map from that root.
        """
        write_map(self.tmp_path / "maps" / SCENE / "map" / "lanelet2_map.osm")
        dataframe = records(10.0, np.pi / 2)

        map_task = OnlineMapTargets(
            str(self.tmp_path / "points"),
            dataframe,
            record_poses(dataframe),
            classes={
                name: {"line_types": source.line_types, "regions": source.regions}
                for name, source in CLASSES.items()
            },
            roi=(-50.0, -25.0, 50.0, 25.0),
            num_points=20,
            min_length=1.0,
            augmentation_margin=5.0,
            map_root_path=str(self.tmp_path / "maps"),
        )

        self.assertEqual(
            map_task.osm_paths, [str(self.tmp_path / "maps" / SCENE / "map" / "lanelet2_map.osm")]
        )

    def test_a_scene_without_a_map_raises(self) -> None:
        """
        Input: a record of a scene that ships no map.
        Expected: a FileNotFoundError when the targets are built.
        """
        with self.assertRaisesRegex(FileNotFoundError, "No lanelet map"):
            task(self.tmp_path, records(10.0, 0.0))
