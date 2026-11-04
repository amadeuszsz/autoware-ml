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

"""Tests for the agent futures the motion task reads from the later records of a scene."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.datamodule.common.motion import MotionTask
from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids

FRAMES = 21


def pose(x: float, yaw: float = 0.0) -> np.ndarray:
    """Ego pose at x along the global x axis, heading yaw."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    matrix[0, 3] = x
    return matrix


def box(center: np.ndarray, instance: str) -> dict:
    """Box record with a center in the frame of its record."""
    params = np.zeros(10, dtype=np.float32)
    params[:3] = center
    params[3:6] = 1.0
    return {
        Box3DDatasetSchema.BOX3D_PARAMS.name: params,
        Box3DDatasetSchema.BOX3D_INSTANCE_ID.name: instance,
    }


def records(
    jump_frame: int | None = None, walker_id: str = "walker", key_every: int = 1
) -> pl.DataFrame:
    """A 2 s scene at 10 Hz, the ego driving 10 m/s along x and two agents.

    The walker moves 2 m/s along global y from (20, 0), the parked car stays at global (30, 5).
    With ``jump_frame`` the walker's id names an object 100 m away from that frame on. Every
    ``key_every``-th frame is a key frame.
    """
    rows = []
    for frame in range(FRAMES):
        ego = pose(1.0 * frame)
        to_frame = np.linalg.inv(ego.astype(np.float64))
        walker_global = np.array([20.0, 0.2 * frame, 0.0])
        if jump_frame is not None and frame >= jump_frame:
            walker_global = walker_global + np.array([100.0, 0.0, 0.0])
        car_global = np.array([30.0, 5.0, 0.0])
        boxes = [
            box((to_frame @ np.append(center, 1.0))[:3], instance)
            for center, instance in ((walker_global, walker_id), (car_global, "car"))
        ]
        rows.append(
            {
                DatasetTableSchema.SCENARIO_ID.name: "scene",
                DatasetTableSchema.TIMESTAMP_SECONDS.name: 100.0 + 0.1 * frame,
                DatasetTableSchema.BOXES_3D.name: boxes,
                DatasetTableSchema.LIDAR_FRAMES.name: [
                    {
                        LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: ego,
                        LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: np.eye(
                            4, dtype=np.float32
                        ),
                        LidarFrameDatasetSchema.lidar_keyframe.name: frame % key_every == 0,
                    }
                ],
            }
        )
    matrix = pl.Array(pl.Float32, shape=(4, 4))
    return pl.DataFrame(rows).with_columns(
        pl.col(DatasetTableSchema.LIDAR_FRAMES.name).cast(
            pl.List(
                pl.Struct(
                    {
                        LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: matrix,
                        LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: matrix,
                        LidarFrameDatasetSchema.lidar_keyframe.name: pl.Boolean,
                    }
                )
            )
        ),
        pl.col(DatasetTableSchema.BOXES_3D.name).cast(
            pl.List(
                pl.Struct(
                    {
                        Box3DDatasetSchema.BOX3D_PARAMS.name: pl.Array(pl.Float32, shape=(10,)),
                        Box3DDatasetSchema.BOX3D_INSTANCE_ID.name: pl.String,
                    }
                )
            )
        ),
    )


def task(dataframe: pl.DataFrame) -> MotionTask:
    """Motion task over 0.5 s steps and 2 agent steps."""
    return MotionTask(
        "/unused",
        dataframe,
        agent_steps=2,
        step_seconds=0.5,
        time_tolerance_seconds=0.03,
        max_speed=50.0,
    )


class TestMotionTask(unittest.TestCase):
    def test_futures_are_expressed_in_the_frame_of_the_current_record(self) -> None:
        """
        Input: the scene at its first record, the ego at x=0 moving 10 m/s.
        Expected: the walker moves to y=1 and y=2 at x=20, the parked car stays at (30, 5), every
          step known.
        """
        sample = task(records()).get_data_sample(0).motion_gt_sample
        ids = hash_instance_ids(["car", "walker"])
        positions = dict(zip(sample.agent_instance_ids.tolist(), sample.agent_future_positions))
        np.testing.assert_allclose(positions[int(ids[1])][:, :2], [[20, 1], [20, 2]], atol=1e-4)
        np.testing.assert_allclose(positions[int(ids[0])][:, :2], [[30, 5], [30, 5]], atol=1e-4)
        self.assertEqual(len(sample.agent_instance_ids), 2)
        self.assertTrue(bool(sample.agent_future_valid.all()))

    def test_steps_past_the_scene_end_are_invalid(self) -> None:
        """
        Input: the record 0.7 s before the end of the scene.
        Expected: the 1.0 s step lies past the last record for every agent.
        """
        sample = task(records()).get_data_sample(FRAMES - 8).motion_gt_sample
        self.assertTrue(sample.agent_future_valid[:, 1].logical_not().all())

    def test_an_id_switch_cuts_the_trajectory(self) -> None:
        """
        Input: the walker id names an object 100 m away from frame 10 on, the futures of frame 0.
        Expected: the walker keeps its 0.5 s step and loses the 1.0 s one, the car keeps both.
        """
        sample = task(records(jump_frame=10)).get_data_sample(0).motion_gt_sample
        valid = dict(zip(sample.agent_instance_ids.tolist(), sample.agent_future_valid.tolist()))
        walker, car = (int(value) for value in hash_instance_ids(["walker", "car"]))
        self.assertEqual(valid[walker], [True, False])
        self.assertEqual(valid[car], [True, True])

    def test_key_frames_sparser_than_the_steps_are_rejected(self) -> None:
        """
        Input: a 10 Hz scene with a key frame every 1 s, read with 0.5 s steps.
        Expected: ValueError, every second step would land on a frame without boxes.
        """
        with self.assertRaisesRegex(ValueError, r"scenarios \['scene'\] come every up to 1.00 s"):
            task(records(key_every=10))

    def test_a_scenario_sparser_than_the_steps_is_rejected_among_dense_ones(self) -> None:
        """
        Input: two dense 10 Hz scenes and one with a key frame every 1 s, read with 0.5 s steps.
        Expected: ValueError naming the sparse scene, a median over all scenes would let it pass.
        """
        scene = DatasetTableSchema.SCENARIO_ID.name
        mixed = pl.concat(
            [
                records().with_columns(pl.lit("dense_a").alias(scene)),
                records().with_columns(pl.lit("dense_b").alias(scene)),
                records(key_every=10).with_columns(pl.lit("sparse").alias(scene)),
            ]
        )

        with self.assertRaisesRegex(ValueError, r"scenarios \['sparse'\]"):
            task(mixed)

    def test_records_without_two_key_frames_in_a_scenario_are_rejected(self) -> None:
        """
        Input: a scene whose only key frame is its first frame.
        Expected: ValueError, no future step can ever be supervised.
        """
        with self.assertRaisesRegex(ValueError, "holds two key frames"):
            task(records(key_every=100))

    def test_boxes_sharing_an_instance_id_are_rejected(self) -> None:
        """
        Input: the walker and the car carry the same instance id.
        Expected: ValueError when the task is built, the futures of the id are ambiguous.
        """
        with self.assertRaisesRegex(ValueError, "two boxes of one instance"):
            task(records(walker_id="car"))

    def test_a_tolerance_of_half_a_step_is_rejected(self) -> None:
        """
        Input: a tolerance as large as half the step.
        Expected: ValueError, one record could serve two steps.
        """
        with self.assertRaisesRegex(ValueError, "half a step"):
            MotionTask(
                "/unused",
                records(),
                agent_steps=2,
                step_seconds=0.1,
                time_tolerance_seconds=0.05,
                max_speed=50.0,
            )
