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

"""Tests for the ego future the planning task reads from the records of a scene."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from autoware_ml.datamodule.common.planning import PlanningTask, resample_path
from autoware_ml.datamodule.tests.test_motion import FRAMES, records

PATH = {
    "path_distances": [1.0, 4.0, 12.0],
    "path_seconds": 1.0,
    "path_step_seconds": 0.2,
    "path_min_spacing": 0.3,
}


def task(dataframe: pl.DataFrame) -> PlanningTask:
    """Planning task over 0.5 s steps, 2 ego steps and a 1 s ego path."""
    return PlanningTask(
        "/unused",
        dataframe,
        ego_steps=2,
        step_seconds=0.5,
        status_step_seconds=0.1,
        time_tolerance_seconds=0.03,
        **PATH,
    )


class TestPlanningTask(unittest.TestCase):
    """The ego future the planning task reads from the records of a scene."""

    def test_the_ego_future_is_expressed_in_the_frame_of_the_current_record(self) -> None:
        """
        Input: the scene at its first record, the ego at x=0 moving 10 m/s.
        Expected: the ego reaches x=5 and x=10.
        """
        sample = task(records()).get_data_sample(0).planning_gt_sample
        np.testing.assert_allclose(sample.ego_future_positions[:, :2], [[5, 0], [10, 0]], atol=1e-4)
        self.assertEqual(sample.ego_future_valid.tolist(), [True, True])

    def test_steps_past_the_scene_end_are_invalid(self) -> None:
        """
        Input: the record 0.7 s before the end of the scene.
        Expected: the 0.5 s step is known, the 1.0 s step lies past the last record.
        """
        sample = task(records()).get_data_sample(FRAMES - 8).planning_gt_sample
        self.assertEqual(sample.ego_future_valid.tolist(), [True, False])

    def test_ego_status_comes_from_the_past_records(self) -> None:
        """
        Input: a record with two past records, the ego driving 10 m/s straight.
        Expected: velocity (10, 0), no acceleration, no yaw rate. The first record has no status.
        """
        planning = task(records())
        sample = planning.get_data_sample(5).planning_gt_sample
        np.testing.assert_allclose(sample.ego_status, [10, 0, 0, 0, 0], atol=1e-3)
        self.assertTrue(sample.ego_status_valid)
        self.assertFalse(planning.get_data_sample(0).planning_gt_sample.ego_status_valid)

    def test_a_tolerance_of_half_a_step_is_rejected(self) -> None:
        """
        Input: a tolerance as large as half the status step.
        Expected: ValueError, one record could serve two steps.
        """
        with self.assertRaisesRegex(ValueError, "half a step"):
            PlanningTask(
                "/unused",
                records(),
                ego_steps=2,
                step_seconds=0.5,
                status_step_seconds=0.1,
                time_tolerance_seconds=0.05,
                **PATH,
            )

    def test_the_ego_path_samples_the_future_at_fixed_arc_lengths(self) -> None:
        """
        Input: the first record, the ego driving 10 m along x over the 1 s path horizon, and the
          record 0.2 s before the scene end.
        Expected: the path holds x=1 and x=4, the 12 m point lies past the horizon. Near the end the
          path covers 2 m, so only the 1 m point is reached.
        """
        planning = task(records())
        sample = planning.get_data_sample(0).planning_gt_sample
        np.testing.assert_allclose(sample.ego_path[:2, :2], [[1, 0], [4, 0]], atol=1e-4)
        self.assertEqual(sample.ego_path_valid.tolist(), [True, True, False])
        self.assertEqual(
            planning.get_data_sample(FRAMES - 3).planning_gt_sample.ego_path_valid.tolist(),
            [
                True,
                False,
                False,
            ],
        )

    def test_the_path_skips_the_pose_noise_of_a_standing_ego(self) -> None:
        """
        Input: three positions within 0.15 m of the start, then a move to x=2.
        Expected: the jitter adds no length, the 1 m point lies at x=1 and the 4 m one is
          not reached.
        """
        positions = np.array([[0, 0, 0], [0.05, 0, 0], [0.1, 0.1, 0], [0.0, 0.1, 0], [2, 0, 0]])
        path, valid = resample_path(positions, np.array([1.0, 4.0]), 0.3)
        np.testing.assert_allclose(path[0], [1, 0, 0], atol=1e-6)
        self.assertEqual(valid.tolist(), [True, False])
