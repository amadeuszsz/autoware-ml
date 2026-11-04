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

"""Tests for the collation of the ego futures."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.dataclasses.batch.planning import PlanningGTBatch, PlanningGTSample
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample


def augmentation(rotation: torch.Tensor, scale: float) -> LiDARTransformationSample:
    """A lidar-space augmentation of one rotation and one scale."""
    return LiDARTransformationSample.create_lidar_transformation_sample(
        rotation_matrix=rotation,
        scale_factor=scale,
        translation_vector=torch.zeros(1, 3),
        transformation_order=[],
    )


def sample() -> PlanningGTSample:
    """The ego at 10 m/s along x, its path reaching 2 m of 3."""
    return PlanningGTSample(
        ego_future_positions=torch.tensor([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        ego_future_valid=torch.tensor([True, True]),
        ego_status=torch.tensor([10.0, 0.0, 1.0, 0.0, 0.2]),
        ego_status_valid=True,
        ego_path=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ego_path_valid=torch.tensor([True, True, False]),
    )


class TestPlanningBatch(unittest.TestCase):
    """The collation of the ego futures."""

    def test_the_augmentation_moves_the_ego_future_and_turns_the_status(self) -> None:
        """
        Input: a 90 degree yaw rotation scaled by 2.
        Expected: the ego future, the path and the velocity rotate to y and double, the yaw rate and
          the path validity stay.
        """
        rotation = torch.tensor(
            [
                [math.cos(math.pi / 2), -math.sin(math.pi / 2), 0.0],
                [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        batch = PlanningGTBatch.collate_gt_samples([sample()], [augmentation(rotation, 2.0)])
        torch.testing.assert_close(
            batch.ego_future_positions[0],
            torch.tensor([[0.0, 10.0], [0.0, 20.0]]),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            batch.ego_status[0], torch.tensor([0.0, 20.0, 0.0, 2.0, 0.2]), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            batch.ego_path[0], torch.tensor([[0.0, 2.0], [0.0, 4.0], [0.0, 4.0]]), atol=1e-5, rtol=0
        )
        self.assertEqual(batch.ego_path_valid[0].tolist(), [True, True, False])

    def test_a_mirror_flips_the_yaw_rate(self) -> None:
        """
        Input: a mirror of the y axis.
        Expected: the yaw rate changes sign.
        """
        mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        batch = PlanningGTBatch.collate_gt_samples([sample()], [augmentation(mirror, 1.0)])
        self.assertAlmostEqual(batch.ego_status[0, 4].item(), -0.2, places=6)
