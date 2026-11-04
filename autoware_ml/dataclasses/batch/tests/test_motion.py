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

"""Tests for the collation of the agent futures."""

from __future__ import annotations

import math

import unittest

import torch

from autoware_ml.dataclasses.batch.motion import MotionGTBatch, MotionGTSample
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample


def sample() -> MotionGTSample:
    """Two agents (ids 7 and 9) moving along x."""
    return MotionGTSample(
        agent_instance_ids=torch.tensor([7, 9]),
        agent_future_positions=torch.tensor(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[5.0, 1.0, 0.0], [6.0, 1.0, 0.0]]]
        ),
        agent_future_valid=torch.tensor([[True, True], [True, False]]),
    )


def augmentation(rotation: torch.Tensor, scale: float) -> LiDARTransformationSample:
    """A lidar-space augmentation of one rotation and one scale."""
    return LiDARTransformationSample.create_lidar_transformation_sample(
        rotation_matrix=rotation,
        scale_factor=scale,
        translation_vector=torch.zeros(1, 3),
        transformation_order=[],
    )


class TestMotion(unittest.TestCase):
    def test_agents_follow_their_box_slots_by_instance_id(self) -> None:
        """
        Input: box slots holding ids 9, 3 (unknown to the motion sample) and padding,
          no augmentation.
        Expected: slot 0 gets the future of id 9, the other slots stay invalid.
        """
        batch = MotionGTBatch.collate_gt_samples([sample()], torch.tensor([[9, 3, -1]]), [None])
        torch.testing.assert_close(
            batch.agent_future_positions[0, 0], torch.tensor([[5.0, 1.0], [6.0, 1.0]])
        )
        self.assertEqual(
            batch.agent_future_valid[0].tolist(), [[True, False], [False, False], [False, False]]
        )

    def test_the_augmentation_moves_the_agent_positions(self) -> None:
        """
        Input: a 90 degree yaw rotation scaled by 2, the slot of id 7.
        Expected: the positions rotate to y and double.
        """
        rotation = torch.tensor(
            [
                [math.cos(math.pi / 2), -math.sin(math.pi / 2), 0.0],
                [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        batch = MotionGTBatch.collate_gt_samples(
            [sample()], torch.tensor([[7]]), [augmentation(rotation, 2.0)]
        )
        torch.testing.assert_close(
            batch.agent_future_positions[0, 0],
            torch.tensor([[0.0, 2.0], [0.0, 4.0]]),
            atol=1e-5,
            rtol=0,
        )

    def test_a_mirror_flips_the_lateral_positions(self) -> None:
        """
        Input: a mirror of the y axis, the slot of id 9.
        Expected: the lateral positions change sign.
        """
        mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        batch = MotionGTBatch.collate_gt_samples(
            [sample()], torch.tensor([[9]]), [augmentation(mirror, 1.0)]
        )
        self.assertEqual(batch.agent_future_positions[0, 0, 0, 1].item(), -1.0)
