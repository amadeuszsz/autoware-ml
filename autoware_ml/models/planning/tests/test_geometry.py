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

"""Tests of the frame helpers of the planner geometry."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.models.planning.geometry import (
    boxes_overlap,
    length_to_ego,
    to_ego,
    to_frame,
    yaw_to_ego,
)


def _frame_from_ego() -> torch.Tensor:
    """A rotated, scaled and shifted frame, and a mirrored one."""
    cos, sin = math.cos(0.3), math.sin(0.3)
    transform = torch.eye(4).repeat(2, 1, 1)
    transform[0, :2, :2] = 1.05 * torch.tensor([[cos, -sin], [sin, cos]])
    transform[0, :2, 3] = torch.tensor([0.1, -0.15])
    transform[1, 1, 1] = -1.0
    return transform


def _to_frame_yaw(yaw: torch.Tensor, frame_from_ego: torch.Tensor) -> torch.Tensor:
    """Headings of the ego frame in the point cloud frame."""
    direction = torch.stack([yaw.cos(), yaw.sin()], dim=-1)
    moved = (frame_from_ego[:, None, :2, :2] @ direction[..., None])[..., 0]
    return torch.atan2(moved[..., 1], moved[..., 0])


class TestEgoFrameHelpers(unittest.TestCase):
    def test_headings_and_lengths_return_to_the_ego_frame(self) -> None:
        """
        Input: ego frame headings and box sizes moved into a rotated, scaled or mirrored frame.
        Expected: the helpers give back the ego frame values.
        """
        frame_from_ego = _frame_from_ego()
        yaw = torch.tensor([[0.5, -2.0], [0.5, -2.0]])
        size = torch.tensor([[[4.0, 2.0]], [[4.0, 2.0]]])
        torch.testing.assert_close(
            yaw_to_ego(_to_frame_yaw(yaw, frame_from_ego), frame_from_ego), yaw
        )
        scaled = size * torch.tensor([1.05, 1.0])[:, None, None]
        torch.testing.assert_close(length_to_ego(scaled, frame_from_ego), size)

    def test_collision_labels_do_not_depend_on_the_augmentation(self) -> None:
        """
        Input: a standing ego and a car beside its front corner, in the ego frame and moved
        into an augmented frame, then brought back with the helpers.
        Expected: the same collision label.
        """
        frame_from_ego = _frame_from_ego()[:1]
        plans = torch.zeros(1, 1, 3, 2)
        agents = torch.tensor([[[[3.0, 1.6]] * 3]])
        size = torch.tensor([[[4.0, 2.0]]])
        yaw = torch.zeros(1, 1)
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        ego_size = torch.tensor([4.5, 1.9])
        expected = boxes_overlap(plans, 1.4, ego_size, agents, size, yaw, mask)
        from_frame = boxes_overlap(
            to_ego(to_frame(plans, frame_from_ego), frame_from_ego),
            1.4,
            ego_size,
            agents,
            length_to_ego(size * 1.05, frame_from_ego),
            yaw_to_ego(_to_frame_yaw(yaw, frame_from_ego), frame_from_ego),
            mask,
        )
        torch.testing.assert_close(from_frame, expected)
