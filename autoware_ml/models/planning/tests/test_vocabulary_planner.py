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

"""Tests for the vocabulary planner, its geometry and its driving commands."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from autoware_ml.models.planning.commands import DrivingCommand, driving_commands
from autoware_ml.models.planning.geometry import boxes_overlap, to_ego, to_frame
from autoware_ml.models.planning.tests.planner_fixtures import build_stages, build_planner
from autoware_ml.models.planning.vocabulary_planner import (
    PathKeyPointsGenerator,
    compose_trajectories,
)

DIMS = 32


def yaw_transform(yaw: float, x: float = 0.0) -> torch.Tensor:
    """A frame turned by ``yaw`` and shifted by ``x`` along its x axis ``(1, 4, 4)``."""
    matrix = torch.eye(4)
    matrix[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    matrix[0, 3] = x
    return matrix[None]


class TestVocabularyPlanner(unittest.TestCase):
    """The vocabulary planner, its geometry and its driving commands."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_trajectories_drive_the_path_at_the_speed_profile(self) -> None:
        """
        Input: a straight path with points at 1, 2 and 3 m, driven at 2 m/s and at 8 m/s for two
          0.5 s steps.
        Expected: 1 m and 2 m at 2 m/s, 4 m and 8 m at 8 m/s, the last path segment extended.
        """
        path = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        trajectories = compose_trajectories(
            path, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[2.0, 2.0], [8.0, 8.0]]), 0.5
        )
        torch.testing.assert_close(trajectories[0, :, :, 0], torch.tensor([[1.0, 2.0], [4.0, 8.0]]))
        self.assertTrue(bool((trajectories[..., 1] == 0).all()))

    def test_path_keypoints_stack_the_heights_over_every_point(self) -> None:
        """
        Input: one path of two points, keypoints at 0 and 1 m height.
        Expected: four keypoints, both heights above the first point, then the second point.
        """
        keypoints = PathKeyPointsGenerator(2, (0.0, 1.0))(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), torch.zeros(1, 1, DIMS)
        )
        expected = torch.tensor(
            [[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [3.0, 4.0, 0.0], [3.0, 4.0, 1.0]]
        )
        torch.testing.assert_close(keypoints[0, 0], expected)

    def test_frames_round_trip_and_commands_read_the_last_known_step(self) -> None:
        """
        Input: points moved into a frame turned by 30 degrees and back, and ego futures ending 3 m
          left, turning right at the last known step with an unknown step after it, and unknown.
        Expected: the points come back, the commands are left, right and straight.
        """
        points = torch.tensor([[[1.0, 2.0], [-3.0, 0.5]]])
        transform = yaw_transform(math.pi / 6, x=2.0)
        torch.testing.assert_close(to_ego(to_frame(points, transform), transform), points)
        futures = torch.tensor(
            [[[5.0, 0.0], [10.0, 3.0]], [[5.0, -3.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
        )
        valid = torch.tensor([[True, True], [True, False], [False, False]])
        self.assertEqual(
            driving_commands(futures, valid, 2.0).tolist(),
            [
                DrivingCommand.LEFT,
                DrivingCommand.RIGHT,
                DrivingCommand.STRAIGHT,
            ],
        )

    def test_boxes_overlap_at_a_shared_step_only(self) -> None:
        """
        Input: an ego plan driving to x=10 and x=20, an agent at x=20 at the second step, another
          one at x=20 at the first step only.
        Expected: the plan collides with the first agent and not with the second.
        """
        plan = torch.tensor([[[[10.0, 0.0], [20.0, 0.0]]]])
        paths = torch.tensor([[[[0.0, 30.0], [20.0, 0.0]], [[20.0, 0.0], [0.0, 30.0]]]])
        size = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
        yaw = torch.zeros(1, 2)
        ego = torch.tensor([4.0, 2.0])
        mask = torch.ones(1, 1, 2, dtype=torch.bool)
        self.assertTrue(
            boxes_overlap(plan, 0.0, ego, paths[:, :1], size[:, :1], yaw[:, :1], mask).item()
        )
        self.assertFalse(
            boxes_overlap(plan, 0.0, ego, paths[:, 1:], size[:, 1:], yaw[:, 1:], mask).item()
        )

    def test_every_layer_scores_what_it_received_and_keeps_the_best(self) -> None:
        """
        Input: two frames planned through a vocabulary of 6 paths and 4 speed profiles, the layers
          keeping (4, 3) then (2, 2), the second frame turned by 90 degrees.
        Expected: the first layer scores the whole vocabulary, the second the 4 paths and 3 speed
          profiles the first kept, the planner returns the 4 trajectories of the last layer, those
          of the turned frame are its vocabulary trajectories turned by 90 degrees.
        """
        planner = build_planner(self.tmp_path, DIMS).eval()
        transforms = torch.cat([torch.eye(4)[None], yaw_transform(math.pi / 2)])
        with torch.no_grad():
            outputs = planner(
                torch.randn(2, DIMS),
                torch.tensor([0, 1]),
                torch.randn(2, 5, DIMS),
                None,
                build_stages(DIMS, 2),
                transforms,
            )
        first, second = outputs.layers
        self.assertEqual(first.path_ids.shape, (2, 6))
        self.assertEqual(first.speed_ids.shape, (2, 4))
        self.assertEqual(first.trajectory_ids.shape, (2, 12))
        self.assertEqual(second.path_ids.shape, (2, 4))
        self.assertEqual(second.speed_ids.shape, (2, 3))
        for frame in range(2):
            self.assertEqual(
                set(second.path_ids[frame].tolist()),
                set(first.path_ids[frame, first.path_logits[frame].topk(4).indices].tolist()),
            )
        self.assertEqual(outputs.trajectories.shape, (2, 4, 2, 2))
        self.assertEqual(outputs.logits.shape, (2, 4))
        vocabulary = planner.trajectory_vocabulary[second.trajectory_ids[1]]
        turned = torch.stack([-vocabulary[..., 1], vocabulary[..., 0]], dim=-1)
        torch.testing.assert_close(outputs.trajectories[1], turned)

    def test_the_loss_targets_the_nearest_entry(self) -> None:
        """
        Input: a ground truth on path 4 at 4 m/s, and the same scores with the logits of path 4
          or of path 0 raised.
        Expected: raising the nearest path lowers its loss, every term is finite.
        """
        planner = build_planner(self.tmp_path, DIMS).eval()
        with torch.no_grad():
            outputs = planner(
                torch.randn(1, DIMS),
                torch.tensor([0]),
                torch.randn(1, 5, DIMS),
                None,
                build_stages(DIMS, 1),
                torch.eye(4)[None],
            )
        path = planner.path_vocabulary[4][None]
        future = compose_trajectories(
            planner.path_vocabulary[4:5],
            torch.tensor([1.0, 2.0, 4.0]),
            torch.tensor([[4.0, 4.0]]),
            0.5,
        )[0]
        layers = list(outputs.layers)
        losses = []
        for raised in (4, 0):
            logits = torch.full_like(layers[0].path_logits, -4.0)
            logits[0, (layers[0].path_ids[0] == raised).nonzero()[0]] = 4.0
            layers[0] = layers[0]._replace(path_logits=logits)
            losses.append(
                planner.loss(
                    layers,
                    outputs.collision_logits,
                    torch.zeros_like(outputs.collision_logits),
                    path,
                    torch.ones(1, 3, dtype=torch.bool),
                    future,
                    torch.ones(1, 2, dtype=torch.bool),
                )
            )
        self.assertTrue(all(torch.isfinite(value) for value in losses[0].values()))
        self.assertLess(losses[0]["loss_plan_path_0"], losses[1]["loss_plan_path_0"])

    def test_a_trajectory_meeting_a_confident_agent_ranks_last(self) -> None:
        """
        Input: two trajectories, the best scored one ending where a confident agent stands, the same
          agent with a low score.
        Expected: the other trajectory is selected for the confident agent, the best one otherwise.
        """
        planner = build_planner(self.tmp_path, DIMS)
        trajectories = torch.tensor([[[[2.0, 0.0], [4.0, 0.0]], [[2.0, 3.0], [3.0, 6.0]]]])
        logits = torch.tensor([[3.0, 0.0]])
        agent = torch.tensor([[[[4.0, 0.0], [4.0, 0.0]]]])
        size, yaw = torch.tensor([[[2.0, 2.0]]]), torch.zeros(1, 1)
        blocked = planner.select(logits, trajectories, agent, size, yaw, torch.tensor([[0.9]]))
        free = planner.select(logits, trajectories, agent, size, yaw, torch.tensor([[0.1]]))
        torch.testing.assert_close(blocked[0], trajectories[0, 1])
        torch.testing.assert_close(free[0], trajectories[0, 0])

    def test_invalid_layer_counts_are_rejected(self) -> None:
        """
        Input: a layer keeping more paths than it receives.
        Expected: ValueError.
        """
        with self.assertRaisesRegex(ValueError, "positive share"):
            build_planner(self.tmp_path, DIMS, keep=((4, 3), (5, 2)))
