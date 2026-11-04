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

"""Tests for the SparseDrive motion planner head."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.dataclasses.batch.planning import PlanningGTBatch
from autoware_ml.dataclasses.models.motion.head_outputs import MotionPlanningHeadOutputs
from autoware_ml.models.detection3d.heads.sparse4d import SparseBox3DEncoder
from autoware_ml.models.motion.motion_head import MapContext
from autoware_ml.models.planning.motion_planning_head import MotionPlanningHead
from autoware_ml.models.planning.tests.planner_fixtures import build_stages, build_planner

DIMS, QUERIES, CLASSES = 32, 12, 3


def build_head(tmp_path: Path, num_map_keys: int = 0, **overrides: object) -> MotionPlanningHead:
    """A small head with random motion anchors, the test vocabulary and the given settings."""
    rng = np.random.default_rng(0)
    motion_anchors = FixedAnchorRecipe({"anchors": rng.normal(size=(CLASSES, 6, 4, 2))})
    torch.manual_seed(0)
    head = MotionPlanningHead(
        num_classes=CLASSES,
        embed_dims=DIMS,
        anchor_recipe=motion_anchors,
        planner=build_planner(tmp_path, DIMS, with_map=num_map_keys > 0),
        ego_height=1.5,
        num_heads=4,
        agent_steps=4,
        num_agent_keys=5,
        queue_length=2,
        queue_stride=1,
        num_map_keys=num_map_keys,
        **overrides,
    )
    return fill_anchors(head)


def random_anchor(batch_size: int) -> torch.Tensor:
    """Random anchors with a unit heading."""
    anchor = torch.randn(batch_size, QUERIES, 10)
    anchor[..., 6:8] = torch.nn.functional.normalize(anchor[..., 6:8], dim=-1)
    return anchor


def run(
    head: MotionPlanningHead, batch_size: int = 2, map_context: MapContext | None = None
) -> tuple[MotionPlanningHeadOutputs, torch.Tensor]:
    """One frame of random detections, returning the outputs and the anchors."""
    anchor = random_anchor(batch_size)
    outputs = head(
        torch.randn(batch_size, QUERIES, DIMS),
        anchor,
        torch.randn(batch_size, QUERIES, CLASSES),
        SparseBox3DEncoder(DIMS),
        torch.randn(batch_size, DIMS),
        torch.zeros(batch_size, dtype=torch.bool),
        None,
        torch.eye(4).repeat(batch_size, 1, 1),
        build_stages(DIMS, batch_size),
        torch.zeros(batch_size, dtype=torch.long),
        torch.eye(4).repeat(batch_size, 1, 1),
        map_context,
    )
    return outputs, anchor


class TestMotionPlanningHead(unittest.TestCase):
    """The SparseDrive motion planner head."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)
        self.head = build_head(self.tmp_path)

    def test_the_ego_anchor_follows_the_frame_of_the_sample(self) -> None:
        """
        Input: the identity frame and a frame rotated by 0.3 rad, scaled by 1.1 and shifted.
        Expected: the identity keeps the ego anchor, the other moves its center, turns its heading
        by 0.3 rad and scales its size.
        """
        cos, sin = math.cos(0.3), math.sin(0.3)
        frame_from_ego = torch.eye(4).repeat(2, 1, 1)
        frame_from_ego[1, :3, :3] = 1.1 * torch.tensor(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]
        )
        frame_from_ego[1, :3, 3] = torch.tensor([0.2, -0.1, 0.05])
        pose = self.head.ego_pose(frame_from_ego)
        anchor = self.head.ego_anchor
        torch.testing.assert_close(pose[0], anchor[:8])
        center_x = float(anchor[0])
        expected_center = torch.tensor([1.1 * cos * center_x + 0.2, 1.1 * sin * center_x - 0.1])
        torch.testing.assert_close(pose[1, :2], expected_center)
        torch.testing.assert_close(pose[1, 3:6], anchor[3:6] + math.log(1.1))
        torch.testing.assert_close(pose[1, 6:8], torch.tensor([sin, cos]))

    def test_the_loss_is_finite_and_reaches_every_parameter(self) -> None:
        """
        Input: two frames, queries 0 and 1 matched to boxes with futures, known ego paths and
          futures.
        Expected: finite terms with a path, speed and trajectory term per planning layer, two motion
          targets, two planned frames, a gradient for every parameter.
        """
        outputs, _ = run(self.head)
        gt = AgentFutureTargets(
            agent_future_positions=torch.randn(2, 3, 4, 2),
            agent_future_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )
        planning_gt = PlanningGTBatch(
            ego_future_positions=torch.tensor([[[1.0, 0.0], [3.0, 0.2]]]).repeat(2, 1, 1),
            ego_future_valid=torch.ones(2, 2, dtype=torch.bool),
            ego_status=torch.randn(2, 5),
            ego_status_valid=torch.tensor([True, False]),
            ego_path=torch.tensor([[[1.0, 0.0], [2.0, 0.1], [4.0, 0.5]]]).repeat(2, 1, 1),
            ego_path_valid=torch.tensor([[True, True, True], [True, True, False]]),
        )
        assigned = torch.full((2, QUERIES), -1)
        assigned[0, 0], assigned[1, 1] = 11, 12
        ids = torch.tensor([[11, -1, -1], [5, 12, -1]])
        boxes = torch.zeros(2, 3, 10)
        boxes[..., 3:5] = 1.0
        losses = self.head.loss(
            outputs, assigned, boxes, ids, gt, planning_gt, torch.eye(4).repeat(2, 1, 1)
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        expected_terms = {
            f"loss_plan_{part}_{layer}"
            for part in ("path", "speed", "trajectory")
            for layer in (0, 1)
        }
        self.assertLessEqual(expected_terms, set(losses))
        self.assertEqual(losses["num_motion_targets"].item(), 2)
        self.assertEqual(losses["num_plans"].item(), 2)
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in self.head.parameters() if p.requires_grad))

    def test_a_loss_over_no_frame_reports_every_term_as_zero(self) -> None:
        """
        Input: the outputs of two frames, neither of them supervised.
        Expected: every term zero and finite, the collision terms too, a gradient for every
          parameter.
        """
        outputs, _ = run(self.head)
        none = torch.zeros(2, dtype=torch.bool)
        gt = AgentFutureTargets(
            agent_future_positions=torch.randn(2, 3, 4, 2),
            agent_future_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )
        planning_gt = PlanningGTBatch(
            ego_future_positions=torch.zeros(2, 2, 2),
            ego_future_valid=torch.ones(2, 2, dtype=torch.bool),
            ego_status=torch.randn(2, 5),
            ego_status_valid=torch.ones(2, dtype=torch.bool),
            ego_path=torch.zeros(2, 3, 2),
            ego_path_valid=torch.ones(2, 3, dtype=torch.bool),
        )

        losses = self.head.loss(
            outputs.select_samples(none),
            torch.full((0, QUERIES), -1),
            torch.zeros(0, 3, 10),
            torch.zeros(0, 3, dtype=torch.long),
            gt.select_samples(none),
            planning_gt.select_samples(none),
            torch.eye(4).repeat(0, 1, 1),
        )

        self.assertLessEqual(
            {"loss_plan_collision", "plan_collision_rate", "num_plans"}, set(losses)
        )
        self.assertTrue(all(value.item() == 0.0 for value in losses.values()))
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in self.head.parameters() if p.requires_grad))

    def test_the_selected_plan_is_one_of_the_kept_trajectories(self) -> None:
        """
        Input: one frame, every detection with a low score.
        Expected: the plan is the best scored of the trajectories the last planning layer kept.
        """
        outputs, anchor = run(self.head.eval(), batch_size=1)
        plan = self.head.select_plan(outputs, anchor, torch.zeros(1, QUERIES))
        best = outputs.plan_logits[0].argmax()
        torch.testing.assert_close(plan[0], outputs.plan_trajectories[0, best])

    def test_a_planner_with_a_map_reads_the_map_queries(self) -> None:
        """
        Input: a planner attending to 3 map queries run on one frame with two different maps, and
          the planners given a map context they do not take.
        Expected: the plan scores differ between the maps, the mismatched calls raise.
        """
        planner = build_head(self.tmp_path, num_map_keys=3).eval()
        scores = []
        for seed in range(2):
            generator = torch.Generator().manual_seed(seed)
            context = MapContext(
                feature=torch.randn(1, 6, DIMS, generator=generator),
                anchor_embed=torch.randn(1, 6, DIMS, generator=generator),
                scores=torch.rand(1, 6, generator=generator),
            )
            planner.reset()
            torch.manual_seed(0)
            with torch.no_grad():
                scores.append(run(planner, batch_size=1, map_context=context)[0].plan_logits)

        self.assertFalse(torch.allclose(scores[0], scores[1]))
        with self.assertRaisesRegex(ValueError, "exactly to a head with a map"):
            planner.reset()
            run(planner, batch_size=1)
        with self.assertRaisesRegex(ValueError, "exactly to a head with a map"):
            run(build_head(self.tmp_path), batch_size=1, map_context=context)

    def test_a_detached_planner_trains_without_touching_the_detection(self) -> None:
        """
        Input: a planner built with detach_detection run on detection features, lidar stages and an
          anchor encoder that track gradients, then its outputs backpropagated.
        Expected: the planner parameters get gradients, the detection features, the stages and the
          anchor encoder none.
        """
        planner = build_head(self.tmp_path, detach_detection=True)
        encoder = SparseBox3DEncoder(DIMS)
        feature = torch.randn(1, QUERIES, DIMS, requires_grad=True)
        stages = build_stages(DIMS, 1)
        stage = stages[0][0].requires_grad_(True)

        outputs = planner(
            feature,
            random_anchor(1),
            torch.randn(1, QUERIES, CLASSES),
            encoder,
            torch.randn(1, DIMS),
            torch.zeros(1, dtype=torch.bool),
            None,
            torch.eye(4)[None],
            stages,
            torch.zeros(1, dtype=torch.long),
            torch.eye(4)[None],
        )
        (outputs.agent_trajectories.sum() + outputs.plan_logits.sum()).backward()

        self.assertIsNone(feature.grad)
        self.assertIsNone(stage.grad)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertIsNotNone(planner.motion_reg[-1].weight.grad)
        self.assertIsNotNone(planner.planner.layers[-1].trajectory_score[-1].weight.grad)
