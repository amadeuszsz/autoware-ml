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

"""Tests for the open-loop planning metrics."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.metrics.geometry.reachability import VehicleGeometry
from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.planning.suite import MotionPlanningMetricSuite, PlanningSummary
from autoware_ml.metrics.motion.tests.eval_output_fixtures import motion_output
from autoware_ml.metrics.motion.suite import AgentMotionSummary

# A 2 m x 1 m body centered on the frame origin
VEHICLE = VehicleGeometry(
    wheel_base=0.5,
    front_overhang=0.5,
    rear_overhang=1.0,
    wheel_tread=0.8,
    left_overhang=0.1,
    right_overhang=0.1,
    max_steer_angle=0.5,
)


def frame(plan_offset: float, blocker: bool) -> dict[str, torch.Tensor]:
    """One car moving 1 m per step, predicted exactly, and an ego plan of two steps.

    With ``blocker`` a parked car stands on the plan's second position.
    """
    future = torch.tensor([[[11.0, 0.0], [12.0, 0.0]]])
    gt_xy = torch.tensor([[10.0, 0.0]])
    gt_futures = future
    if blocker:
        gt_xy = torch.cat([gt_xy, torch.tensor([[4.0, 0.0]])])
        gt_futures = torch.cat([future, torch.tensor([[[4.0, 0.0], [4.0, 0.0]]])])
    count = len(gt_xy)
    return {
        "pred_xy": torch.tensor([[10.0, 0.0]]),
        "pred_labels": torch.tensor([0]),
        "pred_scores": torch.tensor([0.9]),
        "pred_trajectories": future[:, None].repeat(1, 2, 1, 1),
        "pred_mode_scores": torch.tensor([[0.5, 0.5]]),
        "pred_velocities": torch.tensor([[2.0, 0.0]]),
        "gt_xy": gt_xy,
        "gt_labels": torch.zeros(count, dtype=torch.long),
        "gt_futures": gt_futures,
        "gt_valid": torch.ones(count, 2, dtype=torch.bool),
        "gt_size": torch.full((count, 2), 1.0),
        "gt_yaw": torch.zeros(count),
        "plan": torch.tensor([[2.0, plan_offset], [4.0, plan_offset]]),
        "ego_future": torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
        "ego_future_valid": torch.ones(2, dtype=torch.bool),
    }


def evaluate(frames: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Run both metrics of a suite over the frames."""
    suite = MotionPlanningMetricSuite(
        components=[AgentMotionSummary(), PlanningSummary()],
        class_names=("car",),
        score_threshold=0.2,
        step_seconds=1.0,
        vehicle=VEHICLE,
    )
    observed = [torch.ones(len(frame["gt_xy"]), dtype=torch.bool) for frame in frames]
    suite.update(motion_output(frames, observed))
    state = suite.state_for(None, suite.components[0].filter)
    report = suite.components[0].evaluate(state, EvalStage.VAL)
    report.merge(suite.components[1].evaluate(state, EvalStage.VAL))
    return {str(key): report.metric_value(key) for key in report.metric_names()}


class TestPlanningSuite(unittest.TestCase):
    """The open-loop planning metrics."""

    def test_a_plan_on_the_ego_future_scores_zero_errors(self) -> None:
        """
        Input: the plan on the ego future, nothing in the way.
        Expected: zero L2, no collision, the agent metrics still reported.
        """
        report = evaluate([frame(0.0, blocker=False)])
        self.assertEqual(report["L2_1s"], 0.0)
        self.assertEqual(report["L2_2s"], 0.0)
        self.assertEqual(report["collision_2s"], 0.0)
        self.assertEqual(report["minFDE_car"], 0.0)

    def test_a_plan_offset_becomes_its_l2_error(self) -> None:
        """
        Input: the plan 1 m to the side.
        Expected: plan L2 1.
        """
        self.assertEqual(evaluate([frame(1.0, blocker=False)])["L2_2s"], 1.0)

    def test_a_car_on_the_plan_is_a_collision(self) -> None:
        """
        Input: a parked car on the plan's second position.
        Expected: no collision at 1 s, a collision at 2 s.
        """
        report = evaluate([frame(0.0, blocker=True)])
        self.assertEqual(report["collision_1s"], 0.0)
        self.assertEqual(report["collision_2s"], 1.0)

    def test_an_agent_counts_only_at_its_known_steps(self) -> None:
        """
        Input: a plan starting next to the origin, a second car annotated only at the second step,
          far from the plan. Its unknown first position reads as the origin.
        Expected: no collision at 1 s or 2 s.
        """
        sample = frame(0.0, blocker=False)
        sample["plan"] = torch.tensor([[0.5, 0.0], [4.0, 0.0]])
        sample["gt_xy"] = torch.cat([sample["gt_xy"], torch.tensor([[20.0, 5.0]])])
        sample["gt_labels"] = torch.zeros(2, dtype=torch.long)
        sample["gt_futures"] = torch.cat(
            [sample["gt_futures"], torch.tensor([[[0.0, 0.0], [20.0, 5.0]]])]
        )
        sample["gt_valid"] = torch.tensor([[True, True], [False, True]])
        sample["gt_size"] = torch.full((2, 2), 1.0)
        sample["gt_yaw"] = torch.zeros(2)
        report = evaluate([sample])
        self.assertEqual(report["collision_1s"], 0.0)
        self.assertEqual(report["collision_2s"], 0.0)
