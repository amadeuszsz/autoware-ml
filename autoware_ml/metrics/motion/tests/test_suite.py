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

"""Tests for the open-loop motion prediction metrics."""

from __future__ import annotations

import unittest
from typing import Any

import torch

from autoware_ml.metrics.base import EvalStage, MetricFilter
from autoware_ml.metrics.motion.suite import AgentMotionSummary, MotionMetricSuite
from autoware_ml.metrics.report import MetricKey


def frame(prediction_offset: float) -> dict[str, torch.Tensor]:
    """One car moving 1 m per step, predicted with an offset."""
    future = torch.tensor([[[11.0, 0.0], [12.0, 0.0]]])
    return {
        "pred_xy": torch.tensor([[10.0, 0.0]]),
        "pred_labels": torch.tensor([0]),
        "pred_scores": torch.tensor([0.9]),
        "pred_trajectories": (future + torch.tensor([0.0, prediction_offset]))[:, None].repeat(
            1, 2, 1, 1
        ),
        "gt_xy": torch.tensor([[10.0, 0.0]]),
        "gt_labels": torch.zeros(1, dtype=torch.long),
        "gt_futures": future,
        "gt_valid": torch.ones(1, 2, dtype=torch.bool),
        "gt_size": torch.full((1, 2), 1.0),
        "gt_yaw": torch.zeros(1),
    }


def evaluate(frames: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Run the agent metric of a suite over the frames."""
    suite = MotionMetricSuite(
        components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
    )
    observed = [torch.ones(len(frame["gt_xy"]), dtype=torch.bool) for frame in frames]
    suite.update({"motion_frames": frames, "gt_observed": observed, "scored": [True] * len(frames)})
    state = suite.state_for(None, suite.components[0].filter)
    report = suite.components[0].evaluate(state, EvalStage.VAL)
    return {str(key): report.metric_value(key) for key in report.metric_names()}


class _Road(MetricFilter):
    """A region filter keeping every agent."""

    name = "road"

    def keep(self, xyz: Any, context: dict[str, Any]) -> Any:
        return [True] * len(xyz)


class TestMotionSuite(unittest.TestCase):
    """The open-loop motion prediction metrics."""

    def test_a_perfect_prediction_scores_zero_errors(self) -> None:
        """
        Input: the car's future predicted exactly.
        Expected: zero minADE and minFDE, no miss, EPA 1.
        """
        report = evaluate([frame(0.0)])
        self.assertEqual(report["minADE_car"], 0.0)
        self.assertEqual(report["minFDE_car"], 0.0)
        self.assertEqual(report["MR_car"], 0.0)
        self.assertEqual(report["EPA_car"], 1.0)

    def test_offsets_become_displacement_errors_and_misses(self) -> None:
        """
        Input: the car predicted 3 m to the side.
        Expected: minADE and minFDE 3, a miss, EPA 0.
        """
        report = evaluate([frame(3.0)])
        self.assertEqual(report["minADE_car"], 3.0)
        self.assertEqual(report["minFDE_car"], 3.0)
        self.assertEqual(report["MR_car"], 1.0)
        self.assertEqual(report["EPA_car"], 0.0)

    def test_a_prediction_on_a_car_without_a_known_future_is_no_false_positive(self) -> None:
        """
        Input: three cars detected exactly, the first two with a known future predicted exactly, the
          third at the end of its scene without one.
        Expected: no false positive, two hits over three ground truth boxes as in SparseDrive's EPA,
          zero displacement over the cars with a future.
        """
        known = frame(0.0)
        unknown = frame(0.0)
        unknown["pred_xy"] = unknown["gt_xy"] = torch.tensor([[30.0, 0.0]])
        unknown["gt_futures"] = torch.zeros(1, 2, 2)
        unknown["gt_valid"] = torch.zeros(1, 2, dtype=torch.bool)
        other = frame(0.0)
        other["pred_xy"] = other["gt_xy"] = torch.tensor([[-10.0, 0.0]])
        merged = {name: torch.cat([known[name], other[name], unknown[name]]) for name in known}

        report = evaluate([merged])

        self.assertAlmostEqual(report["EPA_car"], 2.0 / 3.0, places=9)
        self.assertEqual(report["minADE_car"], 0.0)
        self.assertEqual(report["minFDE_car"], 0.0)
        self.assertEqual(report["MR_car"], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a second device")
    def test_unscored_flags_follow_the_device_of_the_frames(self) -> None:
        """
        Input: one frame whose tensors live on the GPU.
        Expected: its unscored flag lives there too, so ranks can gather it with the frame fields.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
        )
        sample = {name: value.cuda() for name, value in frame(0.0).items()}
        suite.update(
            {
                "motion_frames": [sample],
                "gt_observed": [torch.ones(1, dtype=torch.bool)],
                "scored": [True],
            }
        )
        self.assertEqual(suite.unscored[0].device, sample["pred_xy"].device)

    def test_only_the_confident_predictions_are_kept(self) -> None:
        """
        Input: a scored frame with a prediction above and one below the score threshold, and an
          unscored copy of it.
        Expected: the scored frame keeps the confident prediction only, the metrics read no other,
          and the unscored frame keeps empty entries.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()], class_names=("car",), score_threshold=0.2
        )
        sample = frame(0.0)
        weak = frame(0.0)
        weak["pred_scores"] = torch.tensor([0.1])
        merged = {name: torch.cat([sample[name], weak[name]]) for name in sample}

        observed = [torch.ones(2, dtype=torch.bool)] * 2
        suite.update(
            {"motion_frames": [merged, merged], "gt_observed": observed, "scored": [True, False]}
        )

        self.assertEqual([len(entry) for entry in suite.pred_trajectories], [1, 0])
        self.assertAlmostEqual(suite.pred_scores[0].item(), 0.9, places=6)
        self.assertEqual([len(entry) for entry in suite.gt_xy], [2, 0])

    def test_boxes_the_tracking_suite_leaves_out_leave_with_their_predictions(self) -> None:
        """
        Input: a car predicted exactly, a second one at 30 m not observed in its stream yet and a
          third one at 60 m, each with its prediction, cars evaluated up to 50 m.
        Expected: the unobserved and the far car leave with their predictions, as in the tracking
          suite, so EPA scores the first car alone.
        """
        suite = MotionMetricSuite(
            components=[AgentMotionSummary()],
            class_names=("car",),
            score_threshold=0.2,
            max_distances={"car": 50.0},
        )
        frames = [frame(0.0), frame(0.0), frame(0.0)]
        for moved, x in zip(frames[1:], (30.0, 60.0), strict=True):
            moved["pred_xy"] = moved["gt_xy"] = torch.tensor([[x, 0.0]])
        merged = {name: torch.cat([item[name] for item in frames]) for name in frames[0]}

        suite.update(
            {
                "motion_frames": [merged],
                "gt_observed": [torch.tensor([True, False, True])],
                "scored": [True],
            }
        )
        report = suite.components[0].evaluate(
            suite.state_for(None, suite.components[0].filter), EvalStage.VAL
        )

        self.assertEqual(len(suite.gt_xy[0]), 1)
        self.assertEqual(len(suite.pred_xy[0]), 1)
        self.assertEqual(report.metric_value(MetricKey(name="EPA_car")), 1.0)

    def test_a_component_with_a_filter_is_refused(self) -> None:
        """
        Input: an agent motion summary configured with a region filter.
        Expected: ValueError, the suite reports no filtered slice and would label the whole scene.
        """
        with self.assertRaisesRegex(ValueError, "reports no filtered slices"):
            MotionMetricSuite(
                components=[AgentMotionSummary(filter=_Road())],
                class_names=("car",),
                score_threshold=0.2,
            )
