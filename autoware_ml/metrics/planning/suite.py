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

"""Open-loop planning metrics next to the motion metrics.

The plan of the frame's command against the ego future, L2 and collision rate with the future
ground truth boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from autoware_ml.metrics.geometry.reachability import VehicleGeometry
from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange
from autoware_ml.metrics.motion.suite import MotionMetricSuite, MotionFrame, MotionState
from autoware_ml.metrics.eval_output import MetricRequirements
from autoware_ml.metrics.report import MetricReport
from autoware_ml.models.planning.geometry import boxes_overlap


@dataclass(frozen=True)
class MotionPlanningFrame(MotionFrame):
    """Trajectories of one frame with the plan.

    Attributes:
        plan: Planned ego positions ``(T_e, 2)``.
        ego_future: Future ego positions ``(T_e, 2)``.
        ego_future_known: Whether the whole ego future is known.
    """

    plan: np.ndarray
    ego_future: np.ndarray
    ego_future_known: bool


@dataclass(frozen=True)
class MotionPlanningState(MotionState):
    """Frames with their plans and the settings of the planning metrics."""

    step_seconds: float
    vehicle: VehicleGeometry


class PlanningSummary(Metric[MotionPlanningState]):
    """L2 error of the plan and its collision rate at every second of the plan."""

    def evaluate(self, state: MotionPlanningState, stage: EvalStage) -> MetricReport:
        """Score the plans of the frames whose whole ego future is known.

        Args:
            state: Frames of the evaluation.
            stage: Evaluation stage, unused.

        Returns:
            The L2 and collision rate at every whole second and their means.
        """
        frames = [frame for frame in state.frames if frame.ego_future_known]
        report = MetricReport()
        if not frames:
            return report
        errors = np.stack([np.linalg.norm(f.plan - f.ego_future, axis=-1) for f in frames])
        collisions = np.stack([plan_collisions(frame, state) for frame in frames])
        steps_per_second = int(round(1.0 / state.step_seconds))
        report.record("num_planned_frames", float(len(frames)))
        steps = range(steps_per_second, errors.shape[1] + 1, steps_per_second)
        l2 = [float(errors[:, step - 1].mean()) for step in steps]
        collision = [float(collisions[:, :step].any(axis=1).mean()) for step in steps]
        for second, (l2_value, collision_value) in enumerate(zip(l2, collision), start=1):
            report.record(f"L2_{second}s", l2_value)
            report.record(f"collision_{second}s", collision_value)
        report.record("L2", float(np.mean(l2)))
        report.record("collision", float(np.mean(collision)))
        return report


def plan_collisions(frame: MotionPlanningFrame, state: MotionPlanningState) -> np.ndarray:
    """Whether the plan overlaps a ground truth agent at every step ``(T_e,)``.

    The agents keep their current heading along their future positions, steps without a known
    agent position do not count.

    Args:
        frame: Frame with the plan and the ground truth futures.
        state: Settings with the ego vehicle.

    Returns:
        Collision flag of every plan step.
    """
    steps = frame.plan.shape[0]
    plan = torch.from_numpy(frame.plan).float()[None, None]
    ego_size = torch.tensor([state.vehicle.length, state.vehicle.width], dtype=torch.float32)
    collisions = np.zeros(steps, dtype=bool)
    if len(frame.gt_xy) == 0:
        return collisions
    for step in range(steps):
        known = torch.from_numpy(frame.gt_valid[:, : step + 1])
        paths = torch.from_numpy(frame.gt_futures[:, : step + 1]).float()
        overlap = boxes_overlap(
            plan[:, :, : step + 1],
            state.vehicle.center_x,
            ego_size,
            paths[None],
            torch.from_numpy(frame.gt_size).float()[None],
            torch.from_numpy(frame.gt_yaw).float()[None],
            known[None],
        )
        collisions[step] = bool(overlap.any())
    return collisions


class MotionPlanningMetricSuite(MotionMetricSuite):
    """Accumulate the trajectories and the plans of every frame."""

    frame_fields = (*MotionMetricSuite.frame_fields, "plan", "ego_future", "ego_future_valid")

    def __init__(
        self,
        components: list[Metric[MotionPlanningState]],
        class_names: tuple[str, ...],
        score_threshold: float,
        step_seconds: float,
        vehicle: VehicleGeometry,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation state.

        Args:
            components: Injected metrics, as in :class:`MotionMetricSuite`.
            class_names: Class names in label order.
            score_threshold: Detection score from which a prediction counts.
            step_seconds: Time between two trajectory steps.
            vehicle: Body of the ego vehicle, the frame origin at its rear axle.
            **kwargs: Forwarded to :class:`MotionMetricSuite`.
        """
        super().__init__(
            components=components,
            class_names=class_names,
            score_threshold=score_threshold,
            **kwargs,
        )
        self.step_seconds = float(step_seconds)
        self.vehicle = vehicle

    def requirements(self) -> MetricRequirements:
        """The motion frames with the plan and the future of the ego.

        Returns:
            The requirements at this suite instance's stage.
        """
        return MetricRequirements(motion=True, plans=True)

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter
    ) -> MotionPlanningState:
        """Collect the frames with the settings of the metrics, without padding frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            The frames of the evaluation.
        """
        state = super().state_for(metric_range, metric_filter)
        return MotionPlanningState(
            frames=state.frames,
            class_names=state.class_names,
            match_distance=state.match_distance,
            miss_distance=state.miss_distance,
            step_seconds=self.step_seconds,
            vehicle=self.vehicle,
        )

    def frame(self, index: int) -> MotionPlanningFrame:
        """Build the accumulated frame at an index."""
        return MotionPlanningFrame(
            **{name: getattr(self, name)[index].cpu().numpy() for name in self.frame_fields[:-1]},
            ego_future_known=bool(self.ego_future_valid[index].all()),
        )
