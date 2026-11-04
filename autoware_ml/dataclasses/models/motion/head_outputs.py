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

"""
Modules to save raw outputs from a motion planning head.
"""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Bool, Float, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class PlanLayerOutputs(NamedTuple):
    """
    Scores one planning layer gave the vocabulary entries it received.

    Attributes:
      path_logits: Score logit of every path the layer refined.
      path_ids: Vocabulary index of the same paths.
      speed_logits: Score logit of every speed profile the layer refined.
      speed_ids: Vocabulary index of the same speed profiles.
      trajectory_logits: Score logit of every trajectory the layer composed from its best paths
        and speed profiles.
      trajectory_ids: Index of the same trajectories, path index times the speed vocabulary size
        plus speed index.
    """

    path_logits: Float[torch.Tensor, "batch_size num_paths"]
    path_ids: Int64[torch.Tensor, "batch_size num_paths"]
    speed_logits: Float[torch.Tensor, "batch_size num_speeds"]
    speed_ids: Int64[torch.Tensor, "batch_size num_speeds"]
    trajectory_logits: Float[torch.Tensor, "batch_size num_trajectories"]
    trajectory_ids: Int64[torch.Tensor, "batch_size num_trajectories"]

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> PlanLayerOutputs:
        """Keep the scores of the samples a mask selects."""
        return PlanLayerOutputs(*(field[mask] for field in self))


class MotionHeadOutputs(BaseModel):
    """
    Dataclass to save the raw outputs of a motion head.

    Agent trajectories are BEV positions at every future step in the frame of the (augmented)
    point cloud, relative to the box center of their detection query.

    Attributes:
      agent_mode_logits: Score logit of every mode of every detection query.
      agent_trajectories: Trajectory of every mode of every detection query.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    agent_mode_logits: Float[torch.Tensor, "batch_size num_queries agent_modes"]
    agent_trajectories: Float[torch.Tensor, "batch_size num_queries agent_modes agent_steps 2"]

    @model_validator(mode="after")
    def _check_agent_shapes(self) -> MotionHeadOutputs:
        """Check that the scores and the trajectories describe the same modes."""
        if tuple(self.agent_trajectories.shape[:3]) != tuple(self.agent_mode_logits.shape):
            raise ValueError(
                "agent_trajectories must lead with the shape of agent_mode_logits "
                f"{tuple(self.agent_mode_logits.shape)}, got "
                f"{tuple(self.agent_trajectories.shape)}."
            )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> MotionHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        return MotionHeadOutputs(
            agent_mode_logits=self.agent_mode_logits[mask],
            agent_trajectories=self.agent_trajectories[mask],
        )


class MotionPlanningHeadOutputs(MotionHeadOutputs):
    """
    Dataclass to save the raw outputs of a motion planning head.

    Plan trajectories are BEV positions of the ego frame origin in the frame of the (augmented)
    point cloud.

    Attributes:
      ego_status: Predicted ego velocity (x, y), acceleration (x, y) and yaw rate.
      plan_logits: Score logit of every trajectory the last planning layer kept.
      plan_trajectories: The same trajectories.
      plan_collision_logits: Logit of every such trajectory colliding with an agent.
      plan_layers: Scores of every planning layer, the training targets of the planner.
    """

    ego_status: Float[torch.Tensor, "batch_size status_dims"]
    plan_logits: Float[torch.Tensor, "batch_size num_plans"]
    plan_trajectories: Float[torch.Tensor, "batch_size num_plans ego_steps 2"]
    plan_collision_logits: Float[torch.Tensor, "batch_size num_plans"]
    plan_layers: tuple[PlanLayerOutputs, ...]

    @model_validator(mode="after")
    def _check_plan_shapes(self) -> MotionPlanningHeadOutputs:
        """Check that the scores and the trajectories describe the same plans."""
        plans = tuple(self.plan_logits.shape)
        if (
            tuple(self.plan_trajectories.shape[:2]) != plans
            or tuple(self.plan_collision_logits.shape) != plans
        ):
            raise ValueError(
                f"plan_trajectories and plan_collision_logits must lead with the shape {plans} of "
                f"plan_logits, got {tuple(self.plan_trajectories.shape)} and "
                f"{tuple(self.plan_collision_logits.shape)}."
            )
        if not self.plan_layers:
            raise ValueError("The planner needs at least one planning layer.")
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> MotionPlanningHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        return MotionPlanningHeadOutputs(
            agent_mode_logits=self.agent_mode_logits[mask],
            agent_trajectories=self.agent_trajectories[mask],
            ego_status=self.ego_status[mask],
            plan_logits=self.plan_logits[mask],
            plan_trajectories=self.plan_trajectories[mask],
            plan_collision_logits=self.plan_collision_logits[mask],
            plan_layers=tuple(layer.select_samples(mask) for layer in self.plan_layers),
        )
