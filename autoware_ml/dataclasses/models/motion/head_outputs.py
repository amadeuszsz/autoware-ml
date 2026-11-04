"""
Modules to save raw outputs from a motion head.
"""

from __future__ import annotations

from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, model_validator

import torch

from autoware_ml.dataclasses.models.sample_selection import select_output_samples


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
        return type(self)(**select_output_samples(self, mask))
