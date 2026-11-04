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

"""Driving commands the ego planner follows, the route intent of a frame."""

from __future__ import annotations

from enum import IntEnum

import torch


class DrivingCommand(IntEnum):
    """High-level command the planner follows."""

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


def driving_commands(
    ego_future: torch.Tensor, ego_future_valid: torch.Tensor, lateral_threshold: float
) -> torch.Tensor:
    """Derive the command of every frame from the lateral offset of its last known future position.

    Args:
        ego_future: Ego positions in the ego frame ``(B, T, 2)``, x forward and y left.
        ego_future_valid: Whether every future position is known ``(B, T)``.
        lateral_threshold: Lateral offset from which the ego turns.

    Returns:
        Command of every frame ``(B,)``, straight for a frame without a known future.
    """
    steps = torch.arange(1, ego_future_valid.shape[1] + 1, device=ego_future_valid.device)
    last = (ego_future_valid.long() * steps).argmax(dim=1)
    lateral = ego_future[torch.arange(ego_future.shape[0], device=ego_future.device), last, 1]
    lateral = torch.where(ego_future_valid.any(dim=1), lateral, torch.zeros_like(lateral))
    commands = torch.full_like(lateral, int(DrivingCommand.STRAIGHT), dtype=torch.long)
    commands[lateral >= lateral_threshold] = int(DrivingCommand.LEFT)
    commands[lateral <= -lateral_threshold] = int(DrivingCommand.RIGHT)
    return commands
