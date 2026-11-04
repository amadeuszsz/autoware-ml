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

"""BEV geometry of the motion planner: frame changes and box overlaps along trajectories."""

from __future__ import annotations

import math

from jaxtyping import Float
import torch


def to_frame(points: torch.Tensor, frame_from_ego: torch.Tensor) -> torch.Tensor:
    """Move BEV points of the ego frame into the frame of the point cloud.

    Args:
        points: Points ``(B, ..., 2)`` in the ego frame.
        frame_from_ego: Transform of every sample from the ego frame to the point cloud frame
            ``(B, 4, 4)``, the lidar-space augmentation of a training sample.

    Returns:
        The points in the point cloud frame, shaped like ``points``.
    """
    flat = points.reshape(points.shape[0], math.prod(points.shape[1:-1]), 2)
    linear = frame_from_ego[:, :2, :2].to(points.dtype)
    moved = flat @ linear.transpose(1, 2) + frame_from_ego[:, None, :2, 3].to(points.dtype)
    return moved.reshape(points.shape)


def to_ego(points: torch.Tensor, frame_from_ego: torch.Tensor) -> torch.Tensor:
    """Move BEV points of the point cloud frame back into the ego frame.

    Args:
        points: Points ``(B, ..., 2)`` in the point cloud frame.
        frame_from_ego: Transform of every sample from the ego frame to the point cloud frame
            ``(B, 4, 4)``.

    Returns:
        The points in the ego frame, shaped like ``points``.
    """
    flat = points.reshape(points.shape[0], math.prod(points.shape[1:-1]), 2)
    flat = flat - frame_from_ego[:, None, :2, 3].to(points.dtype)
    inverse = torch.linalg.inv(frame_from_ego[:, :2, :2].double()).to(points.dtype)
    return (flat @ inverse.transpose(1, 2)).reshape(points.shape)


def yaw_to_ego(
    yaw: Float[torch.Tensor, "*items"],
    frame_from_ego: Float[torch.Tensor, "batch_size 4 4"],
) -> Float[torch.Tensor, "*items"]:
    """Turn headings of the point cloud frame back into the ego frame.

    The heading is moved as a direction, so a mirrored frame mirrors it too.

    Args:
        yaw: Headings in the point cloud frame.
        frame_from_ego: Transform of every sample from the ego frame to the point cloud frame.

    Returns:
        The headings in the ego frame, shaped like ``yaw``.
    """
    direction = torch.stack([yaw.cos(), yaw.sin()], dim=-1)
    origin = frame_from_ego.new_zeros(frame_from_ego.shape[0], 4, 4)
    origin[:, :2, :2] = frame_from_ego[:, :2, :2]
    back = to_ego(direction, origin)
    return torch.atan2(back[..., 1], back[..., 0])


def length_to_ego(
    length: Float[torch.Tensor, "*items"],
    frame_from_ego: Float[torch.Tensor, "batch_size 4 4"],
) -> Float[torch.Tensor, "*items"]:
    """Undo the scaling of the point cloud frame on box lengths.

    Args:
        length: Lengths in the point cloud frame.
        frame_from_ego: Transform of every sample from the ego frame to the point cloud frame.

    Returns:
        The lengths in the ego frame, shaped like ``length``.
    """
    scale = torch.linalg.det(frame_from_ego[:, :2, :2].double()).abs().sqrt().to(length.dtype)
    return length / scale.reshape((-1,) + (1,) * (length.dim() - 1))


def boxes_overlap(
    plans: torch.Tensor,
    ego_center_x: float,
    ego_size: torch.Tensor,
    agent_paths: torch.Tensor,
    agent_size: torch.Tensor,
    agent_yaw: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    """Test every plan for an overlap of the ego box with an agent box at a shared step.

    The ego box heads along its path, an agent box keeps its detected heading. Two boxes overlap
    when a corner of one lies inside the other.

    Args:
        plans: Paths of the frame origin ``(B, P, T, 2)``.
        ego_center_x: Distance of the ego box center ahead of the frame origin.
        ego_size: Ego length and width ``(2,)``.
        agent_paths: Agent paths over the same steps ``(B, N, T, 2)``.
        agent_size: Agent length and width ``(B, N, 2)``.
        agent_yaw: Agent heading ``(B, N)``.
        agent_mask: Whether every agent counts at every step ``(B, N, T)``, false where its position
            is unknown.

    Returns:
        Whether every plan collides ``(B, P)``.
    """
    previous = torch.cat([torch.zeros_like(plans[:, :, :1]), plans[:, :, :-1]], dim=2)
    heading = plans - previous
    ego_yaw = torch.atan2(heading[..., 1], heading[..., 0])
    ego_centers = plans + ego_center_x * torch.stack([ego_yaw.cos(), ego_yaw.sin()], dim=-1)
    ego_corners = box_corners(ego_centers, ego_size.expand(plans.shape[:3] + (2,)), ego_yaw)
    agent_corners = box_corners(
        agent_paths,
        agent_size[:, :, None].expand(agent_paths.shape),
        agent_yaw[..., None].expand(agent_paths.shape[:3]),
    )
    # (B, P, N, T): ego corners in agent boxes and agent corners in ego boxes
    ego_in_agent = points_in_boxes(
        ego_corners[:, :, None],
        agent_paths[:, None],
        agent_size[:, None, :, None],
        agent_yaw[:, None, :, None],
    )
    agent_in_ego = points_in_boxes(
        agent_corners[:, None], ego_centers[:, :, None], ego_size, ego_yaw[:, :, None]
    )
    overlap = (ego_in_agent | agent_in_ego) & agent_mask[:, None]
    return overlap.flatten(start_dim=2).any(dim=-1)


def box_corners(center: torch.Tensor, size: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Corners of BEV boxes ``(..., 4, 2)`` from centers ``(..., 2)``, sizes and headings."""
    half = size / 2
    signs = center.new_tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    local = signs * half[..., None, :]
    cos, sin = yaw.cos()[..., None], yaw.sin()[..., None]
    rotated = torch.stack(
        [local[..., 0] * cos - local[..., 1] * sin, local[..., 0] * sin + local[..., 1] * cos],
        dim=-1,
    )
    return rotated + center[..., None, :]


def points_in_boxes(
    points: torch.Tensor, center: torch.Tensor, size: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Whether any of the four points ``(..., 4, 2)`` is inside the box, broadcast over ``...``."""
    offset = points - center[..., None, :]
    cos, sin = yaw.cos()[..., None], yaw.sin()[..., None]
    along = offset[..., 0] * cos + offset[..., 1] * sin
    across = -offset[..., 0] * sin + offset[..., 1] * cos
    half = size / 2
    inside = (along.abs() <= half[..., 0, None]) & (across.abs() <= half[..., 1, None])
    return inside.any(dim=-1)
