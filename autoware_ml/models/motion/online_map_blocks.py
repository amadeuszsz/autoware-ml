# Copyright (c) Horizon Robotics. All rights reserved.
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

"""Query blocks of a polyline map head: anchor embedding, keypoints and refinement.

A polyline anchor holds the BEV points of one element in meters, flattened to
``(x_0, y_0, x_1, y_1, ...)`` (Sun et al., SparseDrive, 2024).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d import InstanceBank, linear_relu_ln


class SparsePolylineEncoder(nn.Module):
    """Embed polyline anchors into positional features."""

    def __init__(self, embed_dims: int, num_points: int) -> None:
        """Initialize the encoder.

        Args:
            embed_dims: Output embedding dimension.
            num_points: Points of every polyline.
        """
        super().__init__()
        self.pos_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 2 * num_points))

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        """Encode anchors ``(..., 2 * num_points)`` into embeddings ``(..., C)``."""
        return self.pos_fc(anchor)


class SparsePolylineKeyPointsGenerator(nn.Module):
    """Place learned keypoints around every polyline point at fixed heights above the ground."""

    def __init__(
        self,
        *,
        embed_dims: int,
        num_points: int,
        fix_heights: Sequence[float],
        num_learnable_pts: int,
    ) -> None:
        """Initialize the generator.

        Args:
            embed_dims: Instance feature dimension driving the offsets.
            num_points: Points of every polyline.
            fix_heights: Heights of the keypoint layers in the frame.
            num_learnable_pts: Keypoints per polyline point and height.

        Raises:
            ValueError: If there is no height or no keypoint per point.
        """
        super().__init__()
        if not fix_heights or num_learnable_pts < 1:
            raise ValueError(
                f"Polyline keypoints need heights and learnable points, got {fix_heights} and "
                f"{num_learnable_pts}."
            )
        self.num_points = num_points
        self.num_learnable_pts = num_learnable_pts
        self.register_buffer(
            "fix_heights", torch.tensor(list(fix_heights), dtype=torch.float32), persistent=False
        )
        self.num_pts = num_points * len(fix_heights) * num_learnable_pts
        self.learnable_fc = nn.Linear(embed_dims, self.num_pts * 2)
        nn.init.xavier_uniform_(self.learnable_fc.weight)
        nn.init.zeros_(self.learnable_fc.bias)

    def forward(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
        """Generate keypoints ``(B, N, P, 3)`` from anchors ``(B, N, 2 * num_points)``."""
        batch_size, num_anchor = anchor.shape[:2]
        num_heights = self.fix_heights.shape[0]
        offsets = self.learnable_fc(instance_feature).reshape(
            batch_size, num_anchor, self.num_points, num_heights, self.num_learnable_pts, 2
        )
        points = anchor.reshape(batch_size, num_anchor, self.num_points, 1, 1, 2) + offsets
        heights = self.fix_heights[:, None, None].expand(num_heights, self.num_learnable_pts, 1)
        key_points = torch.cat([points, heights.expand(*points.shape[:-1], 1)], dim=-1)
        return key_points.flatten(2, 4)


class SparsePolylineRefinementModule(nn.Module):
    """Refine polyline anchors and classify them."""

    def __init__(self, *, embed_dims: int, num_points: int, num_classes: int) -> None:
        """Initialize the module.

        Args:
            embed_dims: Instance feature dimension.
            num_points: Points of every polyline.
            num_classes: Number of map classes.
        """
        super().__init__()
        self.refine_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2, embed_dims),
            nn.Linear(embed_dims, 2 * num_points),
        )
        self.cls_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, embed_dims),
            nn.Linear(embed_dims, num_classes),
        )
        nn.init.constant_(self.cls_layers[-1].bias, -math.log((1.0 - 0.01) / 0.01))

    def forward(
        self, instance_feature: torch.Tensor, anchor: torch.Tensor, anchor_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine anchors and classify the instances.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Polyline anchors ``(B, N, 2 * num_points)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.

        Returns:
            Refined anchors ``(B, N, 2 * num_points)`` and class logits ``(B, N, num_classes)``.
        """
        refined = anchor + self.refine_layers(instance_feature + anchor_embed)
        return refined, self.cls_layers(instance_feature)


class PolylineInstanceBank(InstanceBank):
    """Instance bank of polyline anchors, which move between frames with the frame alone."""

    def __init__(self, *, num_points: int, **kwargs: object) -> None:
        """Initialize the bank.

        Args:
            num_points: Points of every polyline, the anchors are ``(num_anchor, 2 * num_points)``.
            **kwargs: Remaining arguments of :class:`InstanceBank`.
        """
        super().__init__(anchor_dim=2 * num_points, **kwargs)

    @staticmethod
    def project_anchor(
        anchor: torch.Tensor, time_interval: torch.Tensor, transform: torch.Tensor
    ) -> torch.Tensor:
        """Move cached polylines into the current frame, map elements stand still.

        Args:
            anchor: Cached polylines ``(B, K, 2 * num_points)``.
            time_interval: Frame gaps ``(B,)`` in seconds, unused.
            transform: Frame to frame transforms ``(B, 4, 4)``.

        Returns:
            The polylines in the current frame.
        """
        points = anchor.unflatten(-1, (-1, 2))
        moved = (
            points @ transform[:, None, :2, :2].transpose(-1, -2) + transform[:, None, None, :2, 3]
        )
        return moved.flatten(-2)
