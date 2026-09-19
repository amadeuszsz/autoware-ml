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

"""Voxel feature encoders feeding the PTv3 embedding stem."""

from __future__ import annotations

import torch
import torch.nn as nn

TIME_LAG_COLUMN = 4
POINT_CHANNELS = TIME_LAG_COLUMN + 1
# Channels describing the returns of one side: the offset from the voxel position, their
# intensity, their share of the voxel and their mean time lag
SIDE_CHANNELS = 6
OUTPUT_CHANNELS = POINT_CHANNELS + 2 * SIDE_CHANNELS


class SweepSplitVoxelFeatureEncoder(nn.Module):
    """Summarize the current frame, the past and the future returns of a voxel apart.

    Averaging every point of a voxel together moves the voxel between where a surface is and
    where it was, which is the position the segmentation head has to label. The encoder keeps
    the current frame returns as the position of the voxel and describes the returns of either
    side relative to it, so the displacement between them stays readable instead of collapsing
    into one blurred point. Past and future returns are kept apart for the same reason: a
    surface passing through the voxel leaves them on opposite sides, and one mean over both
    would cancel the motion it is meant to expose. A voxel the current frame never observed
    falls back to the returns around it and reports their time lag, which tells the network
    that nothing was measured there at the moment it labels.

    The module has no parameters, so the same graph runs in training and in the exported
    model.
    """

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Reduce padded voxel points to one feature vector per voxel.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)`` laid
                out as ``(x, y, z, intensity, time_lag)``, where unused slots are zero. The
                time lag is the capture time of the sample minus the one of the point, so it is
                zero for the current frame, positive for a past sweep and negative for a future
                one.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Voxel features of shape ``(num_voxels, 17)``, the position, intensity and time lag
            of the voxel followed by the offset, intensity, share and time lag of its past
            returns and then the same four for its future returns.
        """
        if voxels.shape[2] != POINT_CHANNELS:
            raise ValueError(
                f"SweepSplitVoxelFeatureEncoder requires points laid out as (x, y, z, "
                f"intensity, time_lag), got {voxels.shape[2]} channels."
            )
        slots = torch.arange(voxels.shape[1], device=voxels.device).unsqueeze(0)
        filled = slots < num_points.long().unsqueeze(1)
        time_lag = voxels[..., TIME_LAG_COLUMN]
        current = filled & (time_lag == 0)
        past = filled & (time_lag > 0)
        future = filled & (time_lag < 0)

        current_count, current_mean = self._summarize(voxels, current)
        past_count, past_mean = self._summarize(voxels, past)
        future_count, future_mean = self._summarize(voxels, future)

        # A voxel the current frame never observed is placed where the sweeps around it saw a
        # surface, the nearer side in time first
        core = torch.where(
            current_count > 0,
            current_mean,
            torch.where(past_count > 0, past_mean, future_mean),
        )
        total_count = (current_count + past_count + future_count).clamp(min=1.0)
        return torch.cat(
            [
                core,
                self._describe(core, past_mean, past_count, total_count),
                self._describe(core, future_mean, future_count, total_count),
            ],
            dim=1,
        )

    @staticmethod
    def _summarize(
        voxels: torch.Tensor, selected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count the selected points of every voxel and average their features.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)``.
            selected: Mask of the points to summarize, of shape ``(num_voxels, max_points)``.

        Returns:
            The count of shape ``(num_voxels, 1)`` and the mean of shape
            ``(num_voxels, channels)``, the mean being zero where nothing was selected.
        """
        count = selected.sum(dim=1, keepdim=True).to(voxels.dtype)
        mean = (voxels * selected.unsqueeze(-1)).sum(dim=1) / count.clamp(min=1.0)
        return count, mean

    @staticmethod
    def _describe(
        core: torch.Tensor,
        side_mean: torch.Tensor,
        side_count: torch.Tensor,
        total_count: torch.Tensor,
    ) -> torch.Tensor:
        """Describe the returns of one side relative to the position of the voxel.

        Args:
            core: Voxel features of shape ``(num_voxels, channels)``.
            side_mean: Mean features of this side, of shape ``(num_voxels, channels)``.
            side_count: Number of returns on this side, of shape ``(num_voxels, 1)``.
            total_count: Number of returns in the voxel, of shape ``(num_voxels, 1)``.

        Returns:
            Features of shape ``(num_voxels, 6)``, the offset from the voxel position, the
            intensity, the share and the mean time lag of the side, all zero for a side that
            observed nothing.
        """
        observed = side_count > 0
        offset = torch.where(
            observed, side_mean[:, :3] - core[:, :3], torch.zeros_like(core[:, :3])
        )
        intensity = torch.where(observed, side_mean[:, 3:4], torch.zeros_like(side_count))
        lag = torch.where(observed, side_mean[:, 4:5], torch.zeros_like(side_count))
        return torch.cat([offset, intensity, side_count / total_count, lag], dim=1)
