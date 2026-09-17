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
OUTPUT_CHANNELS = POINT_CHANNELS + 6


class SweepSplitVoxelFeatureEncoder(nn.Module):
    """Summarize the current frame and the sweep returns of a voxel apart from each other.

    Averaging every point of a voxel together moves the voxel between where a surface is and
    where it was, which is the position the segmentation head has to label. The encoder keeps
    the current frame returns as the position of the voxel and describes the sweep returns
    relative to it, so the displacement between the two stays readable instead of collapsing
    into one blurred point. A voxel observed only in an earlier sweep falls back to the sweep
    returns and reports their time lag, which tells the network that nothing was measured
    there in the current frame.

    The module has no parameters, so the same graph runs in training and in the exported
    model.
    """

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Reduce padded voxel points to one feature vector per voxel.

        Args:
            voxels: Padded voxel points of shape ``(num_voxels, max_points, channels)`` laid
                out as ``(x, y, z, intensity, time_lag)``, where unused slots are zero.
            num_points: Number of valid points per voxel of shape ``(num_voxels,)``.

        Returns:
            Voxel features of shape ``(num_voxels, 11)``, the position, intensity and time lag
            of the voxel followed by the offset, intensity, share and time lag of its sweep
            returns.
        """
        if voxels.shape[2] != POINT_CHANNELS:
            raise ValueError(
                f"SweepSplitVoxelFeatureEncoder requires points laid out as (x, y, z, "
                f"intensity, time_lag), got {voxels.shape[2]} channels."
            )
        slots = torch.arange(voxels.shape[1], device=voxels.device).unsqueeze(0)
        filled = slots < num_points.long().unsqueeze(1)
        current = filled & (voxels[..., TIME_LAG_COLUMN] == 0)
        sweep = filled & (voxels[..., TIME_LAG_COLUMN] > 0)

        current_count = current.sum(dim=1, keepdim=True).to(voxels.dtype)
        sweep_count = sweep.sum(dim=1, keepdim=True).to(voxels.dtype)
        current_mean = (voxels * current.unsqueeze(-1)).sum(dim=1) / current_count.clamp(min=1.0)
        sweep_mean = (voxels * sweep.unsqueeze(-1)).sum(dim=1) / sweep_count.clamp(min=1.0)

        core = torch.where(current_count > 0, current_mean, sweep_mean)
        has_sweep = sweep_count > 0
        offset = torch.where(
            has_sweep, sweep_mean[:, :3] - core[:, :3], torch.zeros_like(core[:, :3])
        )
        sweep_intensity = torch.where(has_sweep, sweep_mean[:, 3:4], torch.zeros_like(sweep_count))
        sweep_lag = torch.where(has_sweep, sweep_mean[:, 4:5], torch.zeros_like(sweep_count))
        sweep_share = sweep_count / (current_count + sweep_count).clamp(min=1.0)
        return torch.cat([core, offset, sweep_intensity, sweep_share, sweep_lag], dim=1)
