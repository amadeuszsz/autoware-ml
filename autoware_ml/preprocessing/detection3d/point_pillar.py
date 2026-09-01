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

"""Voxelization preprocessing for point cloud models."""

from __future__ import annotations

from jaxtyping import Float32, Int32, Int64
import torch
from torch import Tensor

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.ops.voxelization.voxelization import hard_voxelize
from autoware_ml.preprocessing.base import ModelInputs


class PillarInputs(ModelInputs):
    """Voxelized point cloud inputs of one batch.

    Attributes:
      voxels: Padded voxel features.
      num_points: Number of points of every voxel.
      voxel_coords: Voxel coordinates with the batch index in the first column.
      point_voxel_indices: Voxel row of every input point in the concatenated batch order,
        -1 when the point was not assigned to a retained voxel.
      num_dropped_voxels: Occupied voxels discarded by the voxel budget.
    """

    voxels: Float32[Tensor, "num_voxels max_num_points num_features"]
    num_points: Int32[Tensor, " num_voxels"]
    voxel_coords: Int32[Tensor, "num_voxels 4"]
    point_voxel_indices: Int64[Tensor, " total_points"]
    num_dropped_voxels: Int64[Tensor, ""]


class PointPillarPreprocessor:
    """Convert batched point clouds into padded voxels for voxel based models.

    The preprocessor voxelizes each point cloud using hard_voxelize, pads variable size
    voxels to max_num_points, and packages the tensors expected by voxel based detectors and
    segmentors.
    """

    # Add class attributes for type checking
    voxel_size: Float32[Tensor, " 3"]
    point_cloud_range: Float32[Tensor, " 6"]

    def __init__(
        self,
        voxel_size: list[float],
        point_cloud_range: list[float],
        max_num_points: int,
        max_voxels: int,
        eval_max_voxels: int | None = None,
        voxelization_z_order_first: bool = True,
    ) -> None:
        """Initialize the PointPillarPreprocessor.

        Args:
            voxel_size: Voxel size along each axis [dx, dy, dz] in meters.
            point_cloud_range: Spatial range [x_min, y_min, z_min, x_max, y_max, z_max] in
                meters.
            max_num_points: Maximum number of points kept per voxel.
            max_voxels: Maximum number of voxels retained per sample during training.
            eval_max_voxels: Maximum number of voxels retained per sample during evaluation
                and inference. Required before the preprocessor runs in evaluation mode.
            voxelization_z_order_first: Whether to transpose the [x, y, z] voxel coordinates
                to [z, y, x]. Kept for the deployed coordinate convention.
        """
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.eval_max_voxels = eval_max_voxels
        self.voxelization_z_order_first = voxelization_z_order_first

    def __call__(self, batch: Batch, *, is_training: bool) -> PillarInputs:
        """Voxelize the batched point clouds.

        Args:
            batch: Collated typed batch with point clouds.
            is_training: Whether the owning model is in training mode. Selects between the
                max_voxels (training) and eval_max_voxels (evaluation) budgets.

        Returns:
            The voxelized inputs.
        """
        if not is_training and self.eval_max_voxels is None:
            raise ValueError(
                "PointPillarPreprocessor is running in evaluation mode but 'eval_max_voxels' "
                "is not set. Set 'eval_max_voxels' in the data_preprocessing config (use the "
                "same value as 'max_voxels' to keep the training-time budget)."
            )
        if batch.points is None:
            raise ValueError("PointPillarPreprocessor requires a point cloud batch.")
        points_list = batch.points

        device = points_list[0].device
        voxel_size = self.voxel_size.to(device=device)
        point_cloud_range = self.point_cloud_range.to(device=device)

        # Concat all points across the batch to a single tensor for voxelization, but keep
        # track of the batch index
        points = torch.cat(points_list, dim=0)
        points_batch_indices = torch.cat(
            [
                torch.full((p.shape[0],), i, device=device, dtype=torch.int32)
                for i, p in enumerate(points_list)
            ],
            dim=0,
        )
        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels if is_training else self.eval_max_voxels,
        )

        # Handle the case where no voxels are generated
        if not len(voxels_data.voxels):
            return PillarInputs(
                voxels=points.new_zeros((0, self.max_num_points, points.shape[1])),
                num_points=torch.zeros((0,), device=points.device, dtype=torch.int32),
                voxel_coords=torch.zeros((0, 4), device=points.device, dtype=torch.int32),
                point_voxel_indices=voxels_data.point_voxel_indices,
                num_dropped_voxels=voxels_data.num_dropped_voxels,
            )

        # Concat batch column to the voxel coordinates
        batch_coords = torch.cat(
            [voxels_data.batch_indices.unsqueeze(1), voxels_data.coords], dim=1
        )
        if self.voxelization_z_order_first:
            # Transpose [x, y, z] to [z, y, x], the coordinate order of the deployed engines
            batch_coords = batch_coords[:, [0, 3, 2, 1]].contiguous()

        return PillarInputs(
            voxels=voxels_data.voxels,
            num_points=voxels_data.num_points,
            voxel_coords=batch_coords,
            point_voxel_indices=voxels_data.point_voxel_indices,
            num_dropped_voxels=voxels_data.num_dropped_voxels,
        )
