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

"""Future of the ego vehicle, the targets of the planner.

A sample holds the ego future in the frame of its raw point cloud. Collation moves it into the
frame of the augmented cloud with the lidar-space augmentation of the sample.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Sequence

from jaxtyping import Bool, Float32
import torch
from torch import Tensor

from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.geometry.utils import transform_points


class PlanningGTSample(NamedTuple):
    """Ego future of one frame, in the frame of its raw point cloud.

    Attributes:
        ego_future_positions: Origin of the frame at every future step of the ego horizon.
        ego_future_valid: Whether the scene holds a frame at the future step.
        ego_status: Ego velocity (x, y), acceleration (x, y) and yaw rate, from the past poses.
        ego_status_valid: Whether the scene holds the past frames the status needs.
        ego_path: Origin of the frame along the future ego path at fixed arc lengths.
        ego_path_valid: Whether the ego travels as far as the arc length of the path point.
    """

    ego_future_positions: Float32[Tensor, "ego_steps 3"]
    ego_future_valid: Bool[Tensor, " ego_steps"]
    ego_status: Float32[Tensor, " 5"]
    ego_status_valid: bool
    ego_path: Float32[Tensor, "path_points 3"]
    ego_path_valid: Bool[Tensor, " path_points"]


class PlanningGTBatch(NamedTuple):
    """Ego futures of a batch, in the frames of the augmented point clouds.

    Attributes:
        ego_future_positions: BEV origin of every frame at every future step.
        ego_future_valid: Whether the scene holds a frame at the future step.
        ego_status: Ego velocity (x, y), acceleration (x, y) and yaw rate of every frame.
        ego_status_valid: Whether the status of a frame is known.
        ego_path: BEV origin of every frame along its future path at fixed arc lengths.
        ego_path_valid: Whether the ego travels as far as every path point.
    """

    ego_future_positions: Float32[Tensor, "batch_size ego_steps 2"]
    ego_future_valid: Bool[Tensor, "batch_size ego_steps"]
    ego_status: Float32[Tensor, "batch_size 5"]
    ego_status_valid: Bool[Tensor, " batch_size"]
    ego_path: Float32[Tensor, "batch_size path_points 2"]
    ego_path_valid: Bool[Tensor, "batch_size path_points"]

    @staticmethod
    def collate_gt_samples(
        planning_gt_samples: Sequence[PlanningGTSample],
        lidar_transformation_samples: Sequence[LiDARTransformationSample | None],
    ) -> PlanningGTBatch:
        """Move the ego futures into the augmented frames.

        Args:
            planning_gt_samples: Ego futures of every sample, in batch order.
            lidar_transformation_samples: Lidar-space augmentation of every sample, None for
                samples that were not augmented.

        Returns:
            PlanningGTBatch: The collated ego futures.

        Raises:
            ValueError: If the sequences disagree on the batch size.
        """
        if len(planning_gt_samples) != len(lidar_transformation_samples):
            raise ValueError(
                f"Got {len(planning_gt_samples)} planning samples and "
                f"{len(lidar_transformation_samples)} transformations."
            )
        matrices = [
            torch.eye(4) if transformation is None else transformation.transformation_matrix.float()
            for transformation in lidar_transformation_samples
        ]
        samples = planning_gt_samples
        return PlanningGTBatch(
            ego_future_positions=torch.stack(
                [
                    transform_points(sample.ego_future_positions, matrix)[..., :2]
                    for sample, matrix in zip(samples, matrices, strict=True)
                ]
            ),
            ego_future_valid=torch.stack([sample.ego_future_valid for sample in samples]),
            ego_status=torch.stack(
                [
                    transform_status(sample.ego_status, matrix)
                    for sample, matrix in zip(samples, matrices, strict=True)
                ]
            ),
            ego_status_valid=torch.tensor(
                [sample.ego_status_valid for sample in samples], dtype=torch.bool
            ),
            ego_path=torch.stack(
                [
                    transform_points(sample.ego_path, matrix)[..., :2]
                    for sample, matrix in zip(samples, matrices, strict=True)
                ]
            ),
            ego_path_valid=torch.stack([sample.ego_path_valid for sample in samples]),
        )

    def select_samples(self, mask: Bool[Tensor, " batch_size"]) -> PlanningGTBatch:
        """Keep the ego futures of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            PlanningGTBatch: The ego futures of the selected samples.
        """
        return PlanningGTBatch(*(field[mask] for field in self))

    def to_device(self, device: torch.device) -> PlanningGTBatch:
        """Move every tensor to ``device``.

        Args:
            device: Target device.

        Returns:
            PlanningGTBatch: The batch on ``device``.
        """
        return PlanningGTBatch(*(field.to(device) for field in self))


def transform_status(status: Float32[Tensor, " 5"], matrix: Float32[Tensor, "4 4"]) -> Tensor:
    """Rotate and scale the ego velocity and acceleration with a lidar-space augmentation.

    A yaw rotation and a uniform scale keep the yaw rate, a mirror flips its sign.

    Args:
        status: Ego velocity (x, y), acceleration (x, y) and yaw rate in the source frame.
        matrix: Augmentation from the source to the target frame.

    Returns:
        The status in the target frame.
    """
    linear = matrix[:2, :2]
    handedness = torch.sign(torch.linalg.det(linear))
    return torch.cat([linear @ status[0:2], linear @ status[2:4], status[4:5] * handedness])
