"""Future trajectories of the annotated agents.

A sample holds the futures in the frame of its raw point cloud. Collation moves them into the
frame of the augmented cloud with the lidar-space augmentation of the sample, the transform the
points and boxes went through, and aligns the agent futures with the collated 3D boxes by
instance identity.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Bool, Float32, Int64
import torch
from torch import Tensor

from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.geometry.utils import transform_points


class MotionGTSample(NamedTuple):
    """Agent futures of one frame, in the frame of its raw point cloud.

    Attributes:
        agent_instance_ids: Identity of every agent of the frame, as the 3D boxes carry it.
        agent_future_positions: Box center of every agent at every future step.
        agent_future_valid: Whether the agent is annotated at the future step.
    """

    agent_instance_ids: Int64[Tensor, " num_agents"]
    agent_future_positions: Float32[Tensor, "num_agents agent_steps 3"]
    agent_future_valid: Bool[Tensor, "num_agents agent_steps"]


class MotionGTBatch(NamedTuple):
    """Agent futures of a batch, in the frames of the augmented point clouds.

    Attributes:
        agent_future_positions: BEV center of every padded 3D box slot at every future step.
        agent_future_valid: Whether a slot holds a box annotated at the future step. False for
            padding slots and for boxes the motion sample does not know.
    """

    agent_future_positions: Float32[Tensor, "batch_size max_num_gt_bboxes agent_steps 2"]
    agent_future_valid: Bool[Tensor, "batch_size max_num_gt_bboxes agent_steps"]

    @staticmethod
    def collate_gt_samples(
        motion_gt_samples: Sequence[MotionGTSample],
        gt_bboxes_instance_ids: Int64[Tensor, "batch_size max_num_gt_bboxes"],
        lidar_transformation_samples: Sequence[LiDARTransformationSample | None],
    ) -> MotionGTBatch:
        """Move the futures into the augmented frames and align the agents with the boxes.

        Args:
            motion_gt_samples: Futures of every sample, in batch order.
            gt_bboxes_instance_ids: Identity of every padded box slot of the collated 3D boxes,
                -1 for padding.
            lidar_transformation_samples: Lidar-space augmentation of every sample, None for
                samples that were not augmented.

        Returns:
            MotionGTBatch: The collated futures.

        Raises:
            ValueError: If the sequences and the box slots disagree on the batch size.
        """
        batch_size = len(motion_gt_samples)
        if batch_size != len(lidar_transformation_samples) or batch_size != len(
            gt_bboxes_instance_ids
        ):
            raise ValueError(
                f"Got {batch_size} motion samples, {len(lidar_transformation_samples)} "
                f"transformations and {len(gt_bboxes_instance_ids)} box rows."
            )
        num_slots = gt_bboxes_instance_ids.shape[1]
        agent_steps = motion_gt_samples[0].agent_future_positions.shape[1]
        agent_positions = torch.zeros(batch_size, num_slots, agent_steps, 2)
        agent_valid = torch.zeros(batch_size, num_slots, agent_steps, dtype=torch.bool)
        for index, (sample, transformation) in enumerate(
            zip(motion_gt_samples, lidar_transformation_samples, strict=True)
        ):
            matrix = (
                torch.eye(4)
                if transformation is None
                else transformation.transformation_matrix.float()
            )
            # Slots of the boxes the motion sample knows, -1 padding never matches
            slot_ids = gt_bboxes_instance_ids[index]
            matches = slot_ids[:, None] == sample.agent_instance_ids[None, :]
            slots, agents = matches.nonzero(as_tuple=True)
            agent_positions[index, slots] = transform_points(
                sample.agent_future_positions[agents], matrix
            )[..., :2]
            agent_valid[index, slots] = sample.agent_future_valid[agents]
        return MotionGTBatch(agent_future_positions=agent_positions, agent_future_valid=agent_valid)

    def select_samples(self, mask: Bool[Tensor, " batch_size"]) -> MotionGTBatch:
        """Keep the futures of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            MotionGTBatch: The futures of the selected samples.
        """
        return MotionGTBatch(*(field[mask] for field in self))

    def to_device(self, device: torch.device) -> MotionGTBatch:
        """Move every tensor to ``device``.

        Args:
            device: Target device.

        Returns:
            MotionGTBatch: The batch on ``device``.
        """
        return MotionGTBatch(*(field.to(device) for field in self))
