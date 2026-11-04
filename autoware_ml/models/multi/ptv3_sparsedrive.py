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

"""PTv3 segmentation and Sparse4D detection with the SparseDrive motion planner.

The planner refines an ego query next to the agents of the motion head and plans the ego vehicle
from it and the fused lidar stages. Every frame plans toward the driving command its ground truth
ego future takes, as the route gives it on the vehicle.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.planning import PlanningGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Sparse4DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.motion.head_outputs import MotionPlanningHeadOutputs
from autoware_ml.models.detection3d.heads.sparse4d import Sparse4DHead, Sparse4DTargets
from autoware_ml.metrics.eval_output import Detection3DEvalOutput
from autoware_ml.metrics.motion.eval_output import MotionEvalFrame
from autoware_ml.models.detection3d.ptv3 import (
    PTv3Det3DFeaturePyramid,
    SparseStage,
    det3d_head_export_args,
    det3d_head_export_dynamic_axes,
    det3d_head_export_input_names,
)
from autoware_ml.models.motion.online_map_head import OnlineMapHead
from autoware_ml.models.planning.commands import driving_commands
from autoware_ml.models.planning.geometry import to_ego
from autoware_ml.models.motion.instance_queue import QueueState
from autoware_ml.models.motion.motion_head import MapContext
from autoware_ml.models.planning.motion_planning_head import (
    MOTION_PLANNING_STATE_INPUT_NAMES,
    MOTION_PLANNING_STATE_OUTPUT_NAMES,
    MotionPlanningHead,
)
from autoware_ml.models.multi.detection3d_frames import detection3d_gt_batch
from autoware_ml.models.multi.ptv3_det3d_seg3d_sparse4d import fuse_export_stages
from autoware_ml.models.multi.ptv3_sparsedrive_motion import (
    PTv3SparseDriveMotionModel,
    export_map_context,
    map_export_inputs,
)
from autoware_ml.models.segmentation3d.ptv3_base import PTv3BaseModel, PTv3ExportContext
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import offset_to_batch

# The planner graph takes the encoder stages of the detection graph first, then these
MOTION_PLANNING_EXPORT_INPUT_NAMES = (
    "instance_feature",
    "prediction",
    "classification",
    "quality",
    *MOTION_PLANNING_STATE_INPUT_NAMES,
    # Slot index the detection graph returned for the last frame
    "carried_index",
    "temp_valid",
    "prev2current",
    "command",
)
MOTION_PLANNING_EXPORT_OUTPUT_NAMES = (
    "agent_mode_logits",
    "agent_trajectories",
    "plan_logits",
    "plan_trajectories",
    "plan_collision_logits",
    "ego_status",
    "plan",
    *MOTION_PLANNING_STATE_OUTPUT_NAMES,
)


class PTv3SparseDriveModel(PTv3SparseDriveMotionModel):
    """PTv3 joint segmentation and Sparse4D detection with the SparseDrive motion planner."""

    def __init__(self, *, motion_head: MotionPlanningHead, **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            motion_head: Motion planner reading the detection queries.
            **kwargs: Arguments of :class:`PTv3SparseDriveMotionModel`.

        Raises:
            ValueError: If the planner samples other stages than the coarsest the detection head
                samples.
        """
        super().__init__(motion_head=motion_head, **kwargs)
        planner_grids = [
            (grid.grid_dims, grid.voxel_size) for grid in motion_head.planner.level_grids
        ]
        det_grids = [(grid.grid_dims, grid.voxel_size) for grid in self.bbox_head.level_grids]
        if planner_grids != det_grids[len(det_grids) - len(planner_grids) :]:
            raise ValueError(
                f"The planner samples the stages {planner_grids}, they must be the coarsest of the "
                f"stages the detection head samples {det_grids}."
            )

    def forward_motion(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        stages: Sequence[SparseStage],
        det3d_outputs: Sparse4DHeadOutputs,
        carried_index: torch.Tensor | None,
        map_context: MapContext | None,
    ) -> MotionPlanningHeadOutputs:
        """Run the motion planner on the last layer detection queries and the fused stages.

        Args:
            multi_task_batch_inputs: Model inputs with the frame metadata and the ego futures.
            stages: Fused stages the detection head sampled.
            det3d_outputs: Outputs of the detection head.
            carried_index: Slot index of the previous frame, None at the first frame.
            map_context: Map queries of a planner with a map.

        Returns:
            The trajectories and the plans.
        """
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        planning_gt = self.planning_gt(multi_task_batch_inputs)
        return self.motion_head(
            det3d_outputs.instance_feature,
            det3d_outputs.prediction[:, -1],
            det3d_outputs.classification[:, -1],
            self.bbox_head.anchor_encoder,
            pooled_stage(stages[-1]),
            self.bbox_head.instance_bank.lane_continues,
            carried_index,
            frame_meta.ego2globals,
            self.planner_stages(stages),
            self.driving_commands(planning_gt, frame_meta.lidar_augmentations),
            frame_meta.lidar_augmentations,
            map_context,
        )

    def planner_stages(self, stages: Sequence[SparseStage]) -> list[SparseStage]:
        """The coarsest fused stages, the ones the planner samples."""
        return list(stages[len(stages) - len(self.motion_head.planner.level_grids) :])

    def planning_gt(self, multi_task_batch_inputs: ModelBatchInputs) -> PlanningGTBatch:
        """Ego futures of the batch, the planner follows their command and learns them.

        Args:
            multi_task_batch_inputs: Model inputs with the ego futures.

        Returns:
            The ego futures of the batch.

        Raises:
            ValueError: If the batch carries no ego futures.
        """
        planning_gt = multi_task_batch_inputs.multi_task_gt_batch.planning_gt_batch
        if planning_gt is None:
            raise ValueError("The planner needs the ego future of every sample.")
        return planning_gt

    def driving_commands(
        self, planning_gt: PlanningGTBatch, frame_from_ego: torch.Tensor
    ) -> torch.Tensor:
        """Command of every frame from its ground truth ego future in the ego frame.

        Args:
            planning_gt: Ego futures of the frames.
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame.

        Returns:
            Command of every frame ``(B,)``.
        """
        return driving_commands(
            to_ego(planning_gt.ego_future_positions, frame_from_ego),
            planning_gt.ego_future_valid,
            self.motion_head.command_lateral_threshold,
        )

    def motion_loss(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        motion_outputs: MotionPlanningHeadOutputs,
        det3d_targets: list[Sparse4DTargets],
        det3d_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Supervise the agents and the planner on the frames with boxes.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            motion_outputs: Raw outputs of the motion planner.
            det3d_targets: Detection targets of every decoder layer over the frames with boxes.
            det3d_mask: Mask of the frames with boxes.

        Returns:
            The motion planning loss terms and their sum under ``loss``.

        Raises:
            ValueError: If the batch carries no agent or ego futures.
        """
        targets = self.agent_future_targets(multi_task_batch_inputs)
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        return self.motion_head.loss(
            motion_outputs.select_samples(det3d_mask),
            det3d_targets[-1].assigned_instance,
            gt_detections.gt_bboxes_3d[det3d_mask],
            gt_detections.gt_bboxes_instance_ids[det3d_mask],
            targets.select_samples(det3d_mask),
            self.planning_gt(multi_task_batch_inputs).select_samples(det3d_mask),
            frame_meta.lidar_augmentations[det3d_mask],
        )

    def motion_eval_frames(
        self,
        batch: ModelBatchInputs,
        outputs: ModelOutputs,
        detection: Detection3DEvalOutput | None,
    ) -> list[MotionEvalFrame]:
        """Pair the trajectories and the plan of every frame with the futures.

        Args:
            batch: Model inputs with the ground truth boxes and futures.
            outputs: Raw outputs from :meth:`forward`.
            detection: Detection frames of the batch, see the motion model.

        Returns:
            One motion frame per sample of the batch, with the plan and the ego future.
        """
        frames = super().motion_eval_frames(batch, outputs, detection)
        planning_gt = self.planning_gt(batch)
        det3d_outputs = outputs.detection3d().sparse4d_head()
        scores = self.bbox_head.query_scores(det3d_outputs).detach().max(dim=-1).values
        anchor = det3d_outputs.prediction[:, -1].detach()
        plans = self.motion_head.select_plan(outputs.motion(), anchor, scores)
        return [
            MotionEvalFrame(
                **{
                    **dict(frame),
                    "plan": plans[index].detach(),
                    "ego_future": planning_gt.ego_future_positions[index],
                    "ego_future_valid": planning_gt.ego_future_valid[index],
                }
            )
            for index, frame in enumerate(frames)
        ]

    def build_motion_export_spec(
        self,
        context: PTv3ExportContext,
        bbox_head: Sparse4DHead,
        online_map_head: OnlineMapHead | None,
    ) -> ExportSpec:
        """Build the motion planner export spec, one frame with the history at the graph boundary.

        Args:
            context: Shared export context.
            bbox_head: Export-prepared detection head of the detection graph.
            online_map_head: Export-prepared map head copy sizing the map context, exactly for a
                planner with a map.

        Returns:
            Export spec whose graph takes the encoder stages the detection graph takes, the
            detection outputs of the frame, the history of the last frame, the slot index the
            detection graph returned for the last frame, the stream continuity, the frame to
            frame transform, the driving command and, with a map, the context outputs of the map
            graph.
        """
        motion_head = self.motion_head.prepare_for_export()
        det_fusion = self.det_fusion
        finest_stage = context.stage_count - det_fusion.num_stages
        stage_args = det3d_head_export_args(context, finest_stage)
        device = stage_args[0].device
        bank = bbox_head.instance_bank
        num_queries, embed_dims = bank.num_anchor, bbox_head.embed_dims
        args = (
            *stage_args,
            torch.zeros(1, num_queries, embed_dims, device=device),
            bank.anchor.detach()[None].to(device),
            torch.zeros(1, num_queries, bbox_head.num_classes, device=device),
            torch.zeros(1, num_queries, 2, device=device),
            *motion_head.export_state_inputs(num_queries, device),
            torch.zeros(1, bank.num_temp_instances, dtype=torch.long, device=device),
            torch.zeros(1, device=device),
            torch.eye(4, device=device)[None],
            torch.zeros(1, dtype=torch.long, device=device),
        )
        map_args, map_names = map_export_inputs(online_map_head, device)
        return ExportSpec(
            module=_MotionPlannerExportModule(
                deepcopy(det_fusion).eval(), motion_head, bbox_head
            ).eval(),
            args=(*args, *map_args),
            input_names=[
                *det3d_head_export_input_names(context.stage_count, finest_stage),
                *MOTION_PLANNING_EXPORT_INPUT_NAMES,
                *map_names,
            ],
            output_names=list(MOTION_PLANNING_EXPORT_OUTPUT_NAMES),
            dynamic_axes=det3d_head_export_dynamic_axes(context.stage_count, finest_stage),
            supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
        )


class _MotionPlannerExportModule(nn.Module):
    """Export-only motion planner, one frame with its history at the graph boundary.

    The graph fuses the encoder stages as the detection graph does, the planner samples the
    coarsest. The point cloud is the ego frame, no augmentation moves it.
    """

    def __init__(
        self,
        det_fusion: PTv3Det3DFeaturePyramid,
        motion_head: MotionPlanningHead,
        bbox_head: Sparse4DHead,
    ) -> None:
        """Initialize the export module.

        Args:
            det_fusion: Export-ready detection feature pyramid.
            motion_head: Export-prepared motion planner.
            bbox_head: Export-prepared detection head lending its anchor encoder.
        """
        super().__init__()
        self.det_fusion = det_fusion
        self.motion_head = motion_head
        self.bbox_head = bbox_head

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Predict the trajectories and select the plan of the command, see the input names.

        The map context outputs of the map graph follow the command exactly for a planner with a
        map.
        """
        stages, rest = fuse_export_stages(self.det_fusion, inputs)
        (
            instance_feature,
            prediction,
            classification,
            quality,
            queue_feature,
            queue_anchor,
            queue_map,
            queue_count,
            queue_age,
            ego_velocity,
            carried_index,
            temp_valid,
            prev2current,
            command,
        ) = rest[: len(MOTION_PLANNING_EXPORT_INPUT_NAMES)]
        outputs, state, ego_velocity = self.motion_head.step(
            instance_feature,
            prediction,
            classification,
            self.bbox_head.anchor_encoder,
            # One frame per graph, so the mean over the deepest voxels pools the frame
            stages[-1].feat.mean(dim=0, keepdim=True),
            ego_velocity,
            QueueState(queue_feature, queue_anchor, queue_map, queue_count, queue_age),
            temp_valid > 0.5,
            carried_index,
            prev2current,
            stages[len(stages) - len(self.motion_head.planner.level_grids) :],
            command,
            torch.eye(4, dtype=prediction.dtype, device=prediction.device)[None],
            export_map_context(rest[len(MOTION_PLANNING_EXPORT_INPUT_NAMES) :]),
        )
        scores = Sparse4DHead._query_scores(classification, quality).max(dim=-1).values
        plan = self.motion_head.select_plan(outputs, prediction, scores)
        return (
            outputs.agent_mode_logits,
            outputs.agent_trajectories,
            outputs.plan_logits,
            outputs.plan_trajectories,
            outputs.plan_collision_logits,
            outputs.ego_status,
            plan,
            *state,
            ego_velocity,
        )


def pooled_stage(stage: SparseStage) -> torch.Tensor:
    """Mean feature of the voxels of every sample of a sparse stage.

    Args:
        stage: Features, voxel coordinates and batch offsets of the stage.

    Returns:
        Mean feature of every sample ``(B, C)``.
    """
    feature, coords, offset = stage
    # offset_to_batch runs in inference mode, the backward needs a normal copy of the index
    batch = offset_to_batch(offset, coords).clone()
    total = feature.new_zeros(offset.shape[0], feature.shape[-1]).index_add_(0, batch, feature)
    count = torch.bincount(batch, minlength=offset.shape[0]).clamp(min=1)
    return total / count[:, None].to(feature.dtype)
