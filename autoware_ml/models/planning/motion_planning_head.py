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

"""SparseDrive motion planner on top of the Sparse4D detection queries.

The motion head refines one ego query next to the detection queries. The ego query drives a
vocabulary planner that scores paths and speed profiles along the sampled lidar stages
(Sun et al., SparseDrive).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float
import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.planning import PlanningGTBatch
from autoware_ml.dataclasses.models.motion.head_outputs import MotionPlanningHeadOutputs
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.models.detection3d.heads.sparse4d import ANCHOR_DIM, linear_relu_ln
from autoware_ml.models.detection3d.ptv3 import SparseStage
from autoware_ml.models.planning.geometry import length_to_ego, to_ego, yaw_to_ego
from autoware_ml.models.motion.instance_queue import QueueState
from autoware_ml.models.motion.motion_head import (
    MOTION_STATE_INPUT_NAMES,
    MapContext,
    MotionHead,
    with_total,
)
from autoware_ml.models.planning.vocabulary_planner import VocabularyPlanner
from autoware_ml.utils.distributed import reduce_mean_count

_X, _Y, _Z, _LN_L, _LN_W, _LN_H, _SIN_YAW, _COS_YAW, _VX, _VY = range(ANCHOR_DIM)

# History and ego velocity at the motion planner export graph boundary
MOTION_PLANNING_STATE_INPUT_NAMES = (*MOTION_STATE_INPUT_NAMES, "ego_velocity")
MOTION_PLANNING_STATE_OUTPUT_NAMES = tuple(
    f"{name}_out" for name in MOTION_PLANNING_STATE_INPUT_NAMES
)


class MotionPlanningHead(MotionHead):
    """Predict the trajectories of the detected agents and plan the ego vehicle."""

    # The ego query follows the detection queries and keeps its history slot
    num_extra_queries = 1

    def __init__(
        self,
        *,
        planner: VocabularyPlanner,
        ego_height: float,
        command_lateral_threshold: float = 2.0,
        loss_plan_status_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the head.

        Args:
            planner: Ego planner driven by the refined ego query, its vehicle sets the ego box.
            ego_height: Height of the ego box in meters, the frame origin on the ground.
            command_lateral_threshold: Lateral offset of the last future ego position from which
                the frame is a turn.
            loss_plan_status_weight: Weight of the ego status loss.
            **kwargs: Arguments of :class:`MotionHead`, its map keys exactly for a planner with a
                map. With ``detach_detection`` the planner reads the lidar stages detached too.

        Raises:
            ValueError: If the ego height or the map use of the planner is invalid.
        """
        super().__init__(**kwargs)
        if ego_height <= 0:
            raise ValueError(f"ego_height must be positive, got {ego_height}.")
        if planner.with_map != (self.num_map_keys > 0):
            raise ValueError("The planner reads the map exactly when the head has map keys.")
        self.ego_steps = planner.ego_steps
        self.command_lateral_threshold = command_lateral_threshold
        self.loss_plan_status_weight = loss_plan_status_weight
        vehicle = planner.vehicle
        ego_anchor = torch.zeros(ANCHOR_DIM)
        ego_anchor[_X] = vehicle.center_x
        ego_anchor[_Z] = ego_height / 2.0
        ego_size = torch.tensor([vehicle.length, vehicle.width, ego_height], dtype=torch.float32)
        ego_anchor[_LN_L : _LN_H + 1] = ego_size.log()
        ego_anchor[_COS_YAW] = 1.0
        self.register_buffer("ego_anchor", ego_anchor, persistent=False)
        self._ego_velocity: torch.Tensor | None = None
        embed_dims = self.embed_dims
        self.ego_feature_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, embed_dims))
        self.ego_embedding = nn.Parameter(torch.zeros(embed_dims))
        self.plan_status = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, embed_dims), nn.Linear(embed_dims, 5)
        )
        self.planner = planner

    def reset(self) -> None:
        """Forget the history and the ego velocity of every lane."""
        super().reset()
        self._ego_velocity = None

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        anchor_encoder: nn.Module,
        ego_context: torch.Tensor,
        lane_continues: torch.Tensor,
        carried_index: torch.Tensor | None,
        lidar2global: torch.Tensor,
        stages: Sequence[SparseStage],
        commands: torch.Tensor,
        frame_from_ego: torch.Tensor,
        map_context: MapContext | None = None,
    ) -> MotionPlanningHeadOutputs:
        """Predict the agent trajectories and the plan of one frame.

        Args:
            instance_feature: Last layer features of the detection queries ``(B, N, C)``.
            anchor: Last layer anchors of the detection queries ``(B, N, 10)``.
            cls_logits: Last layer class logits of the detection queries ``(B, N, num_classes)``.
            anchor_encoder: Anchor encoder of the detection head, shared with the ego anchor.
            ego_context: Scene feature of every frame the ego query starts from ``(B, C)``.
            lane_continues: Whether every lane continues its stream ``(B,)``.
            carried_index: Previous frame query every leading temporal slot comes from
                ``(B, K)``, None at the first frame.
            lidar2global: Poses of the frames with the augmentations folded in ``(B, 4, 4)``.
            stages: Fused lidar stages at the feature strides of the planner.
            commands: Driving command of every frame ``(B,)``.
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame
                ``(B, 4, 4)``.
            map_context: Map queries of every frame, exactly for a planner with a map.

        Returns:
            The agent trajectories and the plan of every frame.
        """
        batch_size = anchor.shape[0]
        if self._queue_state is None or self._queue_state.count.shape[0] != batch_size:
            self._ego_velocity = anchor.new_zeros(batch_size, 2)
        queue_state, carried_index, prev2current = self.begin_frame(
            anchor, carried_index, lidar2global
        )
        outputs, self._queue_state, self._ego_velocity = self.step(
            instance_feature,
            anchor,
            cls_logits,
            anchor_encoder,
            ego_context,
            self._ego_velocity,
            queue_state,
            lane_continues,
            carried_index,
            prev2current,
            stages,
            commands,
            frame_from_ego,
            map_context,
        )
        return outputs

    def ego_pose(
        self, frame_from_ego: Float[torch.Tensor, "batch_size 4 4"]
    ) -> Float[torch.Tensor, "batch_size pose_dims"]:
        """The ego box of every frame as the anchor fields before the velocity.

        The ego box sits at the origin of the ego frame, the anchor moves it into the point
        cloud frame of the sample like the detection anchors, rotated, scaled and mirrored.

        Args:
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame.

        Returns:
            Center, log size and heading of the ego box in the point cloud frame.
        """
        transform = frame_from_ego.to(self.ego_anchor.dtype)
        center = torch.cat(
            [
                self.ego_anchor[_X : _X + 1],
                self.ego_anchor.new_zeros(1),
                self.ego_anchor[_Z : _Z + 1],
                self.ego_anchor.new_ones(1),
            ]
        )
        moved = (transform @ center)[:, :3]
        # The forward axis of the ego in the frame, its length is the isotropic scale
        heading = transform[:, :2, 0]
        scale = heading.norm(dim=-1, keepdim=True)
        log_size = self.ego_anchor[_LN_L : _LN_H + 1] + scale.log()
        return torch.cat(
            [moved, log_size, heading[:, 1:2] / scale, heading[:, 0:1] / scale], dim=-1
        )

    def step(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        anchor_encoder: nn.Module,
        ego_context: torch.Tensor,
        ego_velocity: torch.Tensor,
        queue_state: QueueState,
        lane_continues: torch.Tensor,
        carried_index: torch.Tensor,
        prev2current: torch.Tensor,
        stages: Sequence[SparseStage],
        commands: torch.Tensor,
        frame_from_ego: torch.Tensor,
        map_context: MapContext | None = None,
    ) -> tuple[MotionPlanningHeadOutputs, QueueState, torch.Tensor]:
        """Predict one frame from the state of the last one, the training and export path.

        Args:
            instance_feature: Last layer features of the detection queries ``(B, N, C)``.
            anchor: Last layer anchors of the detection queries ``(B, N, 10)``.
            cls_logits: Last layer class logits of the detection queries ``(B, N, num_classes)``.
            anchor_encoder: Anchor encoder of the detection head, shared with the ego anchor.
            ego_context: Scene feature of every frame the ego query starts from ``(B, C)``.
            ego_velocity: Ego velocity the last frame predicted ``(B, 2)``.
            queue_state: History after the last frame.
            lane_continues: Whether every lane continues its stream ``(B,)``.
            carried_index: Previous frame query every leading temporal slot comes from
                ``(B, K)``.
            prev2current: Transform from the previous to the current frame ``(B, 4, 4)``.
            stages: Fused lidar stages at the feature strides of the planner.
            commands: Driving command of every frame ``(B,)``.
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame
                ``(B, 4, 4)``.
            map_context: Map queries of every frame, exactly for a planner with a map.

        Returns:
            The outputs, the history after this frame and the predicted ego velocity.
        """
        if self.detach_detection:
            ego_context = ego_context.detach()
            stages = [(feat.detach(), coord, offset) for feat, coord, offset in stages]
        batch_size, num_queries = anchor.shape[:2]
        ego_velocity = ego_velocity * lane_continues[:, None].to(anchor.dtype)
        ego_anchor = torch.cat([self.ego_pose(frame_from_ego), ego_velocity], dim=-1)
        ego_feature = self.ego_feature_fc(ego_context) + self.ego_embedding
        attended = self.attend(
            instance_feature,
            anchor,
            cls_logits,
            anchor_encoder,
            ego_feature[:, None],
            ego_anchor[:, None],
            queue_state,
            lane_continues,
            carried_index,
            prev2current,
            map_context,
        )
        context = attended.feature + attended.anchor_embed
        agent_mode_logits, agent_trajectories = self._predict_agents(context, anchor, cls_logits)
        top_keys = attended.top[..., None].expand(-1, -1, self.embed_dims)
        plans = self.planner(
            context[:, num_queries],
            commands,
            context[:, :num_queries].gather(1, top_keys),
            attended.map_keys,
            stages,
            frame_from_ego,
        )
        outputs = MotionPlanningHeadOutputs(
            agent_mode_logits=agent_mode_logits,
            agent_trajectories=agent_trajectories,
            ego_status=self.plan_status(context[:, num_queries]),
            plan_logits=plans.logits,
            plan_trajectories=plans.trajectories,
            plan_collision_logits=plans.collision_logits,
            plan_layers=plans.layers,
        )
        # The history keeps the detection features, the ego query its refined one
        return outputs, self.push(attended), outputs.ego_status[:, 0:2].detach()

    def prepare_for_export(self) -> MotionPlanningHead:
        """Return an export ready copy with explicit attention graphs."""
        head = super().prepare_for_export()
        head.planner = head.planner.prepare_for_export()
        return head

    def export_state_inputs(
        self, num_queries: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        """Reset state sample inputs, in MOTION_PLANNING_STATE_INPUT_NAMES order.

        Args:
            num_queries: Detection queries per frame.
            device: Device of the inputs.

        Returns:
            The empty history and a zero ego velocity of one lane.
        """
        history = super().export_state_inputs(num_queries, device)
        return (*history, torch.zeros(1, 2, device=device))

    def loss(
        self,
        outputs: MotionPlanningHeadOutputs,
        assigned_instance: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_instance_ids: torch.Tensor,
        targets: AgentFutureTargets,
        planning_gt: PlanningGTBatch,
        frame_from_ego: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Supervise the agent trajectories of the matched queries and the planner.

        The planner learns the ego path and future in the ego frame, its collision head the
        trajectories that meet a ground truth agent.

        Args:
            outputs: Raw outputs from :meth:`forward`.
            assigned_instance: Instance the last detection layer matched every query to
                ``(B, N)``, -1 for background.
            gt_boxes: Padded ground truth boxes of the frames ``(B, M, >=7)``.
            gt_instance_ids: Instance id of every padded box slot ``(B, M)``, -1 for padding.
            targets: Futures of the agents of the box slots of the frames.
            planning_gt: Ego futures, statuses and paths of the frames.
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame
                ``(B, 4, 4)``.

        Returns:
            Loss dictionary with the terms and their sum under ``loss``.
        """
        losses = self.agent_losses(outputs, assigned_instance, gt_boxes, gt_instance_ids, targets)
        status_valid = planning_gt.ego_status_valid.float()
        loss_plan_status = (
            (outputs.ego_status.float() - planning_gt.ego_status).abs().sum(-1) * status_valid
        ).sum() / reduce_mean_count(status_valid.sum())

        steps = self.ego_steps
        # The ego box starts at the origin heading along x only in the ego frame
        collision_labels = self.planner.collisions(
            to_ego(outputs.plan_trajectories.detach(), frame_from_ego),
            to_ego(targets.agent_future_positions[:, :, :steps], frame_from_ego),
            length_to_ego(gt_boxes[..., 3:5], frame_from_ego),
            yaw_to_ego(gt_boxes[..., 6], frame_from_ego),
            targets.agent_future_valid[:, :, :steps] & (gt_instance_ids >= 0)[..., None],
        )
        plan_losses = self.planner.loss(
            outputs.plan_layers,
            outputs.plan_collision_logits,
            collision_labels,
            to_ego(planning_gt.ego_path, frame_from_ego),
            planning_gt.ego_path_valid,
            to_ego(planning_gt.ego_future_positions, frame_from_ego),
            planning_gt.ego_future_valid,
        )
        losses["loss_plan_status"] = self.loss_plan_status_weight * loss_plan_status
        losses.update(plan_losses)
        losses["num_plans"] = outputs.agent_mode_logits.new_tensor(
            float(planning_gt.ego_future_valid.any(dim=1).sum())
        )
        losses["plan_collision_rate"] = collision_labels.float().sum() / max(
            collision_labels.numel(), 1
        )
        return with_total(losses)

    def select_plan(
        self,
        outputs: MotionPlanningHeadOutputs,
        agent_anchor: torch.Tensor,
        agent_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Pick the plan of every frame, trajectories that meet a confident agent ranked last.

        Agents follow their most likely trajectory and keep their detected heading.

        Args:
            outputs: Raw outputs from :meth:`forward`.
            agent_anchor: Anchors of the detection queries ``(B, N, 10)``.
            agent_scores: Detection score of every query ``(B, N)``.

        Returns:
            The selected ego trajectory of every frame ``(B, T, 2)``.
        """
        best_mode = outputs.agent_mode_logits.argmax(dim=-1)
        agent_paths = (
            outputs.agent_trajectories.gather(
                2, best_mode[..., None, None, None].expand(-1, -1, 1, self.agent_steps, 2)
            ).squeeze(2)[:, :, : self.ego_steps]
            + agent_anchor[:, :, None, :2]
        )
        return self.planner.select(
            outputs.plan_logits,
            outputs.plan_trajectories,
            agent_paths,
            agent_anchor[..., _LN_L : _LN_W + 1].exp(),
            torch.atan2(agent_anchor[..., _SIN_YAW], agent_anchor[..., _COS_YAW]),
            agent_scores,
        )
