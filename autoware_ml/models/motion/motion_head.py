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

"""SparseDrive agent motion head on top of the Sparse4D detection queries.

The detection queries attend to their own history and to the most confident detections, then
predict the multi-modal trajectory of every agent (Sun et al., SparseDrive, 2024).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.dataclasses.batch.motion import MotionGTBatch
from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.anchored import DataAnchored, anchor_tensor
from autoware_ml.models.detection3d.heads.sparse4d import (
    ANCHOR_DIM,
    DecoupledInstanceAttention,
    linear_relu_ln,
)
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.task_modules.streaming import pos2posemb2d
from autoware_ml.models.motion.instance_queue import (
    QueueState,
    advance_queue,
    push_queue,
    queue_history,
)

_X, _Y, _Z, _LN_L, _LN_W, _LN_H, _SIN_YAW, _COS_YAW, _VX, _VY = range(ANCHOR_DIM)

# History at the motion export graph boundary, in QueueState order
MOTION_STATE_INPUT_NAMES = (
    "queue_feature",
    "queue_anchor",
    "queue_map",
    "queue_count",
    "queue_age",
)
MOTION_STATE_OUTPUT_NAMES = tuple(f"{name}_out" for name in MOTION_STATE_INPUT_NAMES)


class AttendedQueries(NamedTuple):
    """Queries of one frame after the attention layers.

    Attributes:
        feature: Refined features of the detection and the extra queries ``(B, N + E, C)``.
        anchor_embed: Anchor embeddings of the same queries ``(B, N + E, C)``.
        anchors: Anchors of the same queries ``(B, N + E, 10)``.
        instance_feature: Detection features the queries started from ``(B, N, C)``.
        top: Most confident detection queries every query attended to ``(B, K)``.
        queue_state: History of the frame before this frame is pushed.
    """

    feature: torch.Tensor
    anchor_embed: torch.Tensor
    anchors: torch.Tensor
    instance_feature: torch.Tensor
    top: torch.Tensor
    queue_state: QueueState


def _without_grad(module: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Call a module without recording gradients for its parameters."""

    def call(inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return module(inputs)

    return call


def with_total(losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the sum of the ``loss_`` terms under ``loss``."""
    losses["loss"] = sum(value for key, value in losses.items() if key.startswith("loss_"))
    return losses


class MotionHead(DataAnchored):
    """Predict the trajectories of the tracked agents."""

    anchor_names = ("motion_anchors",)
    # Queries after the detection queries that keep their history slot, a subclass adds them
    num_extra_queries = 0

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dims: int,
        anchor_recipe: AnchorRecipe,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        dropout: float = 0.1,
        agent_modes: int = 6,
        agent_steps: int = 12,
        num_agent_keys: int = 50,
        queue_length: int = 4,
        queue_stride: int = 5,
        loss_motion_cls_weight: float = 0.2,
        loss_motion_reg_weight: float = 0.2,
        detach_detection: bool = False,
    ) -> None:
        """Initialize the head.

        Args:
            num_classes: Detection classes, every class has its own motion anchors.
            embed_dims: Feature width of the detection queries.
            anchor_recipe: Recipe of the motion anchors ``(num_classes, agent_modes, agent_steps,
                2)``, positions in the frame of the agent (x along its heading).
            num_layers: Attention layers before the prediction.
            num_heads: Attention heads.
            feedforward_channels: Hidden width of the FFN blocks.
            dropout: Dropout of the attention and FFN blocks.
            agent_modes: Trajectory modes per agent.
            agent_steps: Future steps of the agent trajectories.
            num_agent_keys: Most confident detections every query attends to.
            queue_length: History entries per query, the current frame included.
            queue_stride: Frames between two history entries.
            loss_motion_cls_weight: Weight of the agent mode classification loss.
            loss_motion_reg_weight: Weight of the agent trajectory loss.
            detach_detection: Train the head on the detection outputs without passing its
                gradients back into them and the anchor encoder.

        Raises:
            ValueError: If the history is invalid.
        """
        super().__init__(anchor_recipe)
        if queue_length < 1 or queue_stride < 1:
            raise ValueError(
                f"The history needs queue_length >= 1 and queue_stride >= 1, got {queue_length} "
                f"and {queue_stride}."
            )
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.agent_modes = agent_modes
        self.agent_steps = agent_steps
        self.num_agent_keys = num_agent_keys
        self.loss_motion_cls_weight = loss_motion_cls_weight
        self.loss_motion_reg_weight = loss_motion_reg_weight
        # Filled from the anchor recipe before training, see DataAnchored
        self.register_buffer(
            "motion_anchors", torch.zeros(num_classes, agent_modes, agent_steps, 2)
        )
        self.queue_length = queue_length
        self.queue_stride = queue_stride
        self._queue_state: QueueState | None = None
        self._previous_pose: torch.Tensor | None = None
        self.temporal_attention = nn.ModuleList(
            [DecoupledInstanceAttention(embed_dims, num_heads, dropout) for _ in range(num_layers)]
        )
        # Normalized queries and keys keep the agent attention logits within fp16
        self.temporal_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        self.agent_attention = nn.ModuleList(
            [DecoupledInstanceAttention(embed_dims, num_heads, dropout) for _ in range(num_layers)]
        )
        self.attention_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, feedforward_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feedforward_channels, embed_dims),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_layers)])
        self.motion_mode_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, embed_dims))
        self.motion_cls = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, embed_dims), nn.Linear(embed_dims, 1)
        )
        self.motion_reg = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2, embed_dims), nn.Linear(embed_dims, agent_steps * 2)
        )
        nn.init.constant_(self.motion_cls[-1].bias, -math.log((1.0 - 0.01) / 0.01))
        self.loss_cls = SigmoidFocalLoss()
        self.detach_detection = detach_detection

    def load_anchors(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Copy the motion anchors of the recipe into the head.

        Args:
            arrays: Anchor arrays of the recipe, ``anchors`` ``(num_classes, agent_modes,
                agent_steps, 2)``.
        """
        self.motion_anchors.copy_(
            anchor_tensor(arrays, "anchors", tuple(self.motion_anchors.shape))
        )

    def reset(self) -> None:
        """Forget the history of every lane."""
        self._queue_state = None
        self._previous_pose = None

    def begin_frame(
        self, anchor: torch.Tensor, carried_index: torch.Tensor | None, lidar2global: torch.Tensor
    ) -> tuple[QueueState, torch.Tensor, torch.Tensor]:
        """Read the history of the last frame and the transform from it.

        Args:
            anchor: Last layer anchors of the detection queries ``(B, N, 10)``.
            carried_index: Previous frame query every leading temporal slot comes from
                ``(B, K)``, None at the first frame.
            lidar2global: Poses of the frames with the augmentations folded in ``(B, 4, 4)``.

        Returns:
            The history after the last frame, the slot index and the transform from the last
            frame into this one.
        """
        batch_size, num_queries = anchor.shape[:2]
        pose = lidar2global.to(torch.float64)
        if self._queue_state is None or self._queue_state.count.shape[0] != batch_size:
            self._queue_state = QueueState.empty(
                batch_size,
                self.queue_length,
                num_queries + self.num_extra_queries,
                self.embed_dims,
                self.queue_stride,
                anchor.device,
            )
            self._previous_pose = pose
            # The slots of another batch size carry no query into the new queue
            carried_index = None
        if carried_index is None:
            carried_index = torch.zeros(batch_size, 0, dtype=torch.long, device=anchor.device)
        prev2current = torch.linalg.solve(pose, self._previous_pose).to(anchor.dtype)
        self._previous_pose = pose.clone()
        return self._queue_state, carried_index, prev2current

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        anchor_encoder: nn.Module,
        lane_continues: torch.Tensor,
        carried_index: torch.Tensor | None,
        lidar2global: torch.Tensor,
    ) -> MotionHeadOutputs:
        """Predict the agent trajectories of one frame.

        Args:
            instance_feature: Last layer features of the detection queries ``(B, N, C)``.
            anchor: Last layer anchors of the detection queries ``(B, N, 10)``.
            cls_logits: Last layer class logits of the detection queries ``(B, N, num_classes)``.
            anchor_encoder: Anchor encoder of the detection head.
            lane_continues: Whether every lane continues its stream ``(B,)``.
            carried_index: Previous frame query every leading temporal slot comes from
                ``(B, K)``, None at the first frame.
            lidar2global: Poses of the frames with the augmentations folded in ``(B, 4, 4)``.

        Returns:
            The agent trajectories of every frame.
        """
        queue_state, carried_index, prev2current = self.begin_frame(
            anchor, carried_index, lidar2global
        )
        outputs, self._queue_state = self.step(
            instance_feature,
            anchor,
            cls_logits,
            anchor_encoder,
            queue_state,
            lane_continues,
            carried_index,
            prev2current,
        )
        return outputs

    def step(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        anchor_encoder: nn.Module,
        queue_state: QueueState,
        lane_continues: torch.Tensor,
        carried_index: torch.Tensor,
        prev2current: torch.Tensor,
    ) -> tuple[MotionHeadOutputs, QueueState]:
        """Predict one frame from the history of the last one, the training and export path.

        Args:
            instance_feature: Last layer features of the detection queries ``(B, N, C)``.
            anchor: Last layer anchors of the detection queries ``(B, N, 10)``.
            cls_logits: Last layer class logits of the detection queries ``(B, N, num_classes)``.
            anchor_encoder: Anchor encoder of the detection head.
            queue_state: History after the last frame.
            lane_continues: Whether every lane continues its stream ``(B,)``.
            carried_index: Previous frame query every leading temporal slot comes from
                ``(B, K)``.
            prev2current: Transform from the previous to the current frame ``(B, 4, 4)``.

        Returns:
            The outputs and the history after this frame.
        """
        attended = self.attend(
            instance_feature,
            anchor,
            cls_logits,
            anchor_encoder,
            instance_feature.new_zeros(anchor.shape[0], 0, self.embed_dims),
            anchor.new_zeros(anchor.shape[0], 0, ANCHOR_DIM),
            queue_state,
            lane_continues,
            carried_index,
            prev2current,
        )
        agent_mode_logits, agent_trajectories = self._predict_agents(
            attended.feature + attended.anchor_embed, anchor, cls_logits
        )
        outputs = MotionHeadOutputs(
            agent_mode_logits=agent_mode_logits, agent_trajectories=agent_trajectories
        )
        return outputs, self.push(attended)

    def attend(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        anchor_encoder: nn.Module,
        extra_feature: torch.Tensor,
        extra_anchor: torch.Tensor,
        queue_state: QueueState,
        lane_continues: torch.Tensor,
        carried_index: torch.Tensor,
        prev2current: torch.Tensor,
    ) -> AttendedQueries:
        """Refine the detection and the extra queries over their history and the agents.

        Args:
            instance_feature: Last layer features of the detection queries ``(B, N, C)``.
            anchor: Last layer anchors of the detection queries ``(B, N, 10)``.
            cls_logits: Last layer class logits of the detection queries ``(B, N, num_classes)``.
            anchor_encoder: Anchor encoder of the detection head, shared with the extra queries.
            extra_feature: Features of the extra queries after the detections ``(B, E, C)``.
            extra_anchor: Anchors of the extra queries ``(B, E, 10)``.
            queue_state: History after the last frame.
            lane_continues: Whether every lane continues its stream ``(B,)``.
            carried_index: Previous frame query every leading temporal slot comes from
                ``(B, K)``.
            prev2current: Transform from the previous to the current frame ``(B, 4, 4)``.

        Returns:
            The refined queries and the history of the frame.
        """
        if self.detach_detection:
            instance_feature, anchor = instance_feature.detach(), anchor.detach()
            cls_logits, anchor_encoder = cls_logits.detach(), _without_grad(anchor_encoder)
        batch_size, num_queries = anchor.shape[:2]
        num_extra = extra_feature.shape[1]
        feature = torch.cat([instance_feature, extra_feature], dim=1)
        anchors = torch.cat([anchor, extra_anchor], dim=1)
        anchor_embed = anchor_encoder(anchors)

        queue_state = advance_queue(
            queue_state, lane_continues, carried_index, prev2current, self.queue_stride, num_extra
        )
        history_feature, history_anchor, history_valid = queue_history(queue_state)
        # The current frame leads the keys of the temporal attention
        length = history_feature.shape[2]
        key_feature = torch.cat(
            [feature.detach()[:, :, None], history_feature[:, :, : length - 1]], 2
        )
        key_embed = torch.cat(
            [anchor_embed[:, :, None], anchor_encoder(history_anchor[:, :, : length - 1])], dim=2
        )
        key_valid = torch.cat(
            [torch.ones_like(history_valid[:, :, :1]), history_valid[:, :, : length - 1]], 2
        )

        confidence = cls_logits.detach().max(dim=-1).values
        top = confidence.topk(min(self.num_agent_keys, num_queries), dim=1).indices
        # ONNX sizes an expanded empty tensor as one key, so no extra queries means detections only
        agent_keys = top
        if num_extra > 0:
            extra_keys = torch.arange(num_queries, num_queries + num_extra, device=top.device)
            agent_keys = torch.cat([top, extra_keys.expand(batch_size, -1)], dim=1)

        total = num_queries + num_extra
        for layer in range(self.num_layers):
            flat = self.temporal_attention[layer](
                feature.reshape(batch_size * total, 1, -1),
                anchor_embed.reshape(batch_size * total, 1, -1),
                key_feature.reshape(batch_size * total, length, -1),
                key_embed.reshape(batch_size * total, length, -1),
                key_padding_mask=~key_valid.reshape(batch_size * total, length),
            )
            feature = self.temporal_norm[layer](flat.reshape(batch_size, total, -1))
            keys = agent_keys[..., None].expand(-1, -1, self.embed_dims)
            feature = self.agent_attention[layer](
                feature, anchor_embed, feature.gather(1, keys), anchor_embed.gather(1, keys)
            )
            feature = self.attention_norm[layer](feature)
            feature = self.ffn_norm[layer](feature + self.ffn[layer](feature))
        return AttendedQueries(
            feature=feature,
            anchor_embed=anchor_embed,
            anchors=anchors,
            instance_feature=instance_feature,
            top=top,
            queue_state=queue_state,
        )

    def push(self, attended: AttendedQueries) -> QueueState:
        """Store the frame in the history, the detection features and the refined extra queries."""
        num_queries = attended.instance_feature.shape[1]
        return push_queue(
            attended.queue_state,
            torch.cat([attended.instance_feature, attended.feature[:, num_queries:]], dim=1),
            attended.anchors,
            self.queue_stride,
        )

    def prepare_for_export(self) -> MotionHead:
        """Return an export ready copy with explicit attention graphs in the fp16 range."""
        head = deepcopy(self)
        blocks = [
            *zip(head.temporal_attention, head.temporal_norm, strict=True),
            *zip(head.agent_attention, head.attention_norm, strict=True),
        ]
        for attention_block, norm in blocks:
            attention_block.fold_output_scale(norm)
            attention_block.attn = ExportableMultiheadAttention(attention_block.attn)
        return head.eval()

    def export_state_inputs(
        self, num_queries: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        """Reset state sample inputs for tracing the export, in MOTION_STATE_INPUT_NAMES order.

        Args:
            num_queries: Detection queries per frame.
            device: Device of the inputs.

        Returns:
            The empty history of one lane.
        """
        return tuple(
            QueueState.empty(
                1,
                self.queue_length,
                num_queries + self.num_extra_queries,
                self.embed_dims,
                self.queue_stride,
                device,
            )
        )

    def _predict_agents(
        self, context: torch.Tensor, anchor: torch.Tensor, cls_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn the refined detection queries into trajectories.

        Args:
            context: Refined features with their anchor embeddings ``(B, N, C)``, extra queries
                after them are ignored.
            anchor: Anchors of the detection queries ``(B, N, 10)``.
            cls_logits: Class logits of the detection queries ``(B, N, num_classes)``.

        Returns:
            Mode logits ``(B, N, M)`` and trajectories ``(B, N, M, T, 2)``.
        """
        batch_size, num_queries = anchor.shape[:2]
        context = context[:, :num_queries]
        # Mode queries of an agent: the anchor ends of its class, turned to its heading
        classes = cls_logits.detach().argmax(dim=-1)
        endpoints = self.motion_anchors[:, :, -1][classes]
        # Only the angle of the regressed heading turns the anchors, a zero heading turns none
        heading = anchor[..., _SIN_YAW : _COS_YAW + 1].detach()
        norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        heading = torch.where(
            norm > 1e-6, heading / norm.clamp_min(1e-6), heading.new_tensor([0.0, 1.0])
        )
        sin, cos = heading[..., :1], heading[..., 1:]
        endpoints = torch.stack(
            [
                endpoints[..., 0] * cos - endpoints[..., 1] * sin,
                endpoints[..., 0] * sin + endpoints[..., 1] * cos,
            ],
            dim=-1,
        )
        motion_query = self.motion_mode_fc(pos2posemb2d(endpoints.float(), self.embed_dims // 2))
        motion_query = motion_query + context[:, :, None]
        agent_mode_logits = self.motion_cls(motion_query).squeeze(-1)
        agent_steps = self.motion_reg(motion_query).reshape(
            batch_size, num_queries, self.agent_modes, self.agent_steps, 2
        )
        return agent_mode_logits, agent_steps.cumsum(dim=-2)

    def loss(
        self,
        outputs: MotionHeadOutputs,
        assigned_instance: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_instance_ids: torch.Tensor,
        motion_gt: MotionGTBatch,
    ) -> dict[str, torch.Tensor]:
        """Supervise the agent trajectories of the matched queries.

        Args:
            outputs: Raw outputs from :meth:`forward`.
            assigned_instance: Instance the last detection layer matched every query to
                ``(B, N)``, -1 for background.
            gt_boxes: Padded ground truth boxes of the frames ``(B, M, >=7)``.
            gt_instance_ids: Instance id of every padded box slot ``(B, M)``, -1 for padding.
            motion_gt: Futures of the frames, the agents aligned with the box slots.

        Returns:
            Loss dictionary with the terms and their sum under ``loss``.
        """
        return with_total(
            self.agent_losses(outputs, assigned_instance, gt_boxes, gt_instance_ids, motion_gt)
        )

    def agent_losses(
        self,
        outputs: MotionHeadOutputs,
        assigned_instance: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_instance_ids: torch.Tensor,
        motion_gt: MotionGTBatch,
    ) -> dict[str, torch.Tensor]:
        """Supervise the agent trajectories of the matched queries, the terms without their sum.

        An agent query is a target when the detection head matched it to a box with a known
        future. The best mode of a target is the one closest to its future (winner takes all).

        Args:
            outputs: Raw outputs from :meth:`forward`.
            assigned_instance: Instance the last detection layer matched every query to
                ``(B, N)``, -1 for background.
            gt_boxes: Padded ground truth boxes of the frames ``(B, M, >=7)``.
            gt_instance_ids: Instance id of every padded box slot ``(B, M)``, -1 for padding.
            motion_gt: Futures of the frames, the agents aligned with the box slots.

        Returns:
            The mode and the trajectory loss and the number of targets.
        """
        # Box slot of the instance every query was matched to
        matches = (assigned_instance[..., None] >= 0) & (
            assigned_instance[..., None] == gt_instance_ids[:, None, :]
        )
        has_slot = matches.any(dim=-1)
        slot = matches.float().argmax(dim=-1)
        future = motion_gt.agent_future_positions.gather(
            1, slot[..., None, None].expand(-1, -1, self.agent_steps, 2)
        )
        future_valid = (
            motion_gt.agent_future_valid.gather(1, slot[..., None].expand(-1, -1, self.agent_steps))
            & has_slot[..., None]
        )
        center = gt_boxes[..., :2].gather(1, slot[..., None].expand(-1, -1, 2))
        displacement = future - center[:, :, None]
        target = future_valid.any(dim=-1)
        num_targets = max(int(target.sum()), 1)

        error = (outputs.agent_trajectories.float() - displacement[:, :, None]).norm(dim=-1)
        step_weight = future_valid[:, :, None].float()
        mean_error = (error * step_weight).sum(-1) / step_weight.sum(-1).clamp(min=1.0)
        best = mean_error.argmin(dim=-1)
        mode_targets = nn.functional.one_hot(best, self.agent_modes).float()
        loss_motion_cls = self.loss_cls(
            outputs.agent_mode_logits.flatten(end_dim=1).float(),
            mode_targets.flatten(end_dim=1),
            weights=target.flatten()[:, None].float().expand(-1, self.agent_modes),
            avg_factor=float(num_targets),
        )
        best_error = (
            (
                outputs.agent_trajectories.float()
                .gather(2, best[..., None, None, None].expand(-1, -1, 1, self.agent_steps, 2))
                .squeeze(2)
                - displacement
            )
            .abs()
            .sum(-1)
        )
        loss_motion_reg = (best_error * future_valid).sum() / max(int(future_valid.sum()), 1)
        return {
            "loss_motion_cls": self.loss_motion_cls_weight * loss_motion_cls,
            "loss_motion_reg": self.loss_motion_reg_weight * loss_motion_reg,
            "num_motion_targets": outputs.agent_mode_logits.new_tensor(float(target.sum())),
        }
