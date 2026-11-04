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

"""PTv3 segmentation and Sparse4D detection with the SparseDrive agent motion head.

The motion head predicts the trajectory of every detection query from the queries of the temporal
detection head and their history, so the model trains on scenario streams. An online map head
over the fused stages adds the map queries the motion queries attend to.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int64, Shaped

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.motion.online_map_outputs import OnlineMapHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs
from autoware_ml.dataclasses.models.motion.targets import AgentFutureTargets
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.metrics.eval_output import Detection3DEvalOutput, EvalOutput
from autoware_ml.metrics.motion.eval_output import (
    MotionEvalFrame,
    MotionEvalOutput,
    OnlineMapEvalFrame,
    OnlineMapEvalOutput,
)
from autoware_ml.models.detection3d.heads.sparse4d import (
    Sparse4DFrameOutputs,
    Sparse4DHead,
    Sparse4DTargets,
)
from autoware_ml.models.motion.online_map_head import OnlineMapHead
from autoware_ml.models.motion.instance_queue import QueueState
from autoware_ml.models.motion.motion_head import (
    MOTION_STATE_INPUT_NAMES,
    MOTION_STATE_OUTPUT_NAMES,
    MapContext,
    MotionHead,
)
from autoware_ml.models.multi.detection3d_frames import detection3d_frame_mask, detection3d_gt_batch
from autoware_ml.models.multi.ptv3_det3d_seg3d_sparse4d import (
    PTv3Det3DSeg3DSparse4DModel,
    PTv3Sparse4DHeadExportModule,
    build_sparse4d_head_export_spec,
)
from autoware_ml.models.segmentation3d.ptv3_base import PTv3BaseModel, PTv3ExportContext
from autoware_ml.types.geometry import AnchorFieldIndex
from autoware_ml.utils.deploy import ExportSpec

# Anchor channels of the velocity a detection estimates
_VELOCITY = [AnchorFieldIndex.VELOCITY_X, AnchorFieldIndex.VELOCITY_Y]

# The motion graph takes the detection outputs of the frame and the history of the last one
MOTION_EXPORT_INPUT_NAMES = (
    "instance_feature",
    "prediction",
    "classification",
    *MOTION_STATE_INPUT_NAMES,
    # Slot index the detection graph returned for the last frame
    "carried_index",
    "temp_valid",
    "prev2current",
)
MOTION_EXPORT_OUTPUT_NAMES = (
    "agent_mode_logits",
    "agent_trajectories",
    *MOTION_STATE_OUTPUT_NAMES,
)
# The map graph returns the context the motion graph attends to and the map state of the next
# frame, with export_online_map the polylines of its queries first
ONLINE_MAP_POLYLINE_OUTPUT_NAMES = ("classification", "prediction")
ONLINE_MAP_CONTEXT_OUTPUT_NAMES = ("instance_feature", "anchor_embed", "score")
ONLINE_MAP_STATE_OUTPUT_NAMES = (
    "temp_instance_feature_out",
    "temp_anchor_out",
    "temp_confidence_out",
)
# A motion head with a map also takes the context outputs of the map graph
MOTION_MAP_INPUT_NAMES = ("map_instance_feature", "map_anchor_embed", "map_score")


class PTv3SparseDriveMotionModel(PTv3Det3DSeg3DSparse4DModel):
    """PTv3 joint segmentation and Sparse4D detection with the SparseDrive agent motion head."""

    def __init__(
        self,
        *,
        motion_head: MotionHead,
        motion_loss_weight: float = 1.0,
        motion_eval_queries: int = 300,
        online_map_head: OnlineMapHead | None = None,
        online_map_loss_weight: float = 1.0,
        export_online_map: bool = False,
        online_map_supervised: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            motion_head: Motion head reading the detection queries.
            motion_loss_weight: Weight of the motion loss term.
            motion_eval_queries: Most confident detection queries whose trajectories the motion
                metrics read per frame.
            online_map_head: Online mapping head over the fused stages, exactly for a motion head
                with map keys.
            online_map_loss_weight: Weight of the mapping loss term.
            export_online_map: Whether the map graph also returns the polylines and their class
                scores. Without them it hands the motion graph its context only.
            online_map_supervised: Whether the batches carry the map elements of the map loss.
            **kwargs: Arguments of :class:`PTv3Det3DSeg3DSparse4DModel`.

        Raises:
            ValueError: If the motion head has anchors of other classes, or the motion head and
                the map head disagree on the map.
        """
        super().__init__(**kwargs)
        if motion_head.num_classes != self.bbox_head.num_classes:
            raise ValueError(
                f"The motion head has anchors for {motion_head.num_classes} classes, the detection "
                f"head predicts {self.bbox_head.num_classes}."
            )
        if (online_map_head is None) != (motion_head.num_map_keys == 0):
            raise ValueError(
                "A map head needs a motion head with map keys and a motion head with map keys a "
                "map head."
            )
        if online_map_head is not None and online_map_head.embed_dims != self.bbox_head.embed_dims:
            raise ValueError(
                f"The map head is {online_map_head.embed_dims} wide, the fused stages it samples "
                f"{self.bbox_head.embed_dims}."
            )
        self.motion_head = motion_head
        self.motion_loss_weight = float(motion_loss_weight)
        self.motion_eval_queries = motion_eval_queries
        self.online_map_head = online_map_head
        self.online_map_loss_weight = float(online_map_loss_weight)
        self.export_online_map = export_online_map
        self.online_map_supervised = online_map_supervised

    def reset_streams(self) -> None:
        """Forget the temporal state of every lane, the motion history and the map too."""
        super().reset_streams()
        self.motion_head.reset()
        if self.online_map_head is not None:
            self.online_map_head.reset()

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group the parameters of the base model, the motion head and the map head."""
        groups = dict(super().build_optimizer_groups())
        groups["motion_head"] = [p for p in self.motion_head.parameters() if p.requires_grad]
        if self.online_map_head is not None:
            groups["online_map_head"] = [
                p for p in self.online_map_head.parameters() if p.requires_grad
            ]
        return groups

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the encoder, the detection, segmentation and map heads and the motion head.

        Args:
            multi_task_batch_inputs: Model inputs with the voxels, the frame metadata and the
                detection ground truth.

        Returns:
            The detection, segmentation, map and motion outputs.
        """
        branches = self.forward_branches(multi_task_batch_inputs)
        carried_index = self.bbox_head.instance_bank.carried_query_index
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        online_map_outputs, map_context = None, None
        if self.online_map_head is not None:
            online_map_outputs = self.online_map_head(
                branches.stages,
                timestamps_seconds=frame_meta.timestamps,
                lidar2global=frame_meta.ego2globals,
                scene_tokens=frame_meta.scene_tokens,
            )
            map_context = self.map_context(online_map_outputs)
        motion_outputs = self.forward_motion(
            multi_task_batch_inputs, branches.det3d_outputs, carried_index, map_context
        )
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=None,
                sparse4d_head_outputs=branches.det3d_outputs,
            ),
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=branches.seg3d_logits),
            motion_head_outputs=motion_outputs,
            online_map_head_outputs=online_map_outputs,
        )

    def forward_motion(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        det3d_outputs: Sparse4DHeadOutputs,
        carried_index: Int64[torch.Tensor, "batch_size num_temp_instances"] | None,
        map_context: MapContext | None,
    ) -> MotionHeadOutputs:
        """Run the motion head on the last layer detection queries.

        Args:
            multi_task_batch_inputs: Model inputs with the frame metadata.
            det3d_outputs: Outputs of the detection head.
            carried_index: Slot index of the previous frame, None at the first frame.
            map_context: Map queries of a model with a map head.

        Returns:
            The motion outputs.
        """
        frame_meta = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
        return self.motion_head(
            det3d_outputs.instance_feature,
            det3d_outputs.prediction[:, -1],
            det3d_outputs.classification[:, -1],
            self.bbox_head.anchor_encoder,
            self.bbox_head.instance_bank.lane_continues,
            carried_index,
            frame_meta.ego2globals,
            map_context,
        )

    def map_context(self, online_map_outputs: OnlineMapHeadOutputs) -> MapContext:
        """Map queries of the last layer the motion queries attend to.

        Args:
            online_map_outputs: Raw outputs of the map head.

        Returns:
            Their features, polyline embeddings and best class scores.
        """
        return online_map_context(
            self.online_map_head.anchor_encoder,
            online_map_outputs.instance_feature,
            online_map_outputs.prediction[:, -1],
            online_map_outputs.classification[:, -1],
        )

    def compute_metrics(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        multi_task_outputs: ModelOutputs,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the segmentation, detection, mapping and motion losses.

        The motion targets follow the detection matches, so frames without boxes skip the motion
        loss as they skip the detection loss.

        Raises:
            ValueError: If the batch carries no futures, or no map elements for a map head.
        """
        losses = self.detection_segmentation_metrics(multi_task_batch_inputs, multi_task_outputs)
        metrics = losses.metrics
        motion_metrics = self.motion_loss(
            multi_task_batch_inputs,
            multi_task_outputs.motion(),
            losses.det3d_targets,
            losses.det3d_mask,
        )
        weighted_motion_loss = self.motion_loss_weight * motion_metrics["loss"]
        metrics["weighted_motion_loss"] = weighted_motion_loss
        metrics["loss"] = metrics["loss"] + weighted_motion_loss
        metrics.update({f"motion_{name}": value for name, value in motion_metrics.items()})
        if self.online_map_head is not None and self.online_map_supervised:
            map_gt = multi_task_batch_inputs.multi_task_gt_batch.online_map_gt_batch
            if map_gt is None:
                raise ValueError("The map head needs the map elements of every sample.")
            # Samples that pad a streaming lane or lie between key frames take no part in the loss
            supervised = multi_task_batch_inputs.multi_task_gt_batch.supervised_mask().to(
                multi_task_outputs.online_map().classification.device
            )
            online_map_metrics = self.online_map_head.loss(
                multi_task_outputs.online_map().select_samples(supervised),
                map_gt.select_samples(supervised),
            )
            weighted_online_map_loss = self.online_map_loss_weight * online_map_metrics["loss"]
            metrics["weighted_online_map_loss"] = weighted_online_map_loss
            metrics["loss"] = metrics["loss"] + weighted_online_map_loss
            metrics.update(
                {f"online_map_{name}": value for name, value in online_map_metrics.items()}
            )
        self.log_step_metrics(metrics, multi_task_batch_inputs)
        return metrics

    def agent_future_targets(self, batch: ModelBatchInputs) -> AgentFutureTargets:
        """Align the agent futures of the batch with its box slots, at the future steps of the head.

        Args:
            batch: Model inputs with the ground truth.

        Returns:
            The future of the agent of every box slot.

        Raises:
            ValueError: If the batch carries no agent tracks, or tracks of other future steps
                than the head.
        """
        motion_gt = batch.multi_task_gt_batch.motion_gt_batch
        if motion_gt is None:
            raise ValueError("The motion head needs the agent tracks of every sample.")
        targets = AgentFutureTargets.from_agent_tracks(
            motion_gt.agents, detection3d_gt_batch(batch).gt_bboxes_instance_ids
        )
        steps = targets.agent_future_valid.shape[-1]
        if steps != self.motion_head.agent_steps:
            raise ValueError(
                f"The batch carries futures of {steps} steps, the motion head predicts "
                f"{self.motion_head.agent_steps}."
            )
        return targets

    def motion_loss(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        motion_outputs: MotionHeadOutputs,
        det3d_targets: list[Sparse4DTargets],
        det3d_mask: Bool[torch.Tensor, " batch_size"],
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Supervise the motion head on the frames with boxes.

        Args:
            multi_task_batch_inputs: Model inputs with the ground truth.
            motion_outputs: Raw outputs of the motion head.
            det3d_targets: Detection targets of every decoder layer over the frames with boxes.
            det3d_mask: Mask of the frames with boxes.

        Returns:
            The motion loss terms and their sum under ``loss``.

        Raises:
            ValueError: If the batch carries no futures of the steps of the head.
        """
        targets = self.agent_future_targets(multi_task_batch_inputs)
        gt_detections = detection3d_gt_batch(multi_task_batch_inputs)
        return self.motion_head.loss(
            motion_outputs.select_samples(det3d_mask),
            det3d_targets[-1].assigned_instance,
            gt_detections.gt_bboxes_3d[det3d_mask],
            gt_detections.gt_bboxes_instance_ids[det3d_mask],
            targets.select_samples(det3d_mask),
        )

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> EvalOutput:
        """Produce the detection, segmentation, motion and online map eval data.

        Args:
            batch: Model inputs of the evaluation step.
            outputs: Raw outputs from :meth:`forward`.

        Returns:
            The eval output of the detection, segmentation, motion and online map suites, the
            online map frames when the map branch is supervised.
        """
        eval_output = super().build_eval_output(batch, outputs)
        detection = eval_output.detection3d
        update = {
            "motion": MotionEvalOutput(frames=self.motion_eval_frames(batch, outputs, detection))
        }
        if self.online_map_head is not None and self.online_map_supervised:
            update["online_map"] = OnlineMapEvalOutput(
                frames=self.map_eval_frames(batch, outputs, detection)
            )
        return eval_output.model_copy(update=update)

    def map_eval_frames(
        self,
        batch: ModelBatchInputs,
        outputs: ModelOutputs,
        detection: Detection3DEvalOutput | None,
    ) -> list[OnlineMapEvalFrame]:
        """Pair the decoded map elements with the ground truth elements of every frame.

        Args:
            batch: Model inputs with the map elements.
            outputs: Raw outputs from :meth:`forward`.
            detection: Detection frames of the batch, which carry whether the metrics score the
                frame.

        Returns:
            One online map frame per sample of the batch.

        Raises:
            ValueError: If the batch carries no map elements or no detection frames.
        """
        map_gt = batch.multi_task_gt_batch.online_map_gt_batch
        if map_gt is None:
            raise ValueError("The map metrics need the map elements of every sample.")
        if detection is None:
            raise ValueError("The map metrics score the frames the detection frames score.")
        decoded = self.online_map_head.decode_outputs(outputs.online_map())
        frames = []
        for index, predictions in enumerate(decoded):
            valid = map_gt.labels[index] >= 0
            frames.append(
                OnlineMapEvalFrame(
                    pred_points=predictions.points.detach(),
                    pred_scores=predictions.scores.detach(),
                    pred_labels=predictions.labels,
                    gt_points=map_gt.points[index, valid],
                    gt_labels=map_gt.labels[index, valid],
                    scored=detection.frames[index].scored,
                )
            )
        return frames

    def motion_eval_frames(
        self,
        batch: ModelBatchInputs,
        outputs: ModelOutputs,
        detection: Detection3DEvalOutput | None,
    ) -> list[MotionEvalFrame]:
        """Pair the trajectories of the most confident queries with the futures of every frame.

        Args:
            batch: Model inputs with the ground truth boxes and futures.
            outputs: Raw outputs from :meth:`forward`.
            detection: Detection frames of the batch, which carry the observation flags of the
                boxes and whether the metrics score the frame.

        Returns:
            One motion frame per sample of the batch.

        Raises:
            ValueError: If the batch carries no futures of the steps of the head, or the
                detection frames carry no observation flags.
        """
        if detection is None or not detection.check_all_gt_observed:
            raise ValueError(
                "The motion metrics need the observation flags of the boxes, the temporal head "
                "attaches them to the detection frames."
            )
        targets = self.agent_future_targets(batch)
        det3d_outputs = outputs.detection3d().sparse4d_head()
        motion_outputs = outputs.motion()
        class_scores = self.bbox_head.query_scores(det3d_outputs).detach()
        scores, labels = class_scores.max(dim=-1)
        anchor = det3d_outputs.prediction[:, -1].detach()
        top = scores.topk(min(self.motion_eval_queries, scores.shape[1]), dim=1).indices
        gt_detections = detection3d_gt_batch(batch)
        # A frame without boxes is empty for the motion metrics, as for the detection metrics
        with_boxes = detection3d_frame_mask(gt_detections).tolist()
        frames = []
        for index, count in enumerate(gt_detections.gt_valid_bboxes.tolist()):
            queries = top[index] if with_boxes[index] else top[index, :0]
            boxes = gt_detections.gt_bboxes_3d[index, :count]
            detection_frame = detection.frames[index]
            frames.append(
                MotionEvalFrame(
                    pred_xy=anchor[index, queries, :2],
                    pred_labels=labels[index, queries],
                    pred_scores=scores[index, queries],
                    pred_trajectories=anchor[index, queries, None, None, :2]
                    + motion_outputs.agent_trajectories[index, queries].detach(),
                    pred_mode_scores=motion_outputs.agent_mode_logits[index, queries]
                    .detach()
                    .float()
                    .softmax(dim=-1),
                    pred_velocities=anchor[index, queries][:, _VELOCITY],
                    gt_xy=boxes[:, :2],
                    gt_labels=gt_detections.gt_labels_3d[index, :count],
                    gt_futures=targets.agent_future_positions[index, :count],
                    gt_valid=targets.agent_future_valid[index, :count],
                    gt_observed=detection_frame.gt_observed,
                    scored=detection_frame.scored,
                )
            )
        return frames

    def det3d_export_output_names(self) -> tuple[str, ...]:
        """Frame outputs of the detection graph, the instance features for the motion graph too."""
        return (*super().det3d_export_output_names(), "instance_feature")

    def build_context_export_specs(self, context: PTv3ExportContext) -> dict[str, ExportSpec]:
        """Build the split export specs, the detection graph handing the motion graph its queries.

        Args:
            context: Shared export context.

        Returns:
            The specs of the base model, the detection graph with the instance features as an
            extra output, the map graph of a model with a map head and the motion graph.
        """
        specs = super().build_context_export_specs(context)
        det_module = specs["ptv3_det3d_head"].module
        online_map_head = None
        if self.online_map_head is not None:
            online_map_head = self.online_map_head.prepare_for_export()
            polylines = ONLINE_MAP_POLYLINE_OUTPUT_NAMES if self.export_online_map else ()
            specs["ptv3_online_map_head"] = build_sparse4d_head_export_spec(
                context,
                _OnlineMapExportModule(
                    self.det_fusion,
                    online_map_head,
                    (
                        *polylines,
                        *ONLINE_MAP_CONTEXT_OUTPUT_NAMES,
                        *ONLINE_MAP_STATE_OUTPUT_NAMES,
                    ),
                ),
            )
        specs["ptv3_motion_head"] = self.build_motion_export_spec(
            context, det_module.bbox_head, online_map_head
        )
        return specs

    def build_motion_export_spec(
        self,
        context: PTv3ExportContext,
        bbox_head: Sparse4DHead,
        online_map_head: OnlineMapHead | None,
    ) -> ExportSpec:
        """Build the motion export spec, one frame with the history at the graph boundary.

        Args:
            context: Shared export context.
            bbox_head: Export-prepared detection head of the detection graph.
            online_map_head: Export-prepared map head copy sizing the map context, exactly for a
                motion head with a map.

        Returns:
            Export spec whose graph takes the detection outputs of the frame, the history of the
            last frame, the slot index the detection graph returned for the last frame, the
            stream continuity, the frame to frame transform and, with a map, the outputs of the
            map graph.
        """
        motion_head = self.motion_head.prepare_for_export()
        device = context.stage_feats[0].device
        bank = bbox_head.instance_bank
        num_queries = bank.num_anchor
        args = (
            torch.zeros(1, num_queries, bbox_head.embed_dims, device=device),
            bank.anchor.detach()[None].to(device),
            torch.zeros(1, num_queries, bbox_head.num_classes, device=device),
            *motion_head.export_state_inputs(num_queries, device),
            torch.zeros(1, bank.num_temp_instances, dtype=torch.long, device=device),
            torch.zeros(1, device=device),
            torch.eye(4, device=device)[None],
        )
        map_args, map_names = map_export_inputs(online_map_head, device)
        return ExportSpec(
            module=_MotionExportModule(motion_head, bbox_head).eval(),
            args=(*args, *map_args),
            input_names=[*MOTION_EXPORT_INPUT_NAMES, *map_names],
            output_names=list(MOTION_EXPORT_OUTPUT_NAMES),
            dynamic_axes=None,
            supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
        )


def map_export_inputs(
    online_map_head: OnlineMapHead | None, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], tuple[str, ...]]:
    """Example context outputs of the map graph a motion graph with a map takes, none without."""
    if online_map_head is None:
        return (), ()
    num_queries = online_map_head.instance_bank.num_anchor
    args = (
        torch.zeros(1, num_queries, online_map_head.embed_dims, device=device),
        torch.zeros(1, num_queries, online_map_head.embed_dims, device=device),
        torch.zeros(1, num_queries, device=device),
    )
    return args, MOTION_MAP_INPUT_NAMES


def online_map_context(
    anchor_encoder: nn.Module,
    feature: torch.Tensor,
    prediction: torch.Tensor,
    classification: torch.Tensor,
) -> MapContext:
    """Map context of the last layer map queries.

    Args:
        anchor_encoder: Anchor encoder of the map head.
        feature: Features of the map queries ``(B, Q, C)``.
        prediction: Their polylines ``(B, Q, 2 * P)``.
        classification: Their class logits ``(B, Q, K)``.

    Returns:
        Their features, polyline embeddings and best class scores.
    """
    return MapContext(
        feature=feature,
        anchor_embed=anchor_encoder(prediction),
        scores=classification.sigmoid().max(dim=-1).values,
    )


def export_map_context(map_inputs: Sequence[torch.Tensor]) -> MapContext | None:
    """Map context from the map graph outputs a motion graph takes, None without a map."""
    if not map_inputs:
        return None
    return MapContext(*map_inputs)


class _OnlineMapExportModule(PTv3Sparse4DHeadExportModule):
    """Export-only online map head returning the context the motion graph attends to."""

    frame_output_names = (*Sparse4DFrameOutputs._fields, "anchor_embed", "score")

    def forward(
        self, *inputs: Shaped[torch.Tensor, "..."]
    ) -> tuple[Shaped[torch.Tensor, "..."], ...]:
        """Decode one frame of map queries and return the configured outputs."""
        outputs = self.frame_outputs(*inputs)
        context = online_map_context(
            self.bbox_head.anchor_encoder,
            outputs.instance_feature,
            outputs.prediction,
            outputs.classification,
        )
        named = {
            **outputs._asdict(),
            "anchor_embed": context.anchor_embed,
            "score": context.scores,
        }
        return tuple(named[name] for name in self.output_names)


class _MotionExportModule(nn.Module):
    """Export-only motion head, one frame with its history at the graph boundary."""

    def __init__(
        self,
        motion_head: MotionHead,
        bbox_head: Sparse4DHead,
    ) -> None:
        """Initialize the export module.

        Args:
            motion_head: Export-prepared motion head.
            bbox_head: Export-prepared detection head lending its anchor encoder.
        """
        super().__init__()
        self.motion_head = motion_head
        self.bbox_head = bbox_head

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        prediction: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        classification: Float[torch.Tensor, "batch_size num_queries num_classes"],
        queue_feature: Float[torch.Tensor, "batch_size history_length num_queries embed_dims"],
        queue_anchor: Float[torch.Tensor, "batch_size history_length num_queries anchor_dim"],
        queue_map: Int64[torch.Tensor, "batch_size history_length num_queries"],
        queue_count: Int64[torch.Tensor, " batch_size"],
        queue_age: Int64[torch.Tensor, " batch_size"],
        carried_index: Int64[torch.Tensor, "batch_size num_temp_instances"],
        temp_valid: Float[torch.Tensor, " batch_size"],
        prev2current: Float[torch.Tensor, "batch_size 4 4"],
        *map_inputs: Float[torch.Tensor, "..."],
    ) -> tuple[Shaped[torch.Tensor, "..."], ...]:
        """Predict the trajectories of one frame, see the input names.

        The map context outputs of the map graph follow the history inputs exactly for a motion
        head with a map.
        """
        outputs, state = self.motion_head.step(
            instance_feature,
            prediction,
            classification,
            self.bbox_head.anchor_encoder,
            QueueState(queue_feature, queue_anchor, queue_map, queue_count, queue_age),
            temp_valid > 0.5,
            carried_index,
            prev2current,
            export_map_context(map_inputs),
        )
        return (outputs.agent_mode_logits, outputs.agent_trajectories, *state)
