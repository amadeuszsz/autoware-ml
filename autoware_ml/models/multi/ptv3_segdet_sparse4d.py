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

"""PTv3 joint segmentation and Sparse4D detection model.

A temporal head needs scenario streams, see ``DataModule(streaming=True)``. Its export
carries the instance bank state through the inputs and the outputs of the detection graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.detection3d.heads.sparse4d import (
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
    TEMPORAL_STATE_INPUT_NAMES,
    TEMPORAL_STATE_OUTPUT_NAMES,
    Sparse4DHead,
)
from autoware_ml.models.detection3d.ptv3 import (
    PTv3DetFeatureFusion,
    det_head_export_args,
    det_head_export_dynamic_axes,
    det_head_export_input_names,
    rebuild_coarse_point,
)
from autoware_ml.models.multi.ptv3_segdet import (
    detection_frame_mask,
    detection_gt_batch,
    empty_unsupervised_predictions,
    mask_samples,
)
from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder
from autoware_ml.models.segmentation3d.encoders.voxel import VoxelFeatureEncoder
from autoware_ml.models.segmentation3d.heads.ptv3 import (
    PTv3SegDecoderHead,
    segmentation_eval_output,
    segmentation_point_loss,
    segmentation_predict_outputs,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    PTv3ExportContext,
    build_encoder_export_spec,
    build_ptv3_export_context,
    build_seg_head_export_spec,
    split_block_parameters,
)
from autoware_ml.utils.deploy import ExportSpec

SPARSE4D_EXPORT_OUTPUT_NAMES = (
    "classification",
    "prediction",
    "quality",
    *TEMPORAL_STATE_OUTPUT_NAMES,
    TEMPORAL_QUERY_INDEX_OUTPUT_NAME,
)


class PTv3SegDetSparse4DModel(PTv3BaseModel):
    """PTv3 joint segmentation and Sparse4D detection model."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        seg3d_head: PTv3SegDecoderHead,
        det_fusion: PTv3DetFeatureFusion,
        bbox_head: Sparse4DHead,
        segmentation_loss_weight: float = 1.0,
        detection_loss_weight: float = 1.0,
        grid_size: float | None = None,
        point_cloud_range: Sequence[float] | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 joint segmentation and Sparse4D detection model.

        Args:
            encoder: PTv3 encoder module shared by both branches.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            seg3d_head: Segmentation decoder head owning losses and the
                classifier.
            det_fusion: Fusion of the deepest encoder stage into its parent
                stage, whose sparse voxels the detection head samples.
            bbox_head: Sparse4D query head.
            segmentation_loss_weight: Weight of the segmentation loss term.
            detection_loss_weight: Weight of the detection loss term.
            grid_size: Voxel grid size of the encoder input.
            point_cloud_range: Point-cloud range of the encoder input.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Per-group optimizer overrides.
            scheduler_config: Lightning scheduler metadata.
            **kwargs: Keyword arguments forwarded up the MRO chain.
        """
        super().__init__(
            encoder=encoder,
            voxel_encoder=voxel_encoder,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            **kwargs,
        )
        self.seg3d_head = seg3d_head
        self.det_fusion = det_fusion
        self.bbox_head = bbox_head
        self.segmentation_loss_weight = float(segmentation_loss_weight)
        self.detection_loss_weight = float(detection_loss_weight)

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group pretrained and newly initialized joint-task parameters."""
        encoder_default_params, encoder_block_params = split_block_parameters(self.encoder)
        seg3d_head_params = [
            parameter for parameter in self.seg3d_head.parameters() if parameter.requires_grad
        ]
        det3d_branch_params = [
            parameter
            for module in (self.det_fusion, self.bbox_head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return {
            "encoder_default": encoder_default_params,
            "encoder_block": encoder_block_params,
            "seg3d_head": seg3d_head_params,
            "det3d_branch": det3d_branch_params,
        }

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the voxels, the frame metadata and the boxes of the batch.

        The boxes feed the denoising queries, their instance ids and point counts the tracks.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.

        Returns:
            The keyword arguments of :meth:`forward`.

        Raises:
            ValueError: If the batch carries no frame metadata.
        """
        frame_meta = batch_inputs.multi_task_gt_batch.frame_meta_batch
        if frame_meta is None:
            raise ValueError("The Sparse4D head needs the frame metadata of every sample.")
        gt_detections = detection_gt_batch(batch_inputs)
        return super().forward_inputs(batch_inputs) | {
            "timestamps_seconds": frame_meta.timestamps_seconds,
            "ego2globals": frame_meta.ego2globals,
            "scene_tokens": frame_meta.scene_tokens,
            "gt_bboxes_3d": gt_detections.valid_bboxes_3d(),
            "gt_labels_3d": gt_detections.valid_labels_3d(),
            "gt_instance_ids": gt_detections.valid_bboxes_instance_ids(),
            "gt_num_points": gt_detections.valid_bboxes_num_points(),
        }

    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_dropped_voxels: torch.Tensor,
        time_lag_column: int,
        timestamps_seconds: torch.Tensor,
        ego2globals: torch.Tensor,
        scene_tokens: Sequence[str],
        gt_bboxes_3d: Sequence[torch.Tensor],
        gt_labels_3d: Sequence[torch.Tensor],
        gt_instance_ids: Sequence[torch.Tensor],
        gt_num_points: Sequence[torch.Tensor],
    ) -> ModelOutputs:
        """Run one shared PTv3 encoder pass and branch into both heads.

        Args:
            voxels: Padded voxel points from the data preprocessing.
            num_points: Valid point count per voxel.
            voxel_coords: Voxel coordinates with a leading batch column.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.
            timestamps_seconds: Per-frame lidar timestamps.
            ego2globals: Per-frame transforms from the augmented lidar frame to
                the map frame.
            scene_tokens: Per-frame scene identifiers.
            gt_bboxes_3d: Per-frame ground-truth boxes for the denoising queries.
            gt_labels_3d: Per-frame ground-truth labels for the denoising queries.
            gt_instance_ids: Per-frame instance ids of the ground-truth boxes.
            gt_num_points: Per-frame point counts of the ground-truth boxes.
        """
        point = self.encode(voxels, num_points, voxel_coords, num_dropped_voxels, time_lag_column)
        # Fuse before the decoder, SerializedUnpooling pops the encoder chain
        det_feat, det_grid_coord, det_offset = self.det_fusion(point)
        seg_logits = self.seg3d_head(point)
        det_outputs = self.bbox_head(
            det_feat,
            det_grid_coord,
            det_offset,
            timestamps_seconds=timestamps_seconds,
            lidar2global=ego2globals,
            scene_tokens=scene_tokens,
            gt_boxes=gt_bboxes_3d,
            gt_labels=gt_labels_3d,
            gt_instance_ids=gt_instance_ids,
            gt_num_points=gt_num_points,
        )
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=None,
                sparse4d_head_outputs=det_outputs,
            ),
            segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=seg_logits),
        )

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute combined segmentation and detection losses.

        Frames without ground-truth boxes skip the detection loss.
        """
        seg_logits = outputs.segmentation3d().logits
        det_outputs = outputs.detection3d().sparse4d_head()
        seg_metrics = segmentation_point_loss(self.seg3d_head, seg_logits, batch_inputs)

        gt_detections = detection_gt_batch(batch_inputs)
        det_mask = detection_frame_mask(gt_detections)
        if bool(det_mask.any()):
            det_metrics = self.bbox_head.loss(
                det_outputs.select_samples(det_mask),
                mask_samples(gt_detections.valid_bboxes_3d(), det_mask),
                mask_samples(gt_detections.valid_labels_3d(), det_mask),
                mask_samples(gt_detections.valid_bboxes_instance_ids(), det_mask),
                det_mask.nonzero().flatten().tolist(),
            )
        else:
            # Zero gradients keep the detection branch visible to DDP
            graph_tensors = [
                det_outputs.classification,
                det_outputs.prediction,
                det_outputs.quality,
            ]
            det_metrics = {"loss": sum(tensor.float().sum() for tensor in graph_tensors) * 0.0}

        seg_loss = seg_metrics["loss"]
        weighted_seg_loss = self.segmentation_loss_weight * seg_loss
        weighted_det_loss = self.detection_loss_weight * det_metrics["loss"]
        metrics: dict[str, torch.Tensor] = {
            "seg_loss_ce": seg_metrics["loss_ce"],
            "seg_loss_lovasz": seg_metrics["loss_lovasz"],
            "seg_loss": seg_loss,
            "weighted_seg_loss": weighted_seg_loss,
            "weighted_det_loss": weighted_det_loss,
            "loss": weighted_det_loss + weighted_seg_loss,
        }
        metrics.update({f"det_{name}": value for name, value in det_metrics.items()})
        if self.training:
            # Per-step values of rank 0 to watch the convergence within an epoch
            self.log_dict(
                {
                    f"train_step/{name}": value.detach()
                    for name, value in metrics.items()
                    if "_layer_" not in name or "_layer_-1_" in name
                },
                on_step=True,
                on_epoch=False,
                batch_size=self.get_log_batch_size(batch_inputs),
            )
        return metrics

    def build_eval_output(self, batch: ModelBatchInputs, outputs: ModelOutputs) -> dict[str, Any]:
        """Produce the detection and the current frame segmentation eval data.

        Frames without detection supervision contribute empty predictions. A temporal head adds
        which boxes were observed in the stream so far and the stream of every frame.
        """
        det_mask = detection_frame_mask(detection_gt_batch(batch))
        predictions = empty_unsupervised_predictions(
            self.bbox_head.predict(outputs.detection3d().sparse4d_head()), det_mask
        )
        eval_out = detection_eval_output(
            ModelPredictions(detection3d_predictions=predictions), batch
        )
        tracker = self.bbox_head.tracker
        if tracker is not None:
            eval_out["gt_observed"] = list(tracker.frame_observed)
            eval_out["stream_ids"] = list(tracker.stream_ids)
        eval_out.update(segmentation_eval_output(outputs.segmentation3d().logits, batch))
        return eval_out

    def predict_outputs(
        self, batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Decode the boxes and the point labels of the batch for inference.

        Args:
            batch_inputs: Model inputs holding the grid samples the logits were predicted for.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            The decoded boxes of every sample and the label of every original point.
        """
        return ModelPredictions(
            detection3d_predictions=self.bbox_head.predict(outputs.detection3d().sparse4d_head()),
            segmentation3d_predictions=segmentation_predict_outputs(
                outputs.segmentation3d().logits, batch_inputs
            ),
        )

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build the split ONNX export specs of the encoder, the seg head and the det head.

        Args:
            batch_inputs: Example preprocessed batch with the voxels.

        Returns:
            Export specs of the encoder, the segmentation head and the temporal detection head.

        Raises:
            ValueError: If the grid size or the point cloud range is missing.
        """
        if self.grid_size is None or self.point_cloud_range is None:
            raise ValueError(
                "grid_size and point_cloud_range must be provided at construction time to use "
                "export."
            )
        context = build_ptv3_export_context(self, batch_inputs)
        return {
            "ptv3_encoder": build_encoder_export_spec(context),
            "ptv3_seg3d_head": build_seg_head_export_spec(
                context,
                self.seg3d_head.prepare_for_export(self.EXPORT_ORDER),
                ["pred_labels", "pred_probs"],
            ),
            "ptv3_det3d_head": build_sparse4d_head_export_spec(
                context, self.det_fusion, self.bbox_head.prepare_for_export()
            ),
        }


def build_sparse4d_head_export_spec(
    context: PTv3ExportContext, det_fusion: PTv3DetFeatureFusion, bbox_head: Sparse4DHead
) -> ExportSpec:
    """Build the temporal Sparse4D head export spec from the two coarsest encoder stages.

    Args:
        context: Shared export context.
        det_fusion: Detection stage fusion (copied internally).
        bbox_head: Export-prepared temporal Sparse4D head copy.

    Returns:
        Export spec whose graph takes the bank state of the last frame and returns the next one
        with the query every state slot was selected from.
    """
    module = _PTv3Sparse4DHeadExportModule(deepcopy(det_fusion).eval(), bbox_head).eval()
    stage_args = det_head_export_args(context)
    return ExportSpec(
        module=module,
        args=(*stage_args, *bbox_head.temporal_state_export_inputs(stage_args[0].device)),
        input_param_names=[
            *det_head_export_input_names(context.stage_count),
            *TEMPORAL_STATE_INPUT_NAMES,
        ],
        output_names=list(SPARSE4D_EXPORT_OUTPUT_NAMES),
        dynamic_axes=det_head_export_dynamic_axes(context.stage_count),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


class _PTv3Sparse4DHeadExportModule(nn.Module):
    """Export-only temporal Sparse4D head consuming the two coarsest encoder stages."""

    def __init__(self, det_fusion: PTv3DetFeatureFusion, bbox_head: Sparse4DHead) -> None:
        """Initialize the export-only head module.

        Args:
            det_fusion: Export-ready detection stage fusion.
            bbox_head: Export-ready temporal Sparse4D head.
        """
        super().__init__()
        self.det_fusion = det_fusion
        self.bbox_head = bbox_head

    def forward(
        self,
        skip_feat: torch.Tensor,
        deepest_feat: torch.Tensor,
        cluster: torch.Tensor,
        skip_grid_coord: torch.Tensor,
        temp_instance_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
        temp_confidence: torch.Tensor,
        temp_valid: torch.Tensor,
        time_interval: torch.Tensor,
        prev2current: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Fuse the coarse stages and decode one frame with the bank state of the last one."""
        point = rebuild_coarse_point(skip_feat, deepest_feat, cluster, skip_grid_coord)
        feat, grid_coord, offset = self.det_fusion(point)
        outputs = self.bbox_head.forward_temporal_export(
            feat,
            grid_coord,
            offset,
            temp_instance_feature,
            temp_anchor,
            temp_confidence,
            temp_valid,
            time_interval,
            prev2current,
        )
        return tuple(outputs[name] for name in SPARSE4D_EXPORT_OUTPUT_NAMES)
