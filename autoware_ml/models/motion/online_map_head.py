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

"""Online mapping head: polyline queries decoded over the sparse encoder stages.

Every query refines a polyline of map points and classifies it, the Sparse4D decoder running on
polyline anchors (Sun et al., SparseDrive, 2024).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Float

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.dataclasses.batch.online_map import OnlineMapGTBatch
from autoware_ml.dataclasses.models.motion.online_map_outputs import OnlineMapHeadOutputs
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.detection3d.heads.sparse4d import Refinement, SparseQueryHead
from autoware_ml.models.detection3d.ptv3 import SparseStage
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost
from autoware_ml.models.motion.online_map_blocks import (
    PolylineInstanceBank,
    SparsePolylineEncoder,
    SparsePolylineKeyPointsGenerator,
    SparsePolylineRefinementModule,
)
from autoware_ml.models.motion.online_map_matching import (
    PolylineTargets,
    assign_polylines,
    normalize_polylines,
)
from autoware_ml.utils.distributed import reduce_mean_count


class MapSamplePredictions(NamedTuple):
    """Decoded map elements of one sample, best first.

    Attributes:
        points: Points of every element ``(K, num_points, 2)`` in meters.
        scores: Class score of every element ``(K,)``.
        labels: Class of every element ``(K,)``.
    """

    points: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class OnlineMapHead(SparseQueryHead):
    """Polyline query head sampling the sparse PTv3 encoder stages."""

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dims: int,
        anchor_recipe: AnchorRecipe,
        num_anchor: int,
        num_points: int,
        roi: Sequence[float],
        point_cloud_range: Sequence[float],
        grid_size: float,
        feature_strides: Sequence[int],
        num_temp_instances: int,
        num_decoder: int = 6,
        num_heads: int = 8,
        num_groups: int = 8,
        feedforward_channels: int = 1024,
        dropout: float = 0.1,
        fix_heights: Sequence[float] = (0.0, 0.5),
        num_learnable_pts: int = 2,
        attn_drop: float = 0.15,
        cls_cost_weight: float = 1.0,
        line_cost_weight: float = 10.0,
        loss_cls_weight: float = 1.0,
        loss_line_weight: float = 10.0,
        line_beta: float = 0.01,
        num_output: int = 100,
        default_time_interval: float = 0.1,
        min_time_interval: float = 0.05,
        max_time_interval: float = 2.0,
        confidence_decay: float = 0.36,
    ) -> None:
        """Initialize the map head.

        Args:
            num_classes: Number of map classes.
            embed_dims: Instance feature dimension, the feature width of the sampled stages.
            anchor_recipe: Recipe of the polyline anchors ``(num_anchor, 2 * num_points)``.
            num_anchor: Number of polyline queries per frame.
            num_points: Points of every polyline.
            roi: Mapped region ``(x_min, y_min, x_max, y_max)`` the losses normalize by.
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_strides: Total pooling strides of the sampled stages.
            num_temp_instances: Queries propagated across frames.
            num_decoder: Number of decoder blocks.
            num_heads: Attention heads of the instance attention.
            num_groups: Feature groups of the deformable aggregation.
            feedforward_channels: Hidden width of the FFN blocks.
            dropout: Dropout probability in the attention and FFN blocks.
            fix_heights: Heights of the keypoint layers in the frame.
            num_learnable_pts: Keypoints per polyline point and height.
            attn_drop: Keypoint dropout of the aggregation in training.
            cls_cost_weight: Classification cost weight of the matching.
            line_cost_weight: Line distance weight of the matching.
            loss_cls_weight: Classification loss weight.
            loss_line_weight: Line loss weight.
            line_beta: Smooth L1 transition point of the line distance on normalized points.
            num_output: Largest number of decoded elements per sample.
            default_time_interval: Time interval in seconds without history.
            min_time_interval: Smallest frame gap that keeps the state of a slot.
            max_time_interval: Largest frame gap that keeps the state of a slot.
            confidence_decay: Decay of the cached query confidence per second.

        Raises:
            ValueError: If the region is empty.
        """
        super().__init__()
        x_min, y_min, x_max, y_max = (float(value) for value in roi)
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(f"roi must be (x_min, y_min, x_max, y_max), got {roi}.")
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.roi = (x_min, y_min, x_max, y_max)
        self.loss_cls_weight = loss_cls_weight
        self.loss_line_weight = loss_line_weight
        self.line_cost_weight = line_cost_weight
        self.line_beta = line_beta
        self.num_output = num_output
        self.instance_bank = PolylineInstanceBank(
            num_points=num_points,
            num_anchor=num_anchor,
            embed_dims=embed_dims,
            anchor_recipe=anchor_recipe,
            num_temp_instances=num_temp_instances,
            default_time_interval=default_time_interval,
            min_time_interval=min_time_interval,
            max_time_interval=max_time_interval,
            confidence_decay=confidence_decay,
        )
        self._build_decoder(
            embed_dims=embed_dims,
            num_decoder=num_decoder,
            num_heads=num_heads,
            num_groups=num_groups,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
            attn_drop=attn_drop,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            feature_strides=feature_strides,
            anchor_encoder=SparsePolylineEncoder(embed_dims, num_points),
            kps_generator=lambda: SparsePolylineKeyPointsGenerator(
                embed_dims=embed_dims,
                num_points=num_points,
                fix_heights=fix_heights,
                num_learnable_pts=num_learnable_pts,
            ),
            refine=lambda: SparsePolylineRefinementModule(
                embed_dims=embed_dims, num_points=num_points, num_classes=num_classes
            ),
        )
        self.cls_cost = ClassificationCost(weight=cls_cost_weight)
        self.loss_cls = SigmoidFocalLoss()

    def _refine(
        self,
        block_index: int,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_queries polyline_dims"],
        anchor_embed: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        time_interval: Float[torch.Tensor, " batch_size"],
    ) -> Refinement:
        """Refine the polylines of one block, see :meth:`SparseQueryHead._refine`."""
        anchor, cls_logits = self.refine[block_index](instance_feature, anchor, anchor_embed)
        return Refinement(anchor=anchor, cls_logits=cls_logits, quality=None)

    def reset(self) -> None:
        """Forget the cached map instances of every lane."""
        self.instance_bank.reset()

    def forward(
        self,
        stages: Sequence[SparseStage],
        timestamps_seconds: Float[torch.Tensor, " batch_size"],
        lidar2global: Float[torch.Tensor, "batch_size 4 4"],
        scene_tokens: Sequence[str],
    ) -> OnlineMapHeadOutputs:
        """Decode the polylines of every frame over the sampled sparse encoder stages.

        Args:
            stages: Features, voxel coordinates and batch offsets of the sampled stages, from
                the finest to the coarsest.
            timestamps_seconds: Frame timestamps ``(B,)`` in seconds.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in.
            scene_tokens: Scene identifiers of the frames.

        Returns:
            Per layer outputs of every query.
        """
        state = self.instance_bank.get(
            stages[0].offset.shape[0], timestamps_seconds, lidar2global, scene_tokens
        )
        decoded = self._decode(self.levels(stages), state, None, self.bev_levels(stages))
        self.instance_bank.cache(
            decoded.instance_feature, decoded.anchor, decoded.cls_logits, lidar2global
        )
        return OnlineMapHeadOutputs(
            classification=torch.stack(decoded.classification, dim=1),
            prediction=torch.stack(decoded.prediction, dim=1),
            instance_feature=decoded.instance_feature,
        )

    def assign(
        self, outputs: OnlineMapHeadOutputs, ground_truth: OnlineMapGTBatch
    ) -> list[PolylineTargets]:
        """Match the queries of every layer to the map elements.

        Args:
            outputs: Raw head outputs from :meth:`forward`.
            ground_truth: Map elements of the batch.

        Returns:
            Targets of every decoder layer.
        """
        return [
            assign_polylines(
                outputs.classification[:, layer],
                outputs.prediction[:, layer].unflatten(-1, (self.num_points, 2)),
                ground_truth,
                self.roi,
                self.cls_cost,
                self.line_cost_weight,
                self.line_beta,
            )
            for layer in range(outputs.classification.shape[1])
        ]

    def loss(
        self, outputs: OnlineMapHeadOutputs, ground_truth: OnlineMapGTBatch
    ) -> dict[str, torch.Tensor]:
        """Compute the classification and line losses of every layer.

        Args:
            outputs: Raw head outputs from :meth:`forward`.
            ground_truth: Map elements of the batch.

        Returns:
            Loss dictionary with the terms of every layer and their sum under ``loss``.
        """
        loss_dict: dict[str, torch.Tensor] = {}
        targets = self.assign(outputs, ground_truth)
        num_layers = len(targets)
        for layer, layer_targets in enumerate(targets):
            avg_factor = float(
                reduce_mean_count(outputs.classification.new_tensor(float(layer_targets.num_pos)))
            )
            cls_logits = outputs.classification[:, layer]
            loss_cls = self.loss_cls(
                cls_logits.flatten(end_dim=1).float(),
                layer_targets.cls_targets.flatten(end_dim=1),
                avg_factor=avg_factor,
            )
            points = normalize_polylines(
                outputs.prediction[:, layer].unflatten(-1, (self.num_points, 2)).float(), self.roi
            )
            line_error = F.smooth_l1_loss(
                points, layer_targets.point_targets, reduction="none", beta=self.line_beta
            ).sum(dim=-1)
            loss_line = (line_error.mean(dim=-1) * layer_targets.matched).sum() / avg_factor
            prefix = "layer_-1" if layer == num_layers - 1 else f"layer_{layer}"
            loss_dict[f"{prefix}_loss_cls"] = self.loss_cls_weight * loss_cls
            loss_dict[f"{prefix}_loss_line"] = self.loss_line_weight * loss_line
        loss_dict["num_pos"] = outputs.classification.new_tensor(
            sum(layer_targets.num_pos for layer_targets in targets) / max(num_layers, 1)
        )
        loss_dict["loss"] = sum(value for key, value in loss_dict.items() if "loss" in key)
        return loss_dict

    def decode_outputs(self, outputs: OnlineMapHeadOutputs) -> list[MapSamplePredictions]:
        """Keep the best scored query and class pairs of the last layer of every sample.

        Args:
            outputs: Raw head outputs from :meth:`forward`.

        Returns:
            The decoded elements of every sample.
        """
        scores = outputs.classification[:, -1].sigmoid().flatten(start_dim=1)
        top_scores, top_index = scores.topk(min(self.num_output, scores.shape[1]), dim=1)
        queries = top_index // self.num_classes
        points = outputs.prediction[:, -1].unflatten(-1, (self.num_points, 2))
        return [
            MapSamplePredictions(
                points=points[sample, queries[sample]],
                scores=top_scores[sample],
                labels=top_index[sample] % self.num_classes,
            )
            for sample in range(scores.shape[0])
        ]
