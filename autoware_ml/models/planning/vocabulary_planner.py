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

"""Ego planner scoring a vocabulary of paths and speed profiles coarse to fine.

Every path and every speed profile of a fixed vocabulary is a query of the ego vehicle. Path
queries sample the sparse lidar stages along their path and read the most confident agents,
speed queries read the agents. Every layer keeps its best paths and speed profiles and scores the
trajectories they compose, the last layer also samples the stages along those trajectories and
predicts whether they collide (Sun et al., SparseDrive).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.dataclasses.models.motion.head_outputs import PlanLayerOutputs
from autoware_ml.metrics.geometry.reachability import VehicleGeometry
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.anchored import DataAnchored, anchor_tensor
from autoware_ml.models.detection3d.heads.sparse4d import (
    SparseDeformableAggregation,
    SparseLevelGrid,
    linear_relu_ln,
)
from autoware_ml.models.detection3d.ptv3 import SparseStage
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.planning.commands import DrivingCommand
from autoware_ml.models.planning.geometry import boxes_overlap, to_frame
from autoware_ml.ops.deformable_aggregation.sparse_deformable_aggregation import SparseLevel
from autoware_ml.utils.distributed import reduce_mean_count


def compose_trajectories(
    paths: torch.Tensor, distances: torch.Tensor, speeds: torch.Tensor, step_seconds: float
) -> torch.Tensor:
    """Drive every path at every speed profile.

    The distance covered at every step follows the speed profile. Positions interpolate between
    the path points and extend the last path segment beyond its end.

    Args:
        paths: Path points ``(Kp, P, 2)`` at the arc lengths ``distances`` from the origin.
        distances: Increasing arc lengths of the path points ``(P,)``.
        speeds: Mean speed of every step of every profile ``(Kv, T)``.
        step_seconds: Duration of a step.

    Returns:
        Trajectories ``(Kp, Kv, T, 2)``.
    """
    knots = torch.cat([distances.new_zeros(1), distances])
    points = torch.cat([paths.new_zeros(paths.shape[0], 1, 2), paths], dim=1)
    travelled = (speeds * step_seconds).cumsum(dim=-1)
    segment = (torch.searchsorted(knots, travelled.contiguous(), right=True) - 1).clamp(
        0, knots.shape[0] - 2
    )
    start, end = knots[segment], knots[segment + 1]
    ratio = ((travelled - start) / (end - start))[None, ..., None]
    return points[:, segment] + ratio * (points[:, segment + 1] - points[:, segment])


class PathKeyPointsGenerator(nn.Module):
    """Keypoints along BEV paths at fixed heights above the ground."""

    def __init__(self, num_points: int, heights: Sequence[float]) -> None:
        """Initialize the generator.

        Args:
            num_points: Points of every path.
            heights: Heights of the keypoints above every path point.
        """
        super().__init__()
        self.num_points = num_points
        self.register_buffer("heights", torch.tensor(list(heights)), persistent=False)
        self.num_pts = num_points * len(heights)

    def forward(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
        """Keypoints ``(B, N, P * H, 3)`` of the flattened paths ``(B, N, P * 2)``."""
        batch_size, num_paths = anchor.shape[:2]
        num_heights = self.heights.shape[0]
        points = anchor.reshape(batch_size, num_paths, self.num_points, 1, 2)
        points = points.expand(-1, -1, -1, num_heights, -1)
        heights = self.heights.to(anchor.dtype).view(1, 1, 1, num_heights, 1)
        heights = heights.expand(batch_size, num_paths, self.num_points, -1, -1)
        return torch.cat([points, heights], dim=-1).reshape(batch_size, num_paths, self.num_pts, 3)


class ResidualAttention(nn.Module):
    """Multi-head attention added to its queries, the keys default to the queries."""

    def __init__(self, embed_dims: int, num_heads: int, dropout: float) -> None:
        """Initialize the block.

        Args:
            embed_dims: Feature width.
            num_heads: Attention heads.
            dropout: Attention and residual dropout.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor | None = None) -> torch.Tensor:
        """Attend from ``query`` ``(B, N, C)`` to ``key`` ``(B, K, C)``."""
        key = query if key is None else key
        return query + self.dropout(self.attn(query, key, key)[0])


def _score_head(embed_dims: int) -> nn.Sequential:
    """MLP scoring every query with one logit, starting near a probability of 0.02."""
    head = nn.Sequential(*linear_relu_ln(embed_dims, 2, 2, embed_dims), nn.Linear(embed_dims, 1))
    nn.init.constant_(head[-1].bias, -math.log((1.0 - 0.02) / 0.02))
    return head


class _Candidates(NamedTuple):
    """Queries of one vocabulary part a layer refines.

    Attributes:
        query: Query features ``(B, N, C)``.
        embed: Embeddings of their vocabulary entries ``(B, N, C)``.
        ids: Vocabulary index of every query ``(B, N)``.
    """

    query: torch.Tensor
    embed: torch.Tensor
    ids: torch.Tensor

    def keep(self, logits: torch.Tensor, count: int) -> _Candidates:
        """Keep the ``count`` best scored queries."""
        index = logits.topk(count, dim=1).indices
        features = index[..., None].expand(-1, -1, self.query.shape[-1])
        return _Candidates(
            self.query.gather(1, features),
            self.embed.gather(1, features),
            self.ids.gather(1, index),
        )


class _LayerStep(NamedTuple):
    """Scores of the paths and speed profiles of one layer and the trajectories it composed.

    Attributes:
        path_logits: Score logit of every path the layer got ``(B, Np)``.
        path_ids: Vocabulary index of those paths ``(B, Np)``.
        speed_logits: Score logit of every speed profile the layer got ``(B, Nv)``.
        speed_ids: Vocabulary index of those speed profiles ``(B, Nv)``.
        query: Queries of the trajectories of the kept paths and speed profiles ``(B, Nt, C)``.
        ids: Trajectory vocabulary index of the same trajectories ``(B, Nt)``.
    """

    path_logits: torch.Tensor
    path_ids: torch.Tensor
    speed_logits: torch.Tensor
    speed_ids: torch.Tensor
    query: torch.Tensor
    ids: torch.Tensor

    def scores(self, trajectory_logits: torch.Tensor) -> PlanLayerOutputs:
        """Scores of the layer with the logits of its trajectories ``(B, Nt)``."""
        return PlanLayerOutputs(
            path_logits=self.path_logits,
            path_ids=self.path_ids,
            speed_logits=self.speed_logits,
            speed_ids=self.speed_ids,
            trajectory_logits=trajectory_logits,
            trajectory_ids=self.ids,
        )


class PlanningLayer(nn.Module):
    """Refine the path and speed queries, keep the best and score the trajectories they compose."""

    def __init__(
        self,
        *,
        embed_dims: int,
        num_heads: int,
        num_groups: int,
        num_levels: int,
        dropout: float,
        path_points: int,
        trajectory_steps: int,
        keypoint_heights: Sequence[float],
        keep_paths: int,
        keep_speeds: int,
        with_map: bool,
        refine_trajectories: bool,
    ) -> None:
        """Initialize the layer.

        Args:
            embed_dims: Feature width.
            num_heads: Attention heads.
            num_groups: Feature groups of the stage sampling.
            num_levels: Sampled lidar stages.
            dropout: Attention and residual dropout.
            path_points: Points of every path.
            trajectory_steps: Steps of every trajectory.
            keypoint_heights: Heights the stages are sampled at above every path point.
            keep_paths: Best paths the layer keeps.
            keep_speeds: Best speed profiles the layer keeps.
            with_map: Whether the queries also read the map queries.
            refine_trajectories: Whether the composed trajectories sample the stages, attend to each
                other and predict collisions, as in the last layer.
        """
        super().__init__()
        self.keep_paths = keep_paths
        self.keep_speeds = keep_speeds
        self.path_aggregation = SparseDeformableAggregation(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            kps_generator=PathKeyPointsGenerator(path_points, keypoint_heights),
            proj_drop=dropout,
        )
        self.path_agent_attention = ResidualAttention(embed_dims, num_heads, dropout)
        self.speed_agent_attention = ResidualAttention(embed_dims, num_heads, dropout)
        self.path_map_attention = (
            ResidualAttention(embed_dims, num_heads, dropout) if with_map else None
        )
        self.speed_map_attention = (
            ResidualAttention(embed_dims, num_heads, dropout) if with_map else None
        )
        self.path_norm = nn.LayerNorm(embed_dims)
        self.speed_norm = nn.LayerNorm(embed_dims)
        self.path_self_attention = ResidualAttention(embed_dims, num_heads, dropout)
        self.speed_self_attention = ResidualAttention(embed_dims, num_heads, dropout)
        self.path_self_norm = nn.LayerNorm(embed_dims)
        self.speed_self_norm = nn.LayerNorm(embed_dims)
        self.path_score = _score_head(embed_dims)
        self.speed_score = _score_head(embed_dims)
        self.trajectory_score = _score_head(embed_dims)
        self.refine_trajectories = refine_trajectories
        if refine_trajectories:
            self.trajectory_aggregation = SparseDeformableAggregation(
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                kps_generator=PathKeyPointsGenerator(trajectory_steps, keypoint_heights),
                proj_drop=dropout,
            )
            self.trajectory_norm = nn.LayerNorm(embed_dims)
            self.trajectory_self_attention = ResidualAttention(embed_dims, num_heads, dropout)
            self.trajectory_self_norm = nn.LayerNorm(embed_dims)
            self.collision_score = _score_head(embed_dims)

    def refine_paths(
        self,
        paths: _Candidates,
        paths_in_frame: torch.Tensor,
        agents: torch.Tensor,
        map_keys: torch.Tensor | None,
        levels: Sequence[SparseLevel],
    ) -> torch.Tensor:
        """Sample the stages along every path and read the agents and the map.

        Args:
            paths: Path queries.
            paths_in_frame: Their points in the point cloud frame ``(B, N, P, 2)``.
            agents: Agent keys ``(B, K, C)``.
            map_keys: Map keys ``(B, M, C)``, exactly for a layer with a map.
            levels: Sampled lidar stages.

        Returns:
            The refined path queries ``(B, N, C)``.
        """
        query = self.path_aggregation(paths.query, paths_in_frame.flatten(2), paths.embed, levels)
        query = self.path_agent_attention(query, agents)
        if self.path_map_attention is not None:
            query = self.path_map_attention(query, map_keys)
        query = self.path_norm(query)
        return self.path_self_norm(self.path_self_attention(query))

    def refine_speeds(
        self, speeds: _Candidates, agents: torch.Tensor, map_keys: torch.Tensor | None
    ) -> torch.Tensor:
        """Read the agents and the map with every speed profile query ``(B, N, C)``."""
        query = self.speed_agent_attention(speeds.query, agents)
        if self.speed_map_attention is not None:
            query = self.speed_map_attention(query, map_keys)
        query = self.speed_norm(query)
        return self.speed_self_norm(self.speed_self_attention(query))

    def refine(
        self,
        query: torch.Tensor,
        embed: torch.Tensor,
        trajectories_in_frame: torch.Tensor,
        levels: Sequence[SparseLevel],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the stages along every trajectory, attend among them and score collisions.

        Args:
            query: Trajectory queries ``(B, N, C)``.
            embed: Embeddings of their paths and speed profiles ``(B, N, C)``.
            trajectories_in_frame: Their positions in the point cloud frame ``(B, N, T, 2)``.
            levels: Sampled lidar stages.

        Returns:
            The refined queries and their collision logits ``(B, N)``.
        """
        query = self.trajectory_aggregation(query, trajectories_in_frame.flatten(2), embed, levels)
        query = self.trajectory_norm(query)
        query = self.trajectory_self_norm(self.trajectory_self_attention(query))
        return query, self.collision_score(query).squeeze(-1)


class PlannerOutputs(NamedTuple):
    """Plans of the last layer and the scores of every layer.

    Attributes:
        logits: Score logit of every kept trajectory ``(B, N)``.
        trajectories: The kept trajectories in the point cloud frame ``(B, N, T, 2)``.
        collision_logits: Collision logit of every kept trajectory ``(B, N)``.
        layers: Scores of every layer.
    """

    logits: torch.Tensor
    trajectories: torch.Tensor
    collision_logits: torch.Tensor
    layers: tuple[PlanLayerOutputs, ...]


class VocabularyPlanner(DataAnchored):
    """Plan the ego vehicle by scoring a vocabulary of paths and speed profiles coarse to fine."""

    anchor_names = ("path_vocabulary", "speed_vocabulary", "trajectory_vocabulary")

    def __init__(
        self,
        *,
        embed_dims: int,
        anchor_recipe: AnchorRecipe,
        path_distances: Sequence[float],
        num_paths: int,
        num_speeds: int,
        ego_steps: int,
        step_seconds: float,
        keep: Sequence[Sequence[int]],
        keypoint_heights: Sequence[float],
        vehicle: VehicleGeometry,
        point_cloud_range: Sequence[float],
        grid_size: float,
        feature_strides: Sequence[int],
        num_heads: int = 8,
        num_groups: int = 8,
        dropout: float = 0.1,
        with_map: bool = False,
        collision_score_threshold: float = 0.5,
        collision_steps: int = 4,
        loss_path_weight: float = 0.25,
        loss_speed_weight: float = 0.25,
        loss_trajectory_weight: float = 0.5,
        loss_collision_weight: float = 1.0,
    ) -> None:
        """Initialize the planner.

        Args:
            embed_dims: Feature width of the ego query and the agent keys.
            anchor_recipe: Recipe of the vocabularies, ``paths`` ``(num_paths, P, 2)`` in the
                ego frame, their points at the arc lengths ``path_distances``, and ``speeds``
                ``(num_speeds, ego_steps)``, the mean speed of every step in m/s.
            path_distances: Increasing arc lengths of the path points in meters.
            num_paths: Paths of the vocabulary.
            num_speeds: Speed profiles of the vocabulary.
            ego_steps: Steps of the trajectories.
            step_seconds: Duration of a step.
            keep: Paths and speed profiles every layer keeps, one ``(paths, speeds)`` pair per
                layer, decreasing.
            keypoint_heights: Heights the stages are sampled at above the path and trajectory
                points.
            vehicle: Body of the ego vehicle, the frame origin at its rear axle.
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_strides: Total pooling strides of the sampled fused stages.
            num_heads: Attention heads.
            num_groups: Feature groups of the stage sampling.
            dropout: Attention and residual dropout.
            with_map: Whether the queries also read the map queries.
            collision_score_threshold: Detection score from which an agent blocks a trajectory.
            collision_steps: Leading steps a collision with an agent demotes a trajectory in.
            loss_path_weight: Weight of the path classification loss.
            loss_speed_weight: Weight of the speed profile classification loss.
            loss_trajectory_weight: Weight of the trajectory classification loss.
            loss_collision_weight: Weight of the collision loss.

        Raises:
            ValueError: If the keep counts do not shrink within the vocabulary, or the path
                distances do not increase.
        """
        super().__init__(anchor_recipe)
        counts = [(num_paths, num_speeds), *[tuple(pair) for pair in keep]]
        for (paths_in, speeds_in), (paths_out, speeds_out) in zip(counts, counts[1:]):
            if not (0 < paths_out <= paths_in and 0 < speeds_out <= speeds_in):
                raise ValueError(
                    f"Every layer must keep a positive share of what it gets, got {counts}."
                )
        distances = torch.tensor(list(path_distances), dtype=torch.float32)
        if (
            distances.ndim != 1
            or bool((distances[1:] <= distances[:-1]).any())
            or distances[0] <= 0
        ):
            raise ValueError(f"path_distances must increase from above 0, got {path_distances}.")
        if not 0 < collision_steps <= ego_steps:
            raise ValueError(f"collision_steps must lie in 1..{ego_steps}, got {collision_steps}.")
        self.num_speeds = num_speeds
        self.ego_steps = ego_steps
        self.level_grids = nn.ModuleList(
            [SparseLevelGrid(point_cloud_range, grid_size, stride) for stride in feature_strides]
        )
        self.vehicle = vehicle
        self.collision_score_threshold = collision_score_threshold
        self.collision_steps = collision_steps
        self.with_map = with_map
        self.loss_path_weight = loss_path_weight
        self.loss_speed_weight = loss_speed_weight
        self.loss_trajectory_weight = loss_trajectory_weight
        self.loss_collision_weight = loss_collision_weight
        self.step_seconds = step_seconds
        # Filled from the anchor recipe before training, see DataAnchored
        self.register_buffer("path_distances", distances, persistent=False)
        self.register_buffer("path_vocabulary", torch.zeros(num_paths, len(distances), 2))
        self.register_buffer("speed_vocabulary", torch.zeros(num_speeds, ego_steps))
        self.register_buffer(
            "trajectory_vocabulary", torch.zeros(num_paths * num_speeds, ego_steps, 2)
        )
        self.register_buffer(
            "ego_box_size",
            torch.tensor([vehicle.length, vehicle.width], dtype=torch.float32),
            persistent=False,
        )
        self.path_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, len(distances) * 2),
            nn.Linear(embed_dims, embed_dims),
        )
        self.speed_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, ego_steps), nn.Linear(embed_dims, embed_dims)
        )
        self.command_embedding = nn.Embedding(len(DrivingCommand), embed_dims)
        self.layers = nn.ModuleList(
            [
                PlanningLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_groups=num_groups,
                    num_levels=len(feature_strides),
                    dropout=dropout,
                    path_points=len(distances),
                    trajectory_steps=ego_steps,
                    keypoint_heights=keypoint_heights,
                    keep_paths=keep_paths,
                    keep_speeds=keep_speeds,
                    with_map=with_map,
                    refine_trajectories=index == len(keep) - 1,
                )
                for index, (keep_paths, keep_speeds) in enumerate(keep)
            ]
        )
        self.loss_cls = SigmoidFocalLoss()

    def load_anchors(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Copy the path and speed vocabularies of the recipe and compose their trajectories.

        Args:
            arrays: Anchor arrays of the recipe, ``paths`` and ``speeds``.
        """
        self.path_vocabulary.copy_(
            anchor_tensor(arrays, "paths", tuple(self.path_vocabulary.shape))
        )
        self.speed_vocabulary.copy_(
            anchor_tensor(arrays, "speeds", tuple(self.speed_vocabulary.shape))
        )
        self.trajectory_vocabulary.copy_(
            compose_trajectories(
                self.path_vocabulary, self.path_distances, self.speed_vocabulary, self.step_seconds
            ).flatten(0, 1)
        )

    def forward(
        self,
        ego_query: torch.Tensor,
        commands: torch.Tensor,
        agents: torch.Tensor,
        map_keys: torch.Tensor | None,
        stages: Sequence[SparseStage],
        frame_from_ego: torch.Tensor,
    ) -> PlannerOutputs:
        """Score the vocabulary layer by layer and return the trajectories of the last one.

        Args:
            ego_query: Ego query of every frame ``(B, C)``.
            commands: Driving command of every frame ``(B,)``.
            agents: Agent keys, features with their anchor embeddings ``(B, K, C)``.
            map_keys: Map keys ``(B, M, C)``, exactly for a planner with a map.
            stages: Fused lidar stages at the feature strides of the planner.
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame
                ``(B, 4, 4)``.

        Returns:
            The kept trajectories, their scores and collision logits and the scores of every
            layer.

        Raises:
            ValueError: If the map keys are given to a planner without a map or missing for one
                with a map.
        """
        if (map_keys is None) == self.with_map:
            raise ValueError("The map keys must be given exactly to a planner with a map.")
        levels = [grid(*stage) for grid, stage in zip(self.level_grids, stages, strict=True)]
        batch_size = ego_query.shape[0]
        context = (ego_query + self.command_embedding(commands))[:, None]
        path_embed = self.path_encoder(self.path_vocabulary.flatten(1))[None]
        speed_embed = self.speed_encoder(self.speed_vocabulary)[None]
        lanes = (batch_size, -1, -1)
        paths = _Candidates(
            path_embed + context,
            path_embed.expand(lanes),
            torch.arange(path_embed.shape[1], device=ego_query.device).expand(batch_size, -1),
        )
        speeds = _Candidates(
            speed_embed + context,
            speed_embed.expand(lanes),
            torch.arange(speed_embed.shape[1], device=ego_query.device).expand(batch_size, -1),
        )
        layer_outputs = []
        for layer in self.layers[:-1]:
            paths, speeds, step = self._step(
                layer, paths, speeds, agents, map_keys, levels, frame_from_ego
            )
            layer_outputs.append(step.scores(layer.trajectory_score(step.query).squeeze(-1)))
        last = self.layers[-1]
        paths, speeds, step = self._step(
            last, paths, speeds, agents, map_keys, levels, frame_from_ego
        )
        embed = (paths.embed[:, :, None] + speeds.embed[:, None]).flatten(1, 2)
        trajectories = to_frame(self.trajectory_vocabulary[step.ids], frame_from_ego)
        query, collision_logits = last.refine(step.query, embed, trajectories, levels)
        logits = last.trajectory_score(query).squeeze(-1)
        layer_outputs.append(step.scores(logits))
        return PlannerOutputs(logits, trajectories, collision_logits, tuple(layer_outputs))

    def _step(
        self,
        layer: PlanningLayer,
        paths: _Candidates,
        speeds: _Candidates,
        agents: torch.Tensor,
        map_keys: torch.Tensor | None,
        levels: Sequence[SparseLevel],
        frame_from_ego: torch.Tensor,
    ) -> tuple[_Candidates, _Candidates, _LayerStep]:
        """Refine and score the paths and speed profiles of one layer and compose the best.

        Args:
            layer: The planning layer.
            paths: Path queries the layer gets.
            speeds: Speed profile queries the layer gets.
            agents: Agent keys ``(B, K, C)``.
            map_keys: Map keys ``(B, M, C)``, exactly for a planner with a map.
            levels: Sampled lidar stages.
            frame_from_ego: Transform of every frame from the ego frame to its point cloud frame.

        Returns:
            The kept paths and speed profiles and the trajectory queries they compose.
        """
        paths_in_frame = to_frame(self.path_vocabulary[paths.ids], frame_from_ego)
        paths = paths._replace(
            query=layer.refine_paths(paths, paths_in_frame, agents, map_keys, levels)
        )
        speeds = speeds._replace(query=layer.refine_speeds(speeds, agents, map_keys))
        path_logits = layer.path_score(paths.query).squeeze(-1)
        speed_logits = layer.speed_score(speeds.query).squeeze(-1)
        kept_paths = paths.keep(path_logits, layer.keep_paths)
        kept_speeds = speeds.keep(speed_logits, layer.keep_speeds)
        step = _LayerStep(
            path_logits=path_logits,
            path_ids=paths.ids,
            speed_logits=speed_logits,
            speed_ids=speeds.ids,
            query=(kept_paths.query[:, :, None] + kept_speeds.query[:, None]).flatten(1, 2),
            ids=(kept_paths.ids[:, :, None] * self.num_speeds + kept_speeds.ids[:, None]).flatten(
                1
            ),
        )
        return kept_paths, kept_speeds, step

    def prepare_for_export(self) -> VocabularyPlanner:
        """Return an export ready copy with explicit attention graphs."""
        planner = deepcopy(self)
        for module in planner.modules():
            if isinstance(module, ResidualAttention):
                module.attn = ExportableMultiheadAttention(module.attn)
        return planner.eval()

    def speed_targets(
        self, ego_future: torch.Tensor, ego_future_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean speed of every future step and whether it is known.

        Args:
            ego_future: Future ego positions in the ego frame ``(B, T, 2)``.
            ego_future_valid: Whether every position is known ``(B, T)``.

        Returns:
            Speeds ``(B, T)`` and their validity ``(B, T)``.
        """
        previous = torch.cat([torch.zeros_like(ego_future[:, :1]), ego_future[:, :-1]], dim=1)
        previous_valid = torch.cat(
            [torch.ones_like(ego_future_valid[:, :1]), ego_future_valid[:, :-1]], dim=1
        )
        speeds = (ego_future - previous).norm(dim=-1) / self.step_seconds
        return speeds, ego_future_valid & previous_valid

    def _vocabulary_loss(
        self, logits: torch.Tensor, distance: torch.Tensor, frames: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss on the nearest entry, the other entries weighted by their distance.

        Args:
            logits: Score logit of every candidate ``(B, N)``.
            distance: Distance of every candidate to the ground truth ``(B, N)``.
            frames: Frames with a ground truth ``(B,)``.

        Returns:
            The loss summed over the candidates, averaged over the frames.
        """
        target = distance.argmin(dim=1)
        one_hot = F.one_hot(target, logits.shape[1]).to(logits.dtype)
        weights = torch.where(one_hot > 0, torch.ones_like(distance), distance)
        weights = weights * frames[:, None].to(distance.dtype)
        return self.loss_cls(
            logits.float(),
            one_hot.float(),
            weights=weights.float(),
            avg_factor=float(reduce_mean_count(frames.sum())),
        )

    def loss(
        self,
        layers: Sequence[PlanLayerOutputs],
        collision_logits: torch.Tensor,
        collision_labels: torch.Tensor,
        ego_path: torch.Tensor,
        ego_path_valid: torch.Tensor,
        ego_future: torch.Tensor,
        ego_future_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Supervise the path, speed and trajectory scores of every layer and the collisions.

        The target of every score is the nearest candidate the layer received, the others count
        as negatives weighted by their distance, as in SparseDrive.

        Args:
            layers: Scores of every layer.
            collision_logits: Collision logits of the kept trajectories ``(B, N)``.
            collision_labels: Whether every kept trajectory collides with a ground truth agent
                ``(B, N)``.
            ego_path: Ground truth path in the ego frame at the vocabulary arc lengths
                ``(B, P, 2)``.
            ego_path_valid: Whether every path point is reached ``(B, P)``.
            ego_future: Future ego positions in the ego frame ``(B, T, 2)``.
            ego_future_valid: Whether every position is known ``(B, T)``.

        Returns:
            Loss terms of every layer.
        """
        speeds, speeds_valid = self.speed_targets(ego_future, ego_future_valid)
        path_frames = ego_path_valid.any(dim=1)
        speed_frames = speeds_valid.any(dim=1)
        path_mask = ego_path_valid.to(ego_path.dtype)
        speed_mask = speeds_valid.to(speeds.dtype)
        future_mask = ego_future_valid.to(ego_future.dtype)
        losses = {}
        for index, layer in enumerate(layers):
            paths = self.path_vocabulary[layer.path_ids]
            path_distance = ((paths - ego_path[:, None]).norm(dim=-1) * path_mask[:, None]).sum(
                -1
            ) / path_mask.sum(-1, keepdim=True).clamp(min=1.0)
            speed_distance = (
                (self.speed_vocabulary[layer.speed_ids] - speeds[:, None]).abs()
                * speed_mask[:, None]
            ).sum(-1) / speed_mask.sum(-1, keepdim=True).clamp(min=1.0)
            trajectories = self.trajectory_vocabulary[layer.trajectory_ids]
            trajectory_distance = (
                (trajectories - ego_future[:, None]).norm(dim=-1) * future_mask[:, None]
            ).sum(-1) / future_mask.sum(-1, keepdim=True).clamp(min=1.0)
            losses[f"loss_plan_path_{index}"] = self.loss_path_weight * self._vocabulary_loss(
                layer.path_logits, path_distance, path_frames
            )
            losses[f"loss_plan_speed_{index}"] = self.loss_speed_weight * self._vocabulary_loss(
                layer.speed_logits, speed_distance, speed_frames
            )
            losses[f"loss_plan_trajectory_{index}"] = (
                self.loss_trajectory_weight
                * self._vocabulary_loss(
                    layer.trajectory_logits, trajectory_distance, ego_future_valid.any(dim=1)
                )
            )
        # The mean over every frame and trajectory, zero for a batch without a supervised frame
        collision_loss = F.binary_cross_entropy_with_logits(
            collision_logits.float(), collision_labels.float(), reduction="sum"
        ) / max(collision_labels.numel(), 1)
        losses["loss_plan_collision"] = self.loss_collision_weight * collision_loss
        return losses

    def collisions(
        self,
        trajectories: torch.Tensor,
        agent_paths: torch.Tensor,
        agent_size: torch.Tensor,
        agent_yaw: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Whether every trajectory meets an agent box over the leading collision steps.

        Args:
            trajectories: Ego trajectories ``(B, N, T, 2)``.
            agent_paths: Agent box centers over the same steps ``(B, A, T, 2)``.
            agent_size: Agent length and width ``(B, A, 2)``.
            agent_yaw: Agent heading ``(B, A)``.
            agent_mask: Whether every agent counts at every step ``(B, A, T)``.

        Returns:
            Whether every trajectory collides ``(B, N)``.
        """
        steps = self.collision_steps
        return boxes_overlap(
            trajectories[:, :, :steps],
            self.vehicle.center_x,
            self.ego_box_size,
            agent_paths[:, :, :steps],
            agent_size,
            agent_yaw,
            agent_mask[:, :, :steps],
        )

    def select(
        self,
        logits: torch.Tensor,
        trajectories: torch.Tensor,
        agent_paths: torch.Tensor,
        agent_size: torch.Tensor,
        agent_yaw: torch.Tensor,
        agent_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Pick the best scored trajectory of every frame, colliding ones ranked below the others.

        Args:
            logits: Score logit of every kept trajectory ``(B, N)``.
            trajectories: The kept trajectories ``(B, N, T, 2)``.
            agent_paths: Most likely future box centers of the agents ``(B, A, T, 2)``.
            agent_size: Agent length and width ``(B, A, 2)``.
            agent_yaw: Agent heading ``(B, A)``.
            agent_scores: Detection score of every agent ``(B, A)``.

        Returns:
            The selected trajectory of every frame ``(B, T, 2)``.
        """
        blocking = agent_scores >= self.collision_score_threshold
        collides = self.collisions(
            trajectories,
            agent_paths,
            agent_size,
            agent_yaw,
            blocking[..., None].expand_as(agent_paths[..., 0]),
        )
        scores = logits.sigmoid()
        ranked = torch.where(collides, scores - 2.0, scores)
        lanes = torch.arange(logits.shape[0], device=logits.device)
        return trajectories[lanes, ranked.argmax(dim=-1)]
