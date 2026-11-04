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

"""Small vocabulary planners and lidar stages for the motion planner tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.metrics.geometry.reachability import VehicleGeometry
from autoware_ml.models.detection3d.heads.sparse4d import SparseLevelGrid
from autoware_ml.models.detection3d.ptv3 import SparseStage
from autoware_ml.models.planning.vocabulary_planner import VocabularyPlanner

PATH_DISTANCES = (1.0, 2.0, 4.0)
STEP_SECONDS = 0.5
POINT_CLOUD_RANGE = (-20.0, -20.0, -2.0, 20.0, 20.0, 2.0)
# A 4 m x 2 m body whose center lies 1 m ahead of the rear axle
TEST_VEHICLE = VehicleGeometry(
    wheel_base=2.5,
    front_overhang=0.5,
    rear_overhang=1.0,
    wheel_tread=1.8,
    left_overhang=0.1,
    right_overhang=0.1,
    max_steer_angle=0.5,
)


def vocabulary_recipe(num_paths: int = 6, num_speeds: int = 4) -> FixedAnchorRecipe:
    """Paths bending from right to left and speed profiles from 0 to 6 m/s, two steps each."""
    curvature = np.linspace(-0.2, 0.2, num_paths)[:, None]
    distances = np.asarray(PATH_DISTANCES)[None]
    paths = np.stack([distances * np.ones_like(curvature), curvature * distances**2], axis=-1)
    speeds = np.repeat(np.linspace(0.0, 6.0, num_speeds)[:, None], 2, axis=1)
    return FixedAnchorRecipe({"paths": paths, "speeds": speeds})


def build_planner(tmp_path: Path, embed_dims: int, **overrides: object) -> VocabularyPlanner:
    """A planner over the test vocabulary, one sampled stage and two layers."""
    settings = {
        "embed_dims": embed_dims,
        "anchor_recipe": vocabulary_recipe(),
        "path_distances": PATH_DISTANCES,
        "num_paths": 6,
        "num_speeds": 4,
        "ego_steps": 2,
        "step_seconds": STEP_SECONDS,
        "keep": ((4, 3), (2, 2)),
        "keypoint_heights": (0.0, 1.0),
        "vehicle": TEST_VEHICLE,
        "collision_steps": 2,
        "point_cloud_range": POINT_CLOUD_RANGE,
        "grid_size": 1.0,
        "feature_strides": (1,),
        "num_heads": 4,
        "num_groups": 4,
    }
    return fill_anchors(VocabularyPlanner(**(settings | overrides)))


def build_stages(embed_dims: int, batch_size: int, seed: int = 0) -> list[SparseStage]:
    """One fused stage of random voxels around the ego for every sample."""
    grid = SparseLevelGrid(POINT_CLOUD_RANGE, 1.0, 1)
    generator = torch.Generator().manual_seed(seed)
    num_voxels = 200
    grid_coord = torch.stack(
        [
            torch.randint(0, size, (batch_size * num_voxels,), generator=generator)
            for size in grid.grid_dims
        ],
        dim=1,
    )
    feat = torch.randn(batch_size * num_voxels, embed_dims, generator=generator)
    offset = torch.arange(1, batch_size + 1) * num_voxels
    return [SparseStage(feat, grid_coord, offset)]
