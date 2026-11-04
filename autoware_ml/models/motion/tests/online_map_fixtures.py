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

"""A small polyline map head over the 8 m test grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.models.motion.online_map_head import OnlineMapHead

POINT_CLOUD_RANGE = [0.0, 0.0, -2.0, 8.0, 8.0, 2.0]
NUM_POINTS = 4


def build_head(tmp_path: Path, **overrides: object) -> OnlineMapHead:
    """A small map head over the 8 m test grid with straight anchors along x."""
    rng = np.random.default_rng(0)
    starts = rng.uniform(1.0, 5.0, size=(6, 1, 2))
    anchors = starts + np.stack([np.linspace(0.0, 2.0, NUM_POINTS), np.zeros(NUM_POINTS)], -1)
    kwargs: dict[str, object] = {
        "num_classes": 3,
        "embed_dims": 32,
        "anchor_recipe": FixedAnchorRecipe({"anchors": anchors.reshape(6, -1)}),
        "num_anchor": 6,
        "num_points": NUM_POINTS,
        "roi": (0.0, 0.0, 8.0, 8.0),
        "point_cloud_range": POINT_CLOUD_RANGE,
        "grid_size": 1.0,
        "feature_strides": (1,),
        "num_temp_instances": 2,
        "num_decoder": 2,
        "num_heads": 4,
        "num_groups": 4,
        "feedforward_channels": 64,
        "num_output": 5,
    }
    kwargs.update(overrides)
    return fill_anchors(OnlineMapHead(**kwargs))
