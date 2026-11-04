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


"""K-means polyline anchors of the online map head from the training map elements."""

from __future__ import annotations

import logging

import numpy as np
import torch

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.anchors.kmeans import kmeans
from autoware_ml.dataclasses.batch.online_map import vectorize
from autoware_ml.datamodule.common.online_map import OnlineMapTargets
from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.types.tasks import TaskType

logger = logging.getLogger(__name__)


def collect_centroids(map_targets: OnlineMapTargets, record_stride: int) -> np.ndarray:
    """Centroids of the elements in the mapped region of every read record.

    Args:
        map_targets: Map elements of the records.
        record_stride: Every how many records one is read.

    Returns:
        Centroids ``(E, 2)``.
    """
    centroids = []
    for record in range(0, len(map_targets.osm_paths), record_stride):
        sample = map_targets.get_gt_sample(record)
        points, _, _ = vectorize(sample.elements, np.eye(4), sample.spec)
        centroids.append(points.mean(axis=1))
    return np.concatenate(centroids)


def straight_anchors(centroids: np.ndarray, num_points: int, length: float) -> np.ndarray:
    """Straight polylines along x through every centroid.

    Args:
        centroids: Anchor centers ``(A, 2)``.
        num_points: Points of every polyline.
        length: Length of every polyline in meters.

    Returns:
        Anchors ``(A, 2 * num_points)`` in the flattened polyline layout.
    """
    offsets = np.zeros((num_points, 2))
    offsets[:, 0] = np.linspace(-length / 2, length / 2, num_points)
    return (centroids[:, None] + offsets[None]).reshape(len(centroids), -1)


class PolylineAnchorRecipe(AnchorRecipe):
    """Straight polylines through the k-means centroids of the map elements.

    Every anchor is a segment along the frame x axis through its centroid (Sun et al.,
    SparseDrive, 2024). The map elements come from the online map targets of the motion task.
    """

    name = "online_map_polylines"
    task_types = (TaskType.MOTION,)

    def __init__(
        self,
        *,
        cache_path: str,
        num_anchors: int,
        num_points: int,
        record_stride: int,
        segment_length: float,
        iterations: int,
        seed: int,
    ) -> None:
        """Initialize the recipe.

        Args:
            cache_path: Directory of the cached anchors.
            num_anchors: Number of anchors.
            num_points: Points of every polyline.
            record_stride: Every how many training records one is read.
            segment_length: Length of every anchor polyline in meters.
            iterations: Number of k-means iterations.
            seed: Seed of the initial centroids.
        """
        super().__init__(cache_path)
        self.num_anchors = num_anchors
        self.num_points = num_points
        self.record_stride = record_stride
        self.segment_length = segment_length
        self.iterations = iterations
        self.seed = seed

    def hash_repr(self) -> str:
        """Canonical text of the recipe settings."""
        return (
            f"{type(self).__name__}(num_anchors={self.num_anchors}, num_points={self.num_points}, "
            f"record_stride={self.record_stride}, segment_length={self.segment_length}, "
            f"iterations={self.iterations}, seed={self.seed})"
        )

    def build(self, split: TrainingSplit) -> dict[str, np.ndarray]:
        """Cluster the training map element centroids into straight anchors.

        Args:
            split: Training sources and their records.

        Returns:
            ``anchors`` ``(num_anchors, 2 * num_points)``.

        Raises:
            ValueError: If there are fewer map elements than anchors.
        """
        centroids = []
        for index, source in enumerate(split.sources):
            motion_task = split.build_task(index, TaskType.MOTION)
            source_centroids = collect_centroids(motion_task.online_map, self.record_stride)
            logger.info(f"{source.database.version}: {len(source_centroids):,} map elements")
            centroids.append(source_centroids)
        points = torch.from_numpy(np.concatenate(centroids)).float()
        if len(points) < self.num_anchors:
            raise ValueError(f"{len(points)} map elements cannot seed {self.num_anchors} anchors.")
        clusters, _ = kmeans(points, self.num_anchors, self.iterations, self.seed)
        anchors = straight_anchors(clusters.numpy(), self.num_points, self.segment_length)
        return {"anchors": anchors}
