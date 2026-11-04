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


"""K-means path and speed vocabularies of the planner from the training ego futures."""

from __future__ import annotations

import logging

import numpy as np

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.anchors.kmeans import cluster_modes
from autoware_ml.datamodule.common.planning import PlanningTask
from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.types.tasks import TaskType

logger = logging.getLogger(__name__)


def collect_ego_futures(
    planning_task: PlanningTask, record_stride: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Gather the complete ego paths and the speed profiles of the complete ego futures.

    Args:
        planning_task: Ego futures of the records.
        record_stride: Every how many records one is read.

    Returns:
        Ego paths ``(P, 2)`` and ego speed profiles ``(T,)``.
    """
    paths: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    for record in range(0, len(planning_task.dataset_records_dataframe), record_stride):
        futures = planning_task.get_data_sample(record).planning_gt_sample
        if bool(futures.ego_path_valid.all()):
            paths.append(futures.ego_path[:, :2].numpy())
        if bool(futures.ego_future_valid.all()):
            ego = futures.ego_future_positions[:, :2].numpy()
            steps = np.diff(np.concatenate([np.zeros((1, 2)), ego]), axis=0)
            speeds.append(np.linalg.norm(steps, axis=1) / planning_task.step_seconds)
    return paths, speeds


class PlannerVocabularyRecipe(AnchorRecipe):
    """K-means modes of the complete ego paths and of the complete ego speed profiles."""

    name = "planner_vocabulary"
    task_types = (TaskType.PLANNING,)

    def __init__(
        self,
        *,
        cache_path: str,
        num_paths: int,
        num_speeds: int,
        record_stride: int,
        iterations: int,
        seed: int,
    ) -> None:
        """Initialize the recipe.

        Args:
            cache_path: Directory of the cached anchors.
            num_paths: Paths of the vocabulary.
            num_speeds: Speed profiles of the vocabulary.
            record_stride: Every how many training records one is read.
            iterations: Number of k-means iterations.
            seed: Seed of the initial centroids.
        """
        super().__init__(cache_path)
        self.num_paths = num_paths
        self.num_speeds = num_speeds
        self.record_stride = record_stride
        self.iterations = iterations
        self.seed = seed

    def hash_repr(self) -> str:
        """Canonical text of the recipe settings."""
        return (
            f"{type(self).__name__}(num_paths={self.num_paths}, num_speeds={self.num_speeds}, "
            f"record_stride={self.record_stride}, iterations={self.iterations}, seed={self.seed})"
        )

    def build(self, split: TrainingSplit) -> dict[str, np.ndarray]:
        """Cluster the complete training ego paths and speed profiles.

        Args:
            split: Training sources and their records.

        Returns:
            ``paths`` ``(num_paths, P, 2)`` and ``speeds`` ``(num_speeds, ego_steps)``.
        """
        paths: list[np.ndarray] = []
        speeds: list[np.ndarray] = []
        for index, source in enumerate(split.sources):
            source_paths, source_speeds = collect_ego_futures(
                split.build_task(index, TaskType.PLANNING), self.record_stride
            )
            paths.extend(source_paths)
            speeds.extend(source_speeds)
            logger.info(
                f"{source.database.version}: {len(source_paths)} complete ego paths, "
                f"{len(source_speeds)} complete ego futures"
            )
        return {
            "paths": cluster_modes(paths, self.num_paths, self.iterations, self.seed, "ego paths"),
            "speeds": cluster_modes(
                speeds, self.num_speeds, self.iterations, self.seed, "ego speed profiles"
            ),
        }
