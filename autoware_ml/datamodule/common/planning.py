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

"""Dataset task reading the ego future, status and path of a record."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.dataclasses.batch.planning import PlanningGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.common.motion import record_poses, records_at_offsets


class PlanningTask(BaseDatasetTask):
    """Read the future of the ego vehicle from the later records and its status from the earlier.

    The records of a scene are matched by timestamp as in :class:`MotionTask`. The ego future
    and status are expressed in the frame of the current record's point cloud. The ego path
    follows the ego over a longer horizon and samples it at fixed arc lengths, the route the ego
    takes regardless of its speed.
    """

    COLUMNS = (
        DatasetTableSchema.SCENARIO_ID.name,
        DatasetTableSchema.TIMESTAMP_SECONDS.name,
        DatasetTableSchema.LIDAR_FRAMES.name,
    )

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame,
        *,
        ego_steps: int,
        step_seconds: float,
        status_step_seconds: float,
        time_tolerance_seconds: float,
        path_distances: Sequence[float],
        path_seconds: float,
        path_step_seconds: float,
        path_min_spacing: float,
    ) -> None:
        """
        Initialize the task and index the future and past records of every record.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus.
          ego_steps: Number of future steps of the ego trajectory.
          step_seconds: Time between two future steps.
          status_step_seconds: Time between the past records the ego status is derived from.
          time_tolerance_seconds: Largest gap between a step time and the record serving it.
          path_distances: Increasing arc lengths the ego path is sampled at, in meters.
          path_seconds: Horizon of the ego positions the path is built from.
          path_step_seconds: Time between two of those positions.
          path_min_spacing: Smallest move the path follows, so the pose noise of a standing ego
            adds no length.
        """
        if ego_steps < 1 or min(step_seconds, status_step_seconds) <= 0:
            raise ValueError(
                f"The ego future needs a positive horizon and steps, got ego_steps={ego_steps}, "
                f"step_seconds={step_seconds}, status_step_seconds={status_step_seconds}."
            )
        distances = np.asarray(path_distances, dtype=np.float64)
        if distances.ndim != 1 or distances[0] <= 0 or bool((np.diff(distances) <= 0).any()):
            raise ValueError(f"path_distances must increase from above 0, got {path_distances}.")
        if not 0 < path_step_seconds <= path_seconds or path_min_spacing <= 0:
            raise ValueError(
                f"The ego path needs 0 < path_step_seconds <= path_seconds and a positive "
                f"path_min_spacing, got {path_step_seconds}, {path_seconds}, {path_min_spacing}."
            )
        steps = (step_seconds, status_step_seconds, path_step_seconds)
        if not 0 < time_tolerance_seconds < min(steps) / 2:
            raise ValueError(
                f"time_tolerance_seconds={time_tolerance_seconds} must be positive and below "
                "half a step, so one record never serves two steps."
            )
        super().__init__(database_root_path, dataset_records_dataframe)
        self.ego_steps = ego_steps
        self.step_seconds = step_seconds
        records = self.dataset_records_dataframe
        self.timestamps = records[DatasetTableSchema.TIMESTAMP_SECONDS.name].to_numpy()
        self.poses = record_poses(records)
        self.future_records = records_at_offsets(
            records,
            [step * step_seconds for step in range(1, ego_steps + 1)],
            time_tolerance_seconds,
        )
        self.past_records = records_at_offsets(
            records, [-status_step_seconds, -2 * status_step_seconds], time_tolerance_seconds
        )
        self.path_distances = distances
        self.path_min_spacing = path_min_spacing
        self.path_records = records_at_offsets(
            records,
            [
                step * path_step_seconds
                for step in range(1, round(path_seconds / path_step_seconds) + 1)
            ],
            time_tolerance_seconds,
        )

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the scenario, the time and the lidar frames of the records.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with the columns the ego future is built from.

        Raises:
          ValueError: If a column is missing.
        """
        missing = [
            column for column in self.COLUMNS if column not in dataset_records_dataframe.columns
        ]
        if missing:
            raise ValueError(f"{self} needs the columns {list(self.COLUMNS)}, missing {missing}.")
        return dataset_records_dataframe.select(list(self.COLUMNS))

    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Build the ego future, status and path of one record.

        Args:
          idx: Index of the record.

        Returns:
          ModelGTSample: Sample holding the ego future of the record.
        """
        to_current = np.linalg.inv(self.poses[idx])
        ego_positions = np.zeros((self.ego_steps, 3), dtype=np.float64)
        ego_valid = self.future_records[idx] >= 0
        for step in np.flatnonzero(ego_valid):
            ego_positions[step] = (to_current @ self.poses[self.future_records[idx, step]])[:3, 3]

        ego_status, ego_status_valid = self.ego_status(idx, to_current)
        # The path follows the records up to the first missing one
        path_records = self.path_records[idx]
        known = path_records[: int(np.cumprod(path_records >= 0).sum())]
        path_positions = np.stack(
            [np.zeros(3)] + [(to_current @ self.poses[record])[:3, 3] for record in known]
        )
        ego_path, ego_path_valid = resample_path(
            path_positions, self.path_distances, self.path_min_spacing
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
            planning_gt_sample=PlanningGTSample(
                ego_future_positions=torch.from_numpy(ego_positions).float(),
                ego_future_valid=torch.from_numpy(ego_valid),
                ego_status=torch.from_numpy(ego_status).float(),
                ego_status_valid=ego_status_valid,
                ego_path=torch.from_numpy(ego_path).float(),
                ego_path_valid=torch.from_numpy(ego_path_valid),
            ),
        )

    def ego_status(
        self, idx: int, to_current: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """
        Derive the ego velocity, acceleration and yaw rate from the two past records.

        Args:
          idx: Index of the record.
          to_current: Transform from the global frame to the frame of the record.

        Returns:
          tuple: Velocity (x, y), acceleration (x, y) and yaw rate in the frame of the record,
            and whether both past records exist.
        """
        previous, before = self.past_records[idx]
        if previous < 0 or before < 0:
            return np.zeros(5), False
        previous_pose = to_current @ self.poses[previous]
        before_pose = to_current @ self.poses[before]
        gap = self.timestamps[idx] - self.timestamps[previous]
        earlier_gap = self.timestamps[previous] - self.timestamps[before]
        velocity = -previous_pose[:2, 3] / gap
        earlier_velocity = (previous_pose[:2, 3] - before_pose[:2, 3]) / earlier_gap
        acceleration = (velocity - earlier_velocity) / ((gap + earlier_gap) / 2)
        yaw_rate = -np.arctan2(previous_pose[1, 0], previous_pose[0, 0]) / gap
        return np.concatenate([velocity, acceleration, [yaw_rate]]), True


def resample_path(
    positions: npt.NDArray[np.float64],
    distances: npt.NDArray[np.float64],
    min_spacing: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """
    Sample the path through a sequence of positions at fixed arc lengths from its first one.

    Args:
      positions: Positions ``(S, 3)``, the first one the start of the path.
      distances: Increasing arc lengths to sample at.
      min_spacing: Smallest step the path follows, shorter moves from the last followed
        position are skipped.

    Returns:
      tuple: Positions ``(P, 3)`` at the arc lengths and whether the path reaches them.
    """
    followed = [positions[0]]
    for position in positions[1:]:
        if np.linalg.norm(position[:2] - followed[-1][:2]) >= min_spacing:
            followed.append(position)
    points = np.stack(followed)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))])
    path = np.stack([np.interp(distances, arc, points[:, axis]) for axis in range(3)], axis=-1)
    return path, distances <= arc[-1]
