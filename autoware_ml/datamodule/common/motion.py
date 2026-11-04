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

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema, key_frame_flags
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.dataclasses.batch.motion import MotionGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.common.detection3d import box_instance_ids, flatten_box_fields

logger = logging.getLogger(__name__)


class MotionTask(BaseDatasetTask):
    """Read the futures of the annotated agents from the later records.

    The records of a scene are matched by timestamp: the record at a future step is the one of
    the same scenario closest to the step time, within the tolerance. A step without such a
    record, or an agent missing from it, is invalid. Futures are expressed in the frame of the
    current record's point cloud.
    """

    COLUMNS = (
        DatasetTableSchema.SCENARIO_ID.name,
        DatasetTableSchema.TIMESTAMP_SECONDS.name,
        DatasetTableSchema.BOXES_3D.name,
        DatasetTableSchema.LIDAR_FRAMES.name,
    )

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame,
        *,
        agent_steps: int,
        step_seconds: float,
        time_tolerance_seconds: float,
        max_speed: float,
    ) -> None:
        """
        Initialize the task and index the future records of every record.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus.
          agent_steps: Number of future steps of the agent trajectories.
          step_seconds: Time between two future steps.
          time_tolerance_seconds: Largest gap between a step time and the record serving it.
          max_speed: Largest plausible agent speed in m/s. A future position farther from the
            previous known one ends the trajectory there: the instance id then names another
            object, a pseudo label id switch.

        Raises:
          ValueError: If a setting is out of range, or the key frames come less often than the
            steps.
        """
        if agent_steps < 1 or step_seconds <= 0:
            raise ValueError(
                f"Futures need positive horizons and steps, got agent_steps={agent_steps}, "
                f"step_seconds={step_seconds}."
            )
        if max_speed <= 0:
            raise ValueError(f"max_speed must be positive, got {max_speed}.")
        if not 0 < time_tolerance_seconds < step_seconds / 2:
            raise ValueError(
                f"time_tolerance_seconds={time_tolerance_seconds} must be positive and below "
                "half a step, so one record never serves two steps."
            )
        super().__init__(database_root_path, dataset_records_dataframe)
        self.agent_steps = agent_steps
        self.step_seconds = step_seconds
        self.max_speed = max_speed
        records = self.dataset_records_dataframe
        # Only key frames carry boxes, so every future step of every scenario needs one
        intervals = key_frame_intervals(records)
        if len(records) and not intervals:
            raise ValueError("No scenario of the records holds two key frames to build futures.")
        sparse = {
            scenario: interval
            for scenario, interval in intervals.items()
            if interval > step_seconds + time_tolerance_seconds
        }
        if sparse:
            raise ValueError(
                f"The key frames of scenarios {sorted(sparse)[:5]} come every up to "
                f"{max(sparse.values()):.2f} s, the futures need one every {step_seconds} s."
            )
        self.poses = record_poses(records)
        self.future_records = records_at_offsets(
            records,
            [step * step_seconds for step in range(1, agent_steps + 1)],
            time_tolerance_seconds,
        )
        self.box_offsets, self.box_instance_ids, self.box_centers = flatten_box_centers(records)

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the scenario, the time, the boxes and the lidar frames of the records.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with the columns the futures are built from.

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
        Build the agent futures of one record.

        Args:
          idx: Index of the record.

        Returns:
          ModelGTSample: Sample holding the agent futures of the record.
        """
        to_current = np.linalg.inv(self.poses[idx])
        agent_ids = self.box_instance_ids[self.box_offsets[idx] : self.box_offsets[idx + 1]].copy()
        agent_positions = np.zeros((len(agent_ids), self.agent_steps, 3), dtype=np.float64)
        agent_valid = np.zeros((len(agent_ids), self.agent_steps), dtype=bool)
        for step in range(self.agent_steps):
            future = self.future_records[idx, step]
            if future < 0:
                continue
            begin, end = self.box_offsets[future], self.box_offsets[future + 1]
            found, rows = match_ids(agent_ids, self.box_instance_ids[begin:end])
            transform = to_current @ self.poses[future]
            centers = self.box_centers[begin:end][rows].astype(np.float64)
            agent_positions[found, step] = centers @ transform[:3, :3].T + transform[:3, 3]
            agent_valid[found, step] = True
        current = self.box_centers[self.box_offsets[idx] : self.box_offsets[idx + 1]]
        agent_valid &= plausible_steps(
            current.astype(np.float64),
            agent_positions,
            agent_valid,
            self.max_speed * self.step_seconds,
        )

        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
            motion_gt_sample=MotionGTSample(
                agent_instance_ids=torch.from_numpy(agent_ids),
                agent_future_positions=torch.from_numpy(agent_positions).float(),
                agent_future_valid=torch.from_numpy(agent_valid),
            ),
        )


def record_poses(records: pl.DataFrame) -> npt.NDArray[np.float64]:
    """
    Transform of every record from the frame of its point cloud to the global frame.

    Args:
      records: Records with their lidar frames.

    Returns:
      Ego pose composed with the lidar mounting of the first lidar frame of every record.
    """
    first_frame = pl.col(DatasetTableSchema.LIDAR_FRAMES.name).list.first().struct
    poses = records.select(
        first_frame.field(LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name),
        first_frame.field(LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name),
    )
    ego_to_global = poses.to_series(0).to_numpy().astype(np.float64)
    sensor_to_ego = poses.to_series(1).to_numpy().astype(np.float64)
    return ego_to_global @ sensor_to_ego


def key_frame_intervals(records: pl.DataFrame) -> dict[str, float]:
    """
    Median time between consecutive key frames of every scenario.

    Args:
      records: Records with their scenario, timestamp and lidar frames.

    Returns:
      The median gap in seconds of every scenario holding two key frames.
    """
    scenario = DatasetTableSchema.SCENARIO_ID.name
    time = DatasetTableSchema.TIMESTAMP_SECONDS.name
    gaps = (
        records.select(scenario, time)
        .filter(key_frame_flags(records))
        .sort(scenario, time)
        .with_columns(pl.col(time).diff().over(scenario).alias("gap"))
        .drop_nulls("gap")
        .group_by(scenario)
        .agg(pl.col("gap").median())
    )
    return dict(zip(gaps[scenario].to_list(), gaps["gap"].to_list(), strict=True))


def records_at_offsets(
    records: pl.DataFrame, offsets: Sequence[float], tolerance: float
) -> npt.NDArray[np.int64]:
    """
    Index of the record of the same scenario closest to every record's time plus each offset.

    Args:
      records: Records with their scenario and timestamp.
      offsets: Time offsets in seconds.
      tolerance: Largest gap between the offset time and the matched record.

    Returns:
      One record index per record and offset, -1 where no record lies within the tolerance.
    """
    scenario = DatasetTableSchema.SCENARIO_ID.name
    time = DatasetTableSchema.TIMESTAMP_SECONDS.name
    table = records.select(scenario, time).with_row_index("record")
    candidates = table.rename({"record": "found"}).sort(time)
    columns = []
    for offset in offsets:
        queries = table.with_columns(pl.col(time) + offset).sort(time)
        found = queries.join_asof(
            candidates,
            on=time,
            by=scenario,
            strategy="nearest",
            tolerance=tolerance,
            # Both sides are sorted by time above
            check_sortedness=False,
        )
        columns.append(found.sort("record")["found"].fill_null(-1).to_numpy().astype(np.int64))
    return np.stack(columns, axis=1)


def flatten_box_centers(
    records: pl.DataFrame,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """
    Gather the instance identity and the center of every box of every record.

    Args:
      records: Records with their boxes.

    Returns:
      tuple: Offset of every record's boxes (one more than the records), the identity and the
        center of every box, in record order.
    """
    instance_id = Box3DDatasetSchema.BOX3D_INSTANCE_ID.name
    params = Box3DDatasetSchema.BOX3D_PARAMS.name
    offsets, box_fields = flatten_box_fields(records, [instance_id, params])
    centers = np.ascontiguousarray(box_fields[params].to_numpy()[:, :3], dtype=np.float32)
    return offsets, box_instance_ids(box_fields[instance_id], offsets), centers


def plausible_steps(
    current: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    max_step_distance: float,
) -> npt.NDArray[np.bool_]:
    """
    Cut every trajectory at its first jump beyond the distance an agent can cover.

    Args:
      current: Current position of every agent ``(A, 3)``.
      positions: Future positions ``(A, T, 3)``.
      valid: Whether a future position is known ``(A, T)``.
      max_step_distance: Largest BEV distance covered in one step.

    Returns:
      The steps before the first implausible jump of every agent ``(A, T)``.
    """
    plausible = np.ones_like(valid)
    last = current[:, :2].copy()
    last_step = np.full(len(current), -1)
    for step in range(valid.shape[1]):
        distance = np.linalg.norm(positions[:, step, :2] - last, axis=1)
        jump = valid[:, step] & (distance > max_step_distance * (step - last_step))
        plausible[:, step:] &= ~jump[:, None]
        known = valid[:, step] & plausible[:, step]
        last[known] = positions[known, step, :2]
        last_step[known] = step
    return plausible


def match_ids(
    queries: npt.NDArray[np.int64], candidates: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """
    Find every query identity among the candidates.

    Args:
      queries: Identities to look up.
      candidates: Identities to search.

    Returns:
      tuple: Whether every query was found, and the candidate row of every found query.
    """
    if len(candidates) == 0:
        return np.zeros(len(queries), dtype=bool), np.zeros(0, dtype=np.int64)
    order = np.argsort(candidates)
    positions = np.clip(np.searchsorted(candidates[order], queries), 0, len(candidates) - 1)
    found = candidates[order][positions] == queries
    return found, order[positions[found]]
