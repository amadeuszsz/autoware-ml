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

"""nuScenes family dataset."""

from __future__ import annotations

from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.datamodule.base import Dataset
from autoware_ml.datamodule.samples.meta import FrameMeta


class NuscenesDataset(Dataset):
    """Dataset over nuScenes records. Every task is optional and driven by the configured
    transform pipeline, one class serves detection, segmentation, multiview, and calibration
    workloads.

    The scene token is the nuScenes scene name. nuScenes ships no lanelet maps, so map based
    metric filters skip its scenes through their availability checks.
    """

    def build_meta(self, record: DatasetRecord) -> FrameMeta:
        """Build the nuScenes frame metadata of one record.

        Args:
            record: Dataset record of the sample.

        Returns:
            Frame metadata of the sample.
        """
        keyframe = record.lidar_frames[0]
        return FrameMeta(
            sample_id=record.sample_id,
            scene_token=record.scenario_id,
            timestamp_seconds=record.timestamp_seconds,
            ego2global=keyframe.lidar_frame_ego_pose_to_global_matrix,
            location=record.location,
            vehicle_type=record.vehicle_type,
            prev_exists=record.sample_index > 0,
        )
